#pragma once

#include <vector>
#include "transform.h"

using Sample = Vector2;

struct ReservoirSample {
    int light_id;
    PointAndNormal x;
    Real p_hat;
    Real w;

    ReservoirSample(): light_id(-1), x(PointAndNormal()), w(0), p_hat(0) {};
    ReservoirSample(int light_id, PointAndNormal x, Real w, Real p_hat) : light_id(light_id), x(x), w(w), p_hat(p_hat) {}
};

struct Reservoir {
    int M;
    std::optional<PathVertex> org_vertex;
    ReservoirSample y;
    Real W;
    Real w_sum;

    Reservoir(): M(0), org_vertex({}), y(ReservoirSample()), W(0), w_sum(0) {};
    Reservoir(int M, PathVertex org_vertex, ReservoirSample y, Real W, Real w_sum) : M(M), org_vertex(org_vertex), y(y), W(W), w_sum(w_sum) {}

    // TODO: If neighbor sample is on another object, reject it
    void update(const ReservoirSample& candidate, pcg32_state& rng){
        w_sum += candidate.w;
        M += 1;
        Real random_cdf = next_pcg32_real<Real>(rng);
        
        if (M == 1 || (random_cdf < candidate.w && candidate.light_id != -1)){
            y = candidate;
            W = W * (candidate.p_hat / M);
        }
    }
};

struct ReservoirBuffer {
    ReservoirBuffer() {};
    ReservoirBuffer(int width, int height): width(width), height(height) {
        data.resize(width * height);
        memset(data.data(), 0, sizeof(Reservoir) * data.size());
    }

    Reservoir &operator()(int x, int y) {
        return data[y * width + x];
    }

    const Reservoir &operator()(int x, int y) const {
        return data[y * width + x];
    }

    Reservoir &operator()(int x) {
        return data[x];
    }

    const Reservoir &operator()(int x) const {
        return data[x];
    }

    int width;
    int height;
    std::vector<Reservoir> data;
};

// Algorithm 3: Streaming RIS using weighted reservoir sampling
void resample_importance_sampling(int M, const Scene &scene, Reservoir &r, PathVertex q, Ray ray, pcg32_state &rng) {
    // Sample M light paths candidates
    for (int i=1; i<M+1; i++){
        int light_id = sample_light(scene, next_pcg32_real<Real>(rng));
        Light light = scene.lights[light_id];
        Vector2 rng_uv_params(next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng));
        Real rng_w_param = next_pcg32_real<Real>(rng);
        PointAndNormal x = sample_point_on_light(light, q.position, rng_uv_params, rng_w_param, scene);
        Vector3 dir_to_light = normalize(x.position - q.position);

        // Calculate p
        Real p = light_pmf(scene, light_id) * pdf_point_on_light(light, x, q.position, scene);
        
        // Calculate p_hat
        Spectrum Le = emission(light, -dir_to_light, Real(1), x, scene);
        Spectrum rho = eval(scene.materials[q.material_id], -ray.dir, dir_to_light, q, scene.texture_pool);
        Real G = fabs(dot(dir_to_light, x.normal)) / distance_squared(x.position, q.position);
        Real p_hat = luminance(Le * rho) * G;

        // Update reservoir
        Real w_x = p_hat / p;
        r.update(ReservoirSample(light_id, x, w_x, p_hat), rng);
    }
};

// Init algorithm to init the reservoir buffer
void init_reservoir(const Scene &scene, ReservoirBuffer &G_buffer, int x, int y, pcg32_state &rng){
    Reservoir& r = G_buffer(x, y);
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    
    // Check whether the ray hits the any objects
    if (!vertex_){
        // Do something here, deal with environment map for example
        return;
    }
    
    PathVertex vertex = *vertex_;
    r.org_vertex = vertex;
    resample_importance_sampling(scene.options.reservoir_size, scene, r, vertex, ray, rng);
    
    // check visibility for the light source of reservoir sample
    Ray shadow_ray{vertex.position, normalize(r.y.x.position - vertex.position), get_shadow_epsilon(scene), 
                    (1-get_shadow_epsilon(scene)) * distance(r.y.x.position, vertex.position)};
    if (occluded(scene, shadow_ray)){
        r.W = 0;
    }
};

// Randomly select neighbor pixel to merge their reservoirs with current pixel's reservoir
void spatial_reuse(const Scene &scene, ReservoirBuffer& G_buffer, int x, int y, pcg32_state &rng){
    Reservoir& current_reservoir = G_buffer(x, y);
    if (!current_reservoir.org_vertex){
        return;
    }

    int max_radius = scene.options.max_radius;
    Vector3 cam_org = xform_point(scene.camera.cam_to_world, Vector3{0, 0, 0});

    // Sample a neighbor
    Real theta = next_pcg32_real<Real>(rng) * 2 * M_PI;
    Real radius = next_pcg32_real<Real>(rng) * Real(max_radius);
    int x_ = x + radius * cos(theta);
    int y_ = y + radius * sin(theta);
    x_ = max(0, min(x_, scene.camera.width - 1));
    y_ = max(0, min(y_, scene.camera.height - 1));
    Reservoir neighbor_reservoir = G_buffer(x_, y_);
    if (!neighbor_reservoir.org_vertex){
        return;
    }
    
    // Heuristic rejection
    PathVertex current_pv = *current_reservoir.org_vertex, neighbor_pv = *neighbor_reservoir.org_vertex;
    Real cam_q_dis = distance(cam_org, current_pv.position);
    Real cam_q_prime_dis = distance(cam_org, neighbor_pv.position);
    Real depth_diff = fabs(cam_q_dis - cam_q_prime_dis);
    Real angle_q_q_prime = std::acos(dot(neighbor_pv.geometric_normal, current_pv.geometric_normal));
    if (depth_diff / cam_q_dis >= 0.1 || angle_q_q_prime >= 10 / 180 * c_PI){
        return;
    }

    // Combine reservoirs
    current_reservoir.update(neighbor_reservoir.y, rng);
};

// Compute radiance for each pixel;
Spectrum compute_radiance(const Scene &scene, ReservoirBuffer& G_buffer, int x, int y, pcg32_state &rng){
    Spectrum radiance = make_zero_spectrum();

    Reservoir& r = G_buffer(x, y);
    if (!r.org_vertex){
        return radiance;
    }

    Vector3 cam_org = xform_point(scene.camera.cam_to_world, Vector3{0, 0, 0});
    PathVertex org_vertex = *r.org_vertex;
    Vector3 input_dir = normalize(cam_org - org_vertex.position);

    // If point is light, account for the emission
    if (is_light(scene.shapes[org_vertex.shape_id])){
        radiance += emission(org_vertex, -input_dir, scene);
    }
    // If invalid light sampled from RIS, return radiance
    if (r.y.light_id == -1){
        return radiance;
    }

    Light light = scene.lights[r.y.light_id];
    PointAndNormal point_on_light = r.y.x;
    Vector3 dir_light = normalize(point_on_light.position - org_vertex.position);
    Spectrum Le = emission(light, -dir_light, Real(1), point_on_light, scene);
    Spectrum rho = eval(scene.materials[org_vertex.material_id], input_dir, dir_light, org_vertex, scene.texture_pool);
    // Real G = fabs(dot(-dir_light, point_on_light.normal)) / distance_squared(point_on_light.position, org_vertex.position);
    // Real p1 = r.y.p_hat / r.y.w;
    Real p2 = pdf_sample_bsdf(scene.materials[org_vertex.material_id], input_dir, dir_light, org_vertex, scene.texture_pool);

    // If sampled light pdf is less than or equal to 0, return radiance
    if (p2 <= 0){
        return radiance;
    }
    radiance += rho * Le / p2;
    return radiance;
}

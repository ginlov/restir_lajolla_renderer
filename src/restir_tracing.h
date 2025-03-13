#pragma once

#include <vector>
#include "transform.h"

using Sample = Vector2;

struct ReservoirSample {
    int light_id;
    PointAndNormal x;
    Real p_hat;
    Real w;

    ReservoirSample(): light_id(-1), x(PointAndNormal()), p_hat(0), w(0) {};
    ReservoirSample(int light_id, PointAndNormal x, Real p_hat, Real w) : light_id(light_id), x(x), p_hat(p_hat), w(w) {};
};

struct Reservoir {
    int M;
    std::optional<PathVertex> org_vertex;
    ReservoirSample y;
    Real W;
    Real w_sum;
    Vector2 screen_pos;

    Reservoir(): M(0), org_vertex({}), y(ReservoirSample()), W(0), w_sum(0), screen_pos(Vector2{Real(0), Real(0)}) {};
    Reservoir(int M, PathVertex org_vertex, ReservoirSample y, Real W, Real w_sum, Vector2 screen_pos) : M(M), org_vertex(org_vertex), y(y), W(W), w_sum(w_sum), screen_pos(screen_pos) {}

    // TODO: If neighbor sample is on another object, reject it
    void update(const ReservoirSample& candidate, pcg32_state& rng, int num_candidates=1){
        if (candidate.light_id == -1){
            return;
        }
        w_sum += candidate.w;
        M += num_candidates;
        Real random_cdf = next_pcg32_real<Real>(rng);
        if (random_cdf < candidate.w / w_sum){
            y = candidate;
            W = w_sum / (M * y.p_hat);
        }
    }
};

struct ReservoirBuffer {
    ReservoirBuffer() {};
    ReservoirBuffer(int width, int height): width(width), height(height) {
        data.resize(width * height, Reservoir());
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

// Compute p_hat
Real compute_p_hat(const Scene& scene, Reservoir& reservoir, int light_id, PointAndNormal x, pcg32_state &rng) {
    if (!reservoir.org_vertex){
        // If the vertex is invalid, return 0
        return Real(0);
    }

    PathVertex vertex = *reservoir.org_vertex;
    Ray in_ray = sample_primary(scene.camera, reservoir.screen_pos);
    Ray shadow_ray{vertex.position, normalize(x.position - vertex.position), get_shadow_epsilon(scene), 
        (1-get_shadow_epsilon(scene)) * distance(x.position, vertex.position)};
    Vector3 dir_light = normalize(x.position - vertex.position);
    Real G = 0;
    if (!occluded(scene, shadow_ray)){
        G = max(-dot(dir_light, x.normal), Real(0)) / distance_squared(x.position, vertex.position);
    }

    Spectrum Le = emission(scene.lights[light_id], -dir_light, Real(0), x, scene);
    Spectrum rho = eval(scene.materials[vertex.material_id], -in_ray.dir, dir_light, vertex, scene.texture_pool);
    Real p_hat = luminance(rho * Le) * G;
    return p_hat;
}

// Algorithm 3: Streaming RIS using weighted reservoir sampling
void resample_importance_sampling(int M, const Scene &scene, Reservoir &r, PathVertex q, Ray ray, Vector2 screen_pos, pcg32_state &rng) {
    // Sample M light paths candidates
    for (int i=1; i<M+1; i++){
        int light_id = sample_light(scene, next_pcg32_real<Real>(rng));

        // Invalid light_id
        if (light_id == -1){
            continue;
        }

        Light light = scene.lights[light_id];
        Vector2 rng_uv_params(next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng));
        Real rng_w_param = next_pcg32_real<Real>(rng);
        PointAndNormal x = sample_point_on_light(light, q.position, rng_uv_params, rng_w_param, scene);
        
        // Calculate p
        Real p = light_pmf(scene, light_id) * pdf_point_on_light(light, x, q.position, scene);
        
        // Numerical error
        if ( p <= 0){
            continue;
        }

        // Calculate p_hat
        Real p_hat = compute_p_hat(scene, r, light_id, x, rng);

        // Update reservoir
        Real w_x = p_hat / p;
        r.update(ReservoirSample(light_id, x, p_hat, w_x), rng);
    }
};

// Init algorithm to init the reservoir buffer
void init_reservoir(const Scene &scene, ReservoirBuffer &G_buffer, int x, int y, pcg32_state &rng){
    Reservoir& r = G_buffer(x, y);
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    r.screen_pos = screen_pos;
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    
    // Check whether the ray hits the any objects
    if (!vertex_){
        // Do nothing here, deal with environment map in compute radiance function
        return;
    }
    
    PathVertex vertex = *vertex_;
    r.org_vertex = vertex;
    resample_importance_sampling(scene.options.reservoir_size, scene, r, vertex, ray, screen_pos, rng);
    
    // check visibility for the light source of reservoir sample
    Ray shadow_ray{vertex.position, normalize(r.y.x.position - vertex.position), get_shadow_epsilon(scene), 
                    (1-get_shadow_epsilon(scene)) * distance(r.y.x.position, vertex.position)};
    if (occluded(scene, shadow_ray)){
        r.W = 0;
    }
};

// Merge two reservoirs
void merge_reservoirs(const Scene& scene, Reservoir& source_reservoir, Reservoir merged_reservoir, pcg32_state &rng){
    if (!source_reservoir.org_vertex){
        return;
    }

    ReservoirSample merged_sample = merged_reservoir.y;
    if (merged_sample.light_id == -1){
        return;
    }

    Real p_hat = compute_p_hat(scene, source_reservoir, merged_sample.light_id, merged_sample.x, rng);
    Real w = p_hat * merged_reservoir.W * merged_reservoir.M;
    merged_sample.p_hat = p_hat;
    merged_sample.w = w;

    source_reservoir.update(merged_sample, rng, merged_reservoir.M);

    // correct bias
    int Z = 0;
    if (compute_p_hat(scene, source_reservoir, source_reservoir.y.light_id, source_reservoir.y.x, rng) > 0){
        Z += source_reservoir.M - merged_reservoir.M;
    }
    if (compute_p_hat(scene, merged_reservoir, source_reservoir.y.light_id, source_reservoir.y.x, rng) > 0){
        Z += merged_reservoir.M;
    }
    source_reservoir.W = source_reservoir.W * source_reservoir.M / Z;
}

// Randomly select neighbor pixel to merge their reservoirs with current pixel's reservoir
void spatial_reuse(const Scene &scene, ReservoirBuffer& G_buffer, ReservoirBuffer& target_buffer, int x, int y, pcg32_state &rng){
    target_buffer(x, y) = G_buffer(x, y);

    Reservoir& current_reservoir = target_buffer(x, y);
    if (!current_reservoir.org_vertex){
        return;
    }

    int max_radius = scene.options.max_radius;
    Vector3 cam_org = xform_point(scene.camera.cam_to_world, Vector3{0, 0, 0});

    int patience = 3;
    bool is_valid = false;
    while (!is_valid && patience > 0){
        // Sample a neighbor
        Real theta = next_pcg32_real<Real>(rng) * 2 * M_PI;
        Real radius = next_pcg32_real<Real>(rng) * Real(max_radius);
        int x_ = x + std::round(radius * cos(theta));
        int y_ = y + std::round(radius * sin(theta));
        x_ = max(0, min(x_, scene.camera.width - 1));
        y_ = max(0, min(y_, scene.camera.height - 1));
        Reservoir& neighbor_reservoir = G_buffer(x_, y_);
        if (!neighbor_reservoir.org_vertex){
            patience -= 1;
            continue;
        }
        
        // Heuristic rejection
        PathVertex current_pv = *current_reservoir.org_vertex, neighbor_pv = *neighbor_reservoir.org_vertex;
        Real cam_q_dis = distance(cam_org, current_pv.position);
        Real cam_q_prime_dis = distance(cam_org, neighbor_pv.position);
        Real depth_diff = fabs(cam_q_dis - cam_q_prime_dis);
        Real angle_q_q_prime = std::acos(dot(neighbor_pv.geometric_normal, current_pv.geometric_normal));
        if (depth_diff >= 0.1 * cam_q_dis || angle_q_q_prime >= Real(10) / Real(180) * c_PI){
            patience -= 1;
            continue;
        }
        is_valid = true;

        // Merge two reservoirs
        merge_reservoirs(scene, current_reservoir, neighbor_reservoir, rng);
    }
};

// Compute radiance for each pixel;
Spectrum compute_radiance(const Scene &scene, ReservoirBuffer& G_buffer, int x, int y, pcg32_state &rng){
    Spectrum radiance = make_zero_spectrum();

    Reservoir& r = G_buffer(x, y);

     // If there is no hit point, account for environment map
     if (!r.org_vertex){
        int w = scene.camera.width, h = scene.camera.height;
        Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                        (y + next_pcg32_real<Real>(rng)) / h);
        Ray ray = sample_primary(scene.camera, screen_pos);
        RayDifferential ray_diff = init_ray_differential(w, h);
        if (has_envmap(scene)) {
            const Light &envmap = get_envmap(scene);
            return emission(envmap,
                            -ray.dir, // pointing outwards from light
                            ray_diff.spread,
                            PointAndNormal{}, // dummy parameter for envmap
                            scene);
        }
        return radiance;
    }

    Vector3 cam_org = xform_point(scene.camera.cam_to_world, Vector3{0, 0, 0});
    PathVertex org_vertex = *r.org_vertex;
    Vector3 in_dir = normalize(cam_org - org_vertex.position);

    // If point is light, account for the emission
    if (is_light(scene.shapes[org_vertex.shape_id])){
        radiance += emission(org_vertex, in_dir, scene);
    }
    // If invalid light sampled from RIS, return radiance
    if (r.y.light_id == -1){
        return radiance;
    }

    Light light = scene.lights[r.y.light_id];
    PointAndNormal point_on_light = r.y.x;
    Vector3 dir_light = normalize(point_on_light.position - org_vertex.position);
    Spectrum Le = emission(light, -dir_light, Real(0), point_on_light, scene);
    Spectrum rho = eval(scene.materials[org_vertex.material_id], in_dir, dir_light, org_vertex, scene.texture_pool);
    Real G = max(-dot(dir_light, point_on_light.normal), Real(0)) / distance_squared(point_on_light.position, org_vertex.position);
    Real W = r.W;
    if (W == 0 || isnan(W)){
        return radiance;
    }
    radiance += rho * Le * G * W;
    return radiance;
}

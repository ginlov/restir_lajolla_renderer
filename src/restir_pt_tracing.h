#pragma once

#include "reservoir.h"
#include "shift_mapping.h"

// Calculate light of from current_vertex reflecting to the ray origin
std::tuple<Spectrum, Vector3, Spectrum> visit_vertex(const Scene &scene, Ray ray, RayDifferential ray_diff, PathVertex current_vertex, int depth, pcg32_state &rng){
    if (depth == 0){
        return {make_zero_spectrum(), Vector3{Real(0), Real(0), Real(0)}, make_zero_spectrum()};
    }
    depth -= 1;
    Spectrum curr_ver_radiance = make_zero_spectrum();
    if (is_light(scene.shapes[current_vertex.shape_id])){
        curr_ver_radiance += emission(current_vertex, -ray.dir, scene);
    }

    const Material &mat = scene.materials[current_vertex.material_id];

    // NEE sampling
    Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);

    int light_id = sample_light(scene, light_w);
    const Light &light = scene.lights[light_id];
    PointAndNormal point_on_light = sample_point_on_light(light, current_vertex.position, light_uv, shape_w, scene);
    Spectrum C1 = make_zero_spectrum();
    Real w1 = 0;
    {
        Real G = 0;
        Vector3 dir_light;

        if(!is_envmap(light)){
            dir_light = normalize(point_on_light.position - current_vertex.position);

            // Check visibility
            Ray shadow_ray{current_vertex.position, dir_light, get_shadow_epsilon(scene),
                        (1-get_shadow_epsilon(scene)) * distance(point_on_light.position, current_vertex.position)};

            if (!occluded(scene, shadow_ray)) {
                G = max(-dot(dir_light, point_on_light.normal), Real(0)) / 
                distance_squared(point_on_light.position, current_vertex.position);
            } 
        } else {
            dir_light = -point_on_light.normal;

            Ray shadow_ray{current_vertex.position, dir_light,
                            get_shadow_epsilon(scene),
                        infinity<Real>()};
            if (!occluded(scene, shadow_ray)) {
                G = 1;
            }
        }

        Real p1 = light_pmf(scene, light_id) * pdf_point_on_light(light, point_on_light, current_vertex.position, scene);

        if (G > 0 && p1 > 0){
            Vector3 dir_view = -ray.dir;
            assert(current_vertex.material_id >= 0);
            Spectrum f = eval(mat, dir_view, dir_light, current_vertex, scene.texture_pool);

            Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);

            C1 = G * f * L;

            Real p2 = pdf_sample_bsdf(mat, dir_view, dir_light, current_vertex, scene.texture_pool);

            p2 *= G;
            w1 = (p1 * p1) / (p1 * p1 + p2 * p2);
            C1 /= p1;
        }
    }

    curr_ver_radiance += C1 * w1;

    // Hemispherical sampling
    // Consider any vertex which is hitted by the bsdf ray as a light source
    // Call the visit function to compute it's radiance
    // In case the hit point is on a light source, account for the Multiple Importance Sampling weight=(p2*p2)/(p1*p1 + p2*p2) where p1 is the NEE PDF.
    Vector3 dir_view = -ray.dir;
    Vector2 bsdf_rnd_param_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
    Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
    std::optional<BSDFSampleRecord> bsdf_sample_ = sample_bsdf(
        mat, dir_view, current_vertex, scene.texture_pool, bsdf_rnd_param_uv, bsdf_rnd_param_w
    );

    if (!bsdf_sample_){
        // If cannot sample a bsdf ray, stop tracing and return current radiance
        return {curr_ver_radiance, Vector3{Real(0), Real(0), Real(0)}, make_zero_spectrum()};
    }

    const BSDFSampleRecord &bsdf_sample = *bsdf_sample_;
    Vector3 dir_bsdf = bsdf_sample.dir_out;

    if (bsdf_sample.eta == 0) {
        ray_diff.spread = reflect(ray_diff, current_vertex.mean_curvature, bsdf_sample.roughness);
    } else {
        ray_diff.spread = refract(ray_diff, current_vertex.mean_curvature, bsdf_sample.eta, bsdf_sample.roughness);
    }

    Ray bsdf_ray{current_vertex.position, dir_bsdf, get_intersection_epsilon(scene), infinity<Real>()};
    std::optional<PathVertex> bsdf_vertex = intersect(scene, bsdf_ray);

    Real G;
    if (bsdf_vertex){
        G = fabs(dot(dir_bsdf, bsdf_vertex->geometric_normal)) /
        distance_squared(bsdf_vertex->position, current_vertex.position);
    } else {
        G = 1;
    }

    Spectrum f = eval(mat, dir_view, dir_bsdf, current_vertex, scene.texture_pool);
    Real p2 = pdf_sample_bsdf(mat, dir_view, dir_bsdf, current_vertex, scene.texture_pool);
    if (p2 <= 0){
        // If there is numerical error with the bsdf ray, stop tracing and return current radiance
        return {curr_ver_radiance, Vector3{Real(0), Real(0), Real(0)}, make_zero_spectrum()}; 
    }

    // Convert p2 to area measure
    p2 *= G;

    if (!bsdf_vertex){
        if (has_envmap(scene)){
            // If there is no bsdf hit point but the scene has environment map, consider the environment map lighting as the next lighting
            // and compute the current radiance
            const Light &light = get_envmap(scene);
            Spectrum L = emission(light, -dir_bsdf, ray_diff.spread, PointAndNormal{}, scene);
            Spectrum C2 = G * f * L;
            PointAndNormal light_point{Vector3{0, 0, 0}, -dir_bsdf};
            Real p1 = light_pmf(scene, scene.envmap_light_id) * pdf_point_on_light(light, light_point, current_vertex.position, scene);
            Real w2 = (p2*p2) / (p1*p1 + p2*p2);

            C2 /= p2;
            curr_ver_radiance += C2 * w2;
            return {curr_ver_radiance, dir_bsdf, L};
        } else {
            return {curr_ver_radiance, Vector3{Real(0), Real(0), Real(0)}, make_zero_spectrum()};
        }
    }

    // If there is valid hit point, compute the incoming radiance of that vertex
    auto [next_vertex_radiance, _, __] = visit_vertex(scene, bsdf_ray, ray_diff, *bsdf_vertex, depth, rng);

    Real w2 = 1;
    if (is_light(scene.shapes[bsdf_vertex->shape_id])){
        // If the hit point is on a light source, account for the MIS weight
        int light_id = get_area_light_id(scene.shapes[bsdf_vertex->shape_id]);
        Real p1 = light_pmf(scene, light_id) * pdf_point_on_light(scene.lights[light_id], 
            PointAndNormal{bsdf_vertex->position, bsdf_vertex->geometric_normal}, current_vertex.position, scene);
        w2 = (p2 * p2) / (p1 * p1 + p2 * p2);
    }
    curr_ver_radiance += next_vertex_radiance * G * f * w2 / p2;
    return {curr_ver_radiance, dir_bsdf, next_vertex_radiance};
}

// Merge one candidate to the reservoir
void merge_candidate_to_reservoirs(const Scene &scene, ReservoirPT& target_reservoir, ReservoirPTSample a, Real unnormalized_w, pcg32_state &rng){
    Real M = target_reservoir.M + 1;
    Real w = unnormalized_w/M;

    Real reservoir_w = target_reservoir.y.p_hat * target_reservoir.W;

    Real random_pdf = next_pcg32_real<Real>(rng);
    if (random_pdf < w / (reservoir_w + w) || M == 1){
        target_reservoir.y = a;
    }
    target_reservoir.M = min(target_reservoir.Mc, M);
    if (M == 1){
        target_reservoir.W = 1 / target_reservoir.y.p_hat * w;
    } else {
        target_reservoir.W = 1 / target_reservoir.y.p_hat * (reservoir_w + w);
    }
}

// Merge two reservoirs together
void merge_reservoirs_pt(const Scene& scene, ReservoirPT& target_reservoir, ReservoirPT& merged_reservoir, pcg32_state &rng){
    if (!merged_reservoir.y.reconnection_vertex || !target_reservoir.org_vertex || !merged_reservoir.org_vertex){
        return;
    }

    // shift the candidate from merged_reservoir to target_reservoir domain
    // and vice versa, shift the candidate from target_reservoir to merged_reservoir domain to compute MIS weight
    ReservoirPTSample shifted_sample = reconnection_shift_mapping(scene, merged_reservoir, target_reservoir);
    ReservoirPTSample reverse_shifted_sample = reconnection_shift_mapping(scene, target_reservoir, merged_reservoir);

    // If shifted sample having no chance to be selected, return
    if (shifted_sample.p_hat<=0){
        return;
    }

    // Compute MIS weight base on equation 38 in the paper
    auto [merged_mis, target_mis] = MIS(scene, merged_reservoir, target_reservoir, shifted_sample, reverse_shifted_sample);

    if (merged_mis == 0){
        return;
    }

    // Compute weight, candidate from target reservoir is from the original domain so it does not need to be multiply with jacobian determinant
    PathVertex y = *target_reservoir.org_vertex, x = *merged_reservoir.org_vertex, x1 = *merged_reservoir.y.reconnection_vertex;
    Real jacobian_det = fabs(dot(normalize(x1.position - y.position), x1.geometric_normal) / (dot(normalize(x1.position - x.position), x1.geometric_normal)))*
        distance_squared(x1.position, x.position) / distance_squared(x1.position, y.position);
    Real w_target = target_mis * target_reservoir.y.p_hat * target_reservoir.W;
    Real w_merged = merged_mis * shifted_sample.p_hat * merged_reservoir.W * jacobian_det;

    // In case the target reservoir is empty, just copy the merged reservoir to the target reservoir
    if (target_reservoir.M == 0){
        target_reservoir.y = shifted_sample;
        // M capping
        target_reservoir.M = min(target_reservoir.Mc, target_reservoir.M + merged_reservoir.M);
        if (target_reservoir.y.p_hat == 0){
            target_reservoir.W = 0;
        } else {
            target_reservoir.W = 1 / target_reservoir.y.p_hat * w_merged;
        }
        return;
    }

    // Merge the two reservoirs
    // M capping
    target_reservoir.M = min(target_reservoir.Mc, target_reservoir.M + merged_reservoir.M);
    if (w_target + w_merged == 0){
        return;
    }

    Real random_pdf = next_pcg32_real<Real>(rng);
    if (random_pdf < w_merged / (w_target + w_merged)){
        target_reservoir.y = shifted_sample;
    }
   
    if (target_reservoir.y.p_hat == 0){
        target_reservoir.W = 0;
    } else {
        target_reservoir.W = 1 / target_reservoir.y.p_hat * (w_target + w_merged);
    }
}

// GRIS
void resample_importance_sampling_pt(int M, const Scene &scene, ReservoirPT &r, PathVertex q, Ray ray, RayDifferential ray_diff, int max_depth, pcg32_state &rng){
    Material mat = scene.materials[q.material_id];

    for (int i = 0; i<M; i++){
        ReservoirPTSample a;
        Real w;
        
        // Do bsdf sampling
        Vector2 rnd_param_uv = Vector2{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
        Real rnd_param_w = next_pcg32_real<Real>(rng);
        std::optional<BSDFSampleRecord> bsdf_record_ = sample_bsdf(mat, -ray.dir, q, scene.texture_pool, rnd_param_uv, rnd_param_w);
        
        if (!bsdf_record_){
            // If cannot sample an out_dir, continue the loop sampling with another random parameters
            continue;
        }

        BSDFSampleRecord &bsdf_sample = *bsdf_record_;
        Vector3 dir_out = bsdf_sample.dir_out;
        // Update ray differentials & eta_scale
        if (bsdf_sample.eta == 0) {
            ray_diff.spread = reflect(ray_diff, q.mean_curvature, bsdf_sample.roughness);
        } else {
            ray_diff.spread = refract(ray_diff, q.mean_curvature, bsdf_sample.eta, bsdf_sample.roughness);
        }

        // Trace a ray towards dir_out.
        Ray bsdf_ray{q.position, dir_out, get_intersection_epsilon(scene), infinity<Real>()};
        std::optional<PathVertex> bsdf_vertex = intersect(scene, bsdf_ray);
        
        if (!bsdf_vertex){
            // If do not hit any object, continue the loop sampling with another random parameters
            continue;
        }

        Real G = fabs(dot(dir_out, bsdf_vertex->geometric_normal)) /
                distance_squared(bsdf_vertex->position, q.position);

        // Contribution weight for this candidate
        Real p = pdf_sample_bsdf(mat, -ray.dir, dir_out, q, scene.texture_pool) * G;

        // Check for numerical error
        if (p<=0){
            continue;
        }

        Real W = 1/p;

        Spectrum f = eval(mat, -ray.dir, dir_out, q, scene.texture_pool);
        auto [next_vertex_spectrum, next_out_dir, next_next_spectrum] = visit_vertex(scene, bsdf_ray, ray_diff, *bsdf_vertex, max_depth, rng);

        Real p_hat = luminance(f * next_vertex_spectrum) * G;
        if (p_hat <=0){
            continue;
        }

        w = p_hat * W;
        
        // Modify this sample
        a.F = f * next_vertex_spectrum * G;
        a.rnd_param_uv = rnd_param_uv;
        a.rnd_param_w = rnd_param_w;
        a.p_hat = p_hat;
        
        a.reconnection_vertex = bsdf_vertex;
        a.recon_out_dir = next_out_dir;
        a.recon_next_ver_radiance = next_next_spectrum;
        
        // Check visibility
        Ray shadow_ray{q.position, normalize(a.reconnection_vertex->position - q.position), get_shadow_epsilon(scene), 
            (1-get_shadow_epsilon(scene)) * distance(a.reconnection_vertex->position, q.position)};

        if (occluded(scene, shadow_ray)){
            continue;
        }
        merge_candidate_to_reservoirs(scene, r, a, w, rng);
    }
}

// Init the reservoir buffer
void init_reservoir_pt(const Scene &scene, ReservoirPTBuffer &G_buffer, int x, int y, int max_depth, pcg32_state &rng){
    ReservoirPT& r = G_buffer(x, y);
    r.Mc = Real(scene.options.m_capping);

    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                        (y + next_pcg32_real<Real>(rng)) / h);
    r.screen_pos = screen_pos;
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    
    // Check whether the ray hits any objects
    if (!vertex_){
        // TODO: Do something here to complete the logics
        return;
    }

    PathVertex vertex = *vertex_;
    r.org_vertex = vertex;
    resample_importance_sampling_pt(scene.options.reservoir_size, scene, r, vertex, ray, ray_diff, max_depth, rng);
}

// Select neighbor and reuse it's light path
void spatial_reuse_pt(const Scene &scene, ReservoirPTBuffer &G_buffer, ReservoirPTBuffer& target_buffer, int x, int y, pcg32_state& rng){
    // if (G_buffer(x, y).org_vertex && G_buffer(x, y).M == 0){
    //     std::cout<<"Empty reservoir"<<std::endl;
    // }
    target_buffer(x, y) = ReservoirPT(G_buffer(x, y));

    ReservoirPT& current_reservoir = target_buffer(x, y);
    if (!current_reservoir.org_vertex){
        return;
    }
    
    int max_radius = scene.options.max_radius;

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
        ReservoirPT& neighbor_reservoir = G_buffer(x_, y_);
        if (!neighbor_reservoir.org_vertex || neighbor_reservoir.M == 0){
            patience -= 1;
            continue;
        }

        is_valid = true;
        merge_reservoirs_pt(scene, current_reservoir, neighbor_reservoir, rng);
    }
}

// Compute radiance for each pixel
Spectrum compute_radiance(const Scene &scene, ReservoirPTBuffer& G_buffer, int x, int y, pcg32_state &rng){
    Spectrum radiance = make_zero_spectrum();

    ReservoirPT& r = G_buffer(x, y);

    // If there is no hit point, account for the environment map
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

    if (isnan(r.W)){
        std::cout<<"W is nan"<<std::endl;
    }
    radiance += r.y.F*r.W;
    return radiance;
}
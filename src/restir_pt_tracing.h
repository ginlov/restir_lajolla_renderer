#pragma once

#include "reservoir.h"
#include "shift_mapping.h"

// Calculate light of from current_vertex reflecting to the ray origin
std::tuple<Spectrum, Vector3, Spectrum> visit_vertex(const Scene &scene, Ray ray, RayDifferential ray_diff, PathVertex current_vertex, PathVertex prev_vertex, Real bsdf_pdf, int depth, pcg32_state &rng){
    // If exceed the maximum depth, stop tracing and return current radiance
    if (depth == 0){
        return {make_zero_spectrum(), Vector3{Real(0), Real(0), Real(0)}, make_zero_spectrum()};
    }
    
    // Update current available depth
    depth -= 1;

    // Start computing the radiance
    Spectrum curr_ver_radiance = make_zero_spectrum();

    // If hit point on light, account for the emission
    if (is_light(scene.shapes[current_vertex.shape_id])){
        int light_id = get_area_light_id(scene.shapes[current_vertex.shape_id]);
        const Light& light = scene.lights[light_id];
        Real p2 = light_pmf(scene, light_id) * pdf_point_on_light(light, PointAndNormal{current_vertex.position, current_vertex.geometric_normal}, prev_vertex.position, scene);
        Real w = (p2 * p2) / (p2 * p2 + bsdf_pdf * bsdf_pdf);
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

    // If cannot sample a bsdf ray, stop tracing and return current radiance
    if (!bsdf_sample_){
        return {curr_ver_radiance, Vector3{Real(0), Real(0), Real(0)}, make_zero_spectrum()};
    }

    // Init bsdf ray and trace it
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

    // If there is numerical error with the bsdf ray, stop tracing and return current radiance
    if (p2 <= 0){
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
    auto [next_vertex_radiance, _, __] = visit_vertex(scene, bsdf_ray, ray_diff, *bsdf_vertex, current_vertex, p2, depth, rng);

    curr_ver_radiance += next_vertex_radiance * G * f / p2;
    return {curr_ver_radiance, dir_bsdf, next_vertex_radiance};
}

// Merge one candidate to the reservoir
void merge_candidate_to_reservoirs(const Scene &scene, ReservoirPT& target_reservoir, ReservoirPTSample a, Real unnormalized_w, pcg32_state &rng){
    Real M = target_reservoir.M + 1;
    Real w = unnormalized_w/M;
    Real reservoir_w = target_reservoir.y.p_hat * target_reservoir.W * (M-1)/M;

    // Add the candidate to the reservoir with probability w / (reservoir_w + w)
    Real random_pdf = next_pcg32_real<Real>(rng);
    if (random_pdf < w / (reservoir_w + w)){
        target_reservoir.y = a;
    }

    // M capping and update confidence contribution
    target_reservoir.M = min(target_reservoir.Mc, M);
    target_reservoir.W = Real(1) / target_reservoir.y.p_hat * (w + reservoir_w);
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
    // shifted_sample is now on the target domain Omega
    if (shifted_sample.p_hat<=0){
        return;
    }

    // Compute MIS weight base on equation 38 in the paper
    auto [merged_mis, target_mis] = MIS(scene, merged_reservoir, target_reservoir, shifted_sample, reverse_shifted_sample);

    // If merged_mis == 0, the candidate from merged reservoir has no chance to be selected
    if (merged_mis == 0){
        return;
    }

    // Compute weight, candidate from target reservoir is from the original domain so it does not need to be multiply with jacobian determinant
    PathVertex y = *target_reservoir.org_vertex, x = *merged_reservoir.org_vertex, x1 = *merged_reservoir.y.reconnection_vertex;
    Real jacobian_det = reconnection_jacobian_det(x, x1, y);
    Real w_target = target_mis * target_reservoir.y.p_hat * target_reservoir.W;
    Real w_merged = merged_mis * shifted_sample.p_hat * merged_reservoir.W * jacobian_det;

    // In case the target reservoir is empty, just copy the merged reservoir to the target reservoir
    if (target_reservoir.M == 0){
        target_reservoir.y = shifted_sample;

        // M capping
        target_reservoir.M = min(target_reservoir.Mc, target_reservoir.M + merged_reservoir.M);
        target_reservoir.W = 1 / target_reservoir.y.p_hat * w_merged;
        return;
    }

    // Merge the two reservoirs
    // M capping
    if (w_target + w_merged == 0){
        return;
    }
    target_reservoir.M = min(target_reservoir.Mc, target_reservoir.M + merged_reservoir.M);

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
        
        // Sample a ray to find the next vertex in light path
        Vector2 rnd_param_uv = Vector2{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
        Real rnd_param_w = next_pcg32_real<Real>(rng);
        std::optional<BSDFSampleRecord> bsdf_record_ = sample_bsdf(mat, -ray.dir, q, scene.texture_pool, rnd_param_uv, rnd_param_w);
        
        // If cannot sample a bsdf ray, continue the loop sampling with another random parameters
        if (!bsdf_record_){
            continue;
        }

        // Init bsdf ray
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
        
        // If do not hit any object, continue the loop sampling with another random parameters
        if (!bsdf_vertex){
            continue;
        }

        // Geometry term
        Real G = fabs(dot(dir_out, bsdf_vertex->geometric_normal)) /
                distance_squared(bsdf_vertex->position, q.position);

        // Contribution weight for this candidate SHOULD WE MULTIPLE P BY G HERE?
        Real p = pdf_sample_bsdf(mat, -ray.dir, dir_out, q, scene.texture_pool) * G;
        // Check for numerical error
        if (p<=0){
            continue;
        }
        Real W = Real(1) / p;
        Spectrum f = eval(mat, -ray.dir, dir_out, q, scene.texture_pool);

        // Compute the next vertex radiance
        auto [next_vertex_spectrum, next_out_dir, next_next_spectrum] = visit_vertex(scene, bsdf_ray, ray_diff, *bsdf_vertex, q, p, max_depth, rng);

        // Compute unnomarlized pdf p_hat
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

    // Sample a ray and find the hit point on the scene.
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                        (y + next_pcg32_real<Real>(rng)) / h);
    r.screen_pos = screen_pos;
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);
    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    
    // If no hit point, return
    if (!vertex_){
        // TODO: Do something here to complete the logics
        return;
    }

    // If ray hits scene, update org_vertex and start GRIS
    PathVertex vertex = *vertex_;
    r.org_vertex = vertex;
    resample_importance_sampling_pt(scene.options.reservoir_size, scene, r, vertex, ray, ray_diff, max_depth, rng);
}

// Select neighbor and reuse it's light path
void spatial_reuse_pt(const Scene &scene, ReservoirPTBuffer &G_buffer, ReservoirPTBuffer& target_buffer, int x, int y, pcg32_state& rng){
    target_buffer(x, y) = ReservoirPT(G_buffer(x, y));
    ReservoirPT& current_reservoir = target_buffer(x, y);

    // If current reservoir is not tagged with any vertex, return
    if (!current_reservoir.org_vertex){
        return;
    }
    
    // int max_radius = scene.options.max_radius;
    // int patience = 3;
    // bool is_valid = false;
    // while (!is_valid && patience > 0){
    //     // Sample a neighbor
    //     Real theta = next_pcg32_real<Real>(rng) * 2 * M_PI;
    //     Real radius = next_pcg32_real<Real>(rng) * Real(max_radius);
    //     int x_ = x + std::round(radius * cos(theta));
    //     int y_ = y + std::round(radius * sin(theta));
    //     x_ = max(0, min(x_, scene.camera.width - 1));
    //     y_ = max(0, min(y_, scene.camera.height - 1));
    //     ReservoirPT& neighbor_reservoir = G_buffer(x_, y_);
        
    //     // If the neighbor has no vertex tagged with or is empty, stop merging
    //     if (!neighbor_reservoir.org_vertex || neighbor_reservoir.M == 0){
    //         patience -= 1;
    //         continue;
    //     }

    //     is_valid = true;
    //     merge_reservoirs_pt(scene, current_reservoir, neighbor_reservoir, rng);
    // }

    std::vector<std::pair<int, int>> neighbors = sample_low_discrepancy_neighbors(scene.options.neighbors_per_pixel, x, y, scene.options.max_radius, scene.camera.width, scene.camera.height);
    if (neighbors.size() == 0){
        // If there are no neighbors
        return;
    }

    for (int i = 0; i<neighbors.size(); i++){
        int x_ = neighbors[i].first;
        int y_ = neighbors[i].second;
        ReservoirPT& neighbor_reservoir = G_buffer(x_, y_);
        if (!neighbor_reservoir.org_vertex || neighbor_reservoir.M == 0){
            continue;
        }

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

    // NEE
    Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);
    int light_id = sample_light(scene, light_w);
    const Light &light = scene.lights[light_id];
    PointAndNormal point_on_light =
        sample_point_on_light(light, org_vertex.position, light_uv, shape_w, scene);

    // Next, we compute w1*C1/p1. We store C1/p1 in C1.
    Spectrum C1 = make_zero_spectrum();
    Real w1 = 0;
    // Remember "current_path_throughput" already stores all the path contribution on and before v_i.
    // So we only need to compute G(v_{i}, v_{i+1}) * f(v_{i-1}, v_{i}, v_{i+1}) * L(v_{i}, v_{i+1})
    {
        // Let's first deal with C1 = G * f * L.
        // Let's first compute G.
        Real G = 0;
        Vector3 dir_light;
        // The geometry term is different between directional light sources and
        // others. Currently we only have environment maps as directional light sources.
        if (!is_envmap(light)) {
            dir_light = normalize(point_on_light.position - org_vertex.position);
            // If the point on light is occluded, G is 0. So we need to test for occlusion.
            // To avoid self intersection, we need to set the tnear of the ray
            // to a small "epsilon". We set the epsilon to be a small constant times the
            // scale of the scene, which we can obtain through the get_shadow_epsilon() function.
            Ray shadow_ray{org_vertex.position, dir_light, 
                           get_shadow_epsilon(scene),
                           (1 - get_shadow_epsilon(scene)) *
                               distance(point_on_light.position, org_vertex.position)};
            if (!occluded(scene, shadow_ray)) {
                // geometry term is cosine at v_{i+1} divided by distance squared
                // this can be derived by the infinitesimal area of a surface projected on
                // a unit sphere -- it's the Jacobian between the area measure and the solid angle
                // measure.
                G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                    distance_squared(point_on_light.position, org_vertex.position);
            }
        } else {
            // The direction from envmap towards the point is stored in
            // point_on_light.normal.
            dir_light = -point_on_light.normal;
            // If the point on light is occluded, G is 0. So we need to test for occlusion.
            // To avoid self intersection, we need to set the tnear of the ray
            // to a small "epsilon" which we define as c_shadow_epsilon as a global constant.
            Ray shadow_ray{org_vertex.position, dir_light, 
                           get_shadow_epsilon(scene),
                           infinity<Real>() /* envmaps are infinitely far away */};
            if (!occluded(scene, shadow_ray)) {
                // We integrate envmaps using the solid angle measure,
                // so the geometry term is 1.
                G = 1;
            }
        }

        // Before we proceed, we first compute the probability density p1(v1)
        // The probability density for light sampling to sample our point is
        // just the probability of sampling a light times the probability of sampling a point
        Real p1 = light_pmf(scene, light_id) *
            pdf_point_on_light(light, point_on_light, org_vertex.position, scene);

        // We don't need to continue the computation if G is 0.
        // Also sometimes there can be some numerical issue such that we generate
        // a light path with probability zero
        if (G > 0 && p1 > 0) {
            // Let's compute f (BSDF) next.
            Vector3 dir_view = in_dir;
            assert(org_vertex.material_id >= 0);
            Spectrum f = eval(scene.materials[org_vertex.material_id], dir_view, dir_light, org_vertex, scene.texture_pool);

            // Evaluate the emission
            // We set the footprint to zero since it is not fully clear how
            // to set it in this case.
            // One way is to use a roughness based heuristics, but we have multi-layered BRDFs.
            // See "Real-time Shading with Filtered Importance Sampling" from Colbert et al.
            // for the roughness based heuristics.
            Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);

            // C1 is just a product of all of them!
            C1 = G * f * L;
        
            // Next let's compute w1

            // Remember that we want to set
            // w1 = p_1(v^1)^2 / (p_1(v^1)^2 + p_2(v^1)^2)
            // Notice that all of the probability density share the same path prefix and those cancel out.
            // Therefore we only need to account for the generation of the vertex v_{i+1}.

            // The probability density for our hemispherical sampling to sample 
            Real p2 = pdf_sample_bsdf(
                scene.materials[org_vertex.material_id], dir_view, dir_light, org_vertex, scene.texture_pool);
            // !!!! IMPORTANT !!!!
            // In general, p1 and p2 now live in different spaces!!
            // our BSDF API outputs a probability density in the solid angle measure
            // while our light probability density is in the area measure.
            // We need to make sure that they are in the same space.
            // This can be done by accounting for the Jacobian of the transformation
            // between the two measures.
            // In general, I recommend to transform everything to area measure 
            // (except for directional lights) since it fits to the path-space math better.
            // Converting a solid angle measure to an area measure is just a
            // multiplication of the geometry term G (let solid angle be dS, area be dA,
            // we have dA/dS = G).
            p2 *= G;

            w1 = (p1*p1) / (p1*p1 + p2*p2);
            C1 /= p1;
        }
    }
    radiance += C1 * w1;

    // Account for light from reservoir rather than bsdf sampling
    radiance += r.y.F*r.W;
    return radiance;
}
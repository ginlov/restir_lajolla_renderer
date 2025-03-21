#pragma once

#include "reservoir.h"

Real reconnection_jacobian_det(const PathVertex &x, const PathVertex &x1, const PathVertex &y){
    return fabs(dot(normalize(x1.position - y.position), x1.geometric_normal) / 
                            dot(normalize(x1.position - x.position), x1.geometric_normal)) * distance_squared(x.position, x1.position) /
                            distance_squared(x1.position, y.position);
}

ReservoirPTSample reconnection_shift_mapping(const Scene &scene, ReservoirPT source_reservoir, ReservoirPT target_reservoir){
    // Shift the candidate from source_reservoir to target_reservoir domain
    // That means connect org vertex of target reservoir(y) to the reconnection vertex of source reservoir(x1)
    // The org vertex of source resesrvoir acts as (x)
    ReservoirPTSample path_to_shift = source_reservoir.y;

    // If there is no reconnection vertex, return
    if (!path_to_shift.reconnection_vertex){
        return ReservoirPTSample();
    }


    PathVertex y = *target_reservoir.org_vertex;
    PathVertex x1 = *path_to_shift.reconnection_vertex;
    // Check visibility
    Ray shadow_ray{y.position, normalize(x1.position - y.position), get_shadow_epsilon(scene), (1-get_shadow_epsilon(scene)) * distance(x1.position, y.position)};
    if (occluded(scene, shadow_ray)){
        return ReservoirPTSample();
    }

    const Material &mat = scene.materials[x1.material_id];
    Vector3 dir_view = normalize(y.position - x1.position);

    // Re-compute radiance for the reconnection vertex
    Spectrum radiance = make_zero_spectrum();
    // If is light, account for the emission
    if (is_light(scene.shapes[x1.shape_id])){
        radiance += emission(x1, dir_view, scene);
    }

    // NEE
    Vector2 light_uv = path_to_shift.recon_nee_light_uv;
    Real light_w = path_to_shift.recon_nee_light_w;
    Real shape_w = path_to_shift.recon_nee_shape_w;

    int light_id = sample_light(scene, light_w);
    const Light &light = scene.lights[light_id];
    PointAndNormal point_on_light = sample_point_on_light(light, x1.position, light_uv, shape_w, scene);
    Spectrum C1 = make_zero_spectrum();
    Real w1 = 0;
    {
        Real G = 0;
        Vector3 dir_light;

        if(!is_envmap(light)){
            dir_light = normalize(point_on_light.position - x1.position);

            // Check visibility
            Ray shadow_ray{x1.position, dir_light, get_shadow_epsilon(scene),
                        (1-get_shadow_epsilon(scene) * distance(point_on_light.position, x1.position))};

            if (!occluded(scene, shadow_ray)) {
                G = max(-dot(dir_light, point_on_light.normal), Real(0)) / 
                distance_squared(point_on_light.position, x1.position);
            }
        } else {
            dir_light = -point_on_light.normal;

            Ray shadow_ray{x1.position, dir_light,
                            get_shadow_epsilon(scene),
                        infinity<Real>()};
            if (!occluded(scene, shadow_ray)) {
                G = 1;
            }
        }

        Real p1 = light_pmf(scene, light_id) * pdf_point_on_light(light, point_on_light, x1.position, scene);

        if (G > 0 && p1 > 0){
            assert(x1.material_id >= 0);
            Spectrum f = eval(mat, dir_view, dir_light, x1, scene.texture_pool);

            Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);

            C1 = G * f * L;

            Real p2 = pdf_sample_bsdf(mat, dir_view, dir_light, x1, scene.texture_pool);

            p2 *= G;
            w1 = (p1 * p1) / (p1 * p1 + p2 * p2);
            C1 /= p1;
        }
    }

    radiance += C1 * w1;

    // hemispherical sampling
    if (path_to_shift.recon_out_dir.x != 0 || path_to_shift.recon_out_dir.y != 0 || path_to_shift.recon_out_dir.z != 0){
        Ray bsdf_ray{x1.position, path_to_shift.recon_out_dir, get_intersection_epsilon(scene), infinity<Real>()};
        std::optional<PathVertex> bsdf_vertex = intersect(scene, bsdf_ray);

        Real G;
        if (bsdf_vertex){
            G = fabs(dot(path_to_shift.recon_out_dir, bsdf_vertex->geometric_normal)) /
            distance_squared(bsdf_vertex->position, x1.position);
        } else {
            G = 1;
        }

        Spectrum f = eval(mat, dir_view, path_to_shift.recon_out_dir, x1, scene.texture_pool);
        Real p2 = pdf_sample_bsdf(mat, dir_view, path_to_shift.recon_out_dir, x1, scene.texture_pool);
        if (p2 >0) {
            // Convert p2 to area measure
            p2 *= G;

            if (!bsdf_vertex && has_envmap(scene)){
                Spectrum L = path_to_shift.recon_next_ver_radiance;
                Spectrum C2 = G * f * L;
                PointAndNormal light_point{Vector3{0, 0, 0}, -path_to_shift.recon_out_dir};
                Real p1 = light_pmf(scene, scene.envmap_light_id) * pdf_point_on_light(light, light_point, x1.position, scene);
                Real w2 = (p2*p2) / (p1*p1 + p2*p2);

                C2 /= p2;
                radiance += C2 * w2;
            } else {
                Spectrum next_vertex_radiance = path_to_shift.recon_next_ver_radiance;
                Real w2 = 1;
                if (is_light(scene.shapes[bsdf_vertex->shape_id])){
                    int light_id = get_area_light_id(scene.shapes[bsdf_vertex->shape_id]);
                    Real p1 = light_pmf(scene, light_id) * pdf_point_on_light(scene.lights[light_id], 
                        PointAndNormal{bsdf_vertex->position, bsdf_vertex->geometric_normal}, x1.position, scene);
                    w2 = (p2 * p2) / (p1 * p1 + p2 * p2);
                }
                radiance += next_vertex_radiance * G * f * w2 / p2;
            }
        }
    }

    // Calculate integrand
    Vector3 cam_org = xform_point(scene.camera.cam_to_world, Vector3{0, 0, 0});
    Spectrum f = eval(scene.materials[y.material_id], normalize(cam_org - y.position), -dir_view, y, scene.texture_pool);
    Real G = fabs(dot(dir_view, x1.geometric_normal)) /
            distance_squared(x1.position, y.position);

    Real p_hat = luminance(f*radiance) * G;
    Spectrum F = f * radiance * G;

    ReservoirPTSample a = path_to_shift;
    a.F = F;
    a.p_hat = p_hat;
    return a;
}

std::tuple<Real, Real> MIS(const Scene &scene, ReservoirPT& source_reservoir, ReservoirPT& target_reservoir, ReservoirPTSample& shifted_sample, ReservoirPTSample& reverse_shifted_sample){
    // Target reservoir is the reservoir containing the vertex which need sample a light path
    // Source reservoir is the reservoir containing the candidate light path need to be shifted
    // Shifted sample is the shifted version of source reservoir candidate
    // Reverse shifted sample is the shifted version of target reservoir to source reservoir domain

    Real M = source_reservoir.M + target_reservoir.M;
    Real R = target_reservoir.M;
    Real M_R = source_reservoir.M;

    PathVertex x = *target_reservoir.org_vertex, x1 = *target_reservoir.y.reconnection_vertex;
    // Calculate the MIS weight for the target reservoir candidate
    ReservoirPTSample y = target_reservoir.y, inverse_y = reverse_shifted_sample;
    // MIS = 1/M + 1/M \sum_{all candidates not in target domain} p_hat(y)/|R|p_hat(y) + (M - |R|)p_hat_j(y)
    Real jacobian_det = reconnection_jacobian_det(x, x1, *source_reservoir.org_vertex);
    Real p_hat = y.p_hat;
    Real p_hat_j = inverse_y.p_hat * jacobian_det;
    Real target_mis = 0;
    if (R > 0){
        target_mis = R*(1/M + 1/M * p_hat* M_R / (p_hat * R + p_hat_j * M_R));
    }

    // Calculate the MIS weight for the source reservoir candidate
    x1 = *shifted_sample.reconnection_vertex;
    y = shifted_sample, inverse_y = source_reservoir.y;
    // MIS = (M-|R|)/M * (p_hat_j(y))/ (|R|p_hat(y) + (M-|R|)p_hat_j(y))
    jacobian_det = reconnection_jacobian_det(x, x1, *source_reservoir.org_vertex);
    p_hat = y.p_hat;
    p_hat_j = inverse_y.p_hat * jacobian_det;
    Real source_mis = M_R/M * (p_hat_j * M_R) / (R * p_hat + M_R * p_hat_j);

    return {source_mis, target_mis};
}
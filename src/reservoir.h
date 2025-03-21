#pragma once

#include <vector>

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
        w_sum += candidate.w*num_candidates;
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

// For now: ReservoirPTSample only support reconnection shift
struct ReservoirPTSample {
    Spectrum F; // Cached integrand value of the sample (original vertex). F = f * L * G
    Vector2 rnd_param_uv; // random uv_param at hit x1.
    Real rnd_param_w; // random w_param at hit x1.
    Real p_hat; // p_hat of the light path on current domain (i.e, domain of original vertex).

    // Reconnection infomation
    std::optional<PathVertex> reconnection_vertex; // also x1 since we only support reconnection shift.
    Vector3 recon_out_dir; // out_dir bsdf at x1.
    Spectrum recon_next_ver_radiance; // next vertex radiance at x1.
    Vector2 recon_nee_light_uv; // nee light uv at x1 to recalculate nee lighting.
    Real recon_nee_light_w; // nee light w at x1 to recalculate nee lighting.
    Real recon_nee_shape_w; // nee shape w at x1 to recalculate nee lighting.

    ReservoirPTSample(): F(make_zero_spectrum()), rnd_param_uv(Vector2{Real(0), Real(0)}), rnd_param_w(Real(0)), recon_out_dir(Vector3{Real(0), Real(0), Real(0)}), recon_next_ver_radiance(make_zero_spectrum()) {};
    ReservoirPTSample(Spectrum F, Vector2 rnd_param_uv, Real rnd_param_w, std::optional<PathVertex> reconnection_vertex, Vector3 recon_out_dir, Spectrum recon_next_ver_radiance):
    F(F), rnd_param_uv(rnd_param_uv), rnd_param_w(rnd_param_w), reconnection_vertex(reconnection_vertex), recon_out_dir(recon_out_dir), recon_next_ver_radiance(recon_next_ver_radiance) {};
};

struct ReservoirPT {
    ReservoirPTSample y; // ReservoirPTSample containing candidate light path noted by the next vertex.
    Real Mc; // M-capping value.
    Real M;// Confidence weight (for e.g., M-capping).
    Real W;// Unbiased contribution weight. W = 1/p_hat * w_sum.
    std::optional<PathVertex> org_vertex; // hit point of camera on screen_pos
    Vector2 screen_pos; // screen position where camera hits the ray.

    ReservoirPT(): y(ReservoirPTSample()), Mc(36), M(0), W(0), org_vertex({}), screen_pos(Vector2{Real(0), Real(0)}) {};
    ReservoirPT(ReservoirPTSample y, Real Mc, Real M, Real W, std::optional<PathVertex> org_vertex, Vector2 screen_pos, int num_nee, int num_bsdf): 
                y(y), Mc(Mc), M(M), W(W), org_vertex(org_vertex), screen_pos(screen_pos) {};
};

struct ReservoirPTBuffer {
    ReservoirPTBuffer() {};
    ReservoirPTBuffer(int width, int height): width(width), height(height){
        data.resize(width * height, ReservoirPT());
    }

    ReservoirPT &operator()(int x, int y){
        return data[y*width + x];
    }

    const ReservoirPT &operator()(int x, int y) const {
        return data[y*width + x];
    }

    ReservoirPT &operator()(int x){
        return data[x];
    }

    const ReservoirPT &operator()(int x) const{
        return data[x];
    }

    int width;
    int height;
    std::vector<ReservoirPT> data;
};

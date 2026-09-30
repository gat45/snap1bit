// Sonde v3 : comparaison directe HTP0 vs CPU sur le MEME graphe (meme donnees quantisees).
// Aucune dequantisation manuelle : le backend CPU est la reference (comme test-backend-ops).
// Mode id : b = identite N=K -> out[m][n] = w_deq[m][n], carte de faute par bande k de 32.
// argv: [device] [type=q1_0|q4_0] [K] [M] [N] [mode=id|rnd]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

// rnd-officiel : donnees aleatoires uniformes [-1,1] comme test-backend-ops
static bool g_uniform = false;
static unsigned g_seed = 42;

int main(int argc, char ** argv) {
    const char * want   = argc > 1 ? argv[1] : "HTP0";
    const char * ty     = argc > 2 ? argv[2] : "q1_0";
    const int    K      = argc > 3 ? atoi(argv[3]) : 256;
    const int    M      = argc > 4 ? atoi(argv[4]) : 16;
    const int    N      = argc > 5 ? atoi(argv[5]) : 256;
    const char * mode   = argc > 6 ? argv[6] : "id";
    const bool identity  = strcmp(mode, "id") == 0;
    const enum ggml_type wtype = strcmp(ty, "q4_0") == 0 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q1_0;

    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr, dev_cpu = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); j++) {
            ggml_backend_dev_t d = ggml_backend_reg_dev_get(reg, j);
            const char * nm = ggml_backend_dev_name(d);
            if (strcmp(nm, want) == 0) dev = d;
            if (strcmp(nm, "CPU") == 0) dev_cpu = d;
        }
    }
    if (!dev || !dev_cpu) { printf("device manquant\n"); return 1; }
    ggml_backend_t be_htp = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t be_cpu = ggml_backend_dev_init(dev_cpu, nullptr);

    // poids : deterministes, ou uniformes [-1,1] type officiel si argv[7]=uni
    std::vector<float> w((size_t) K * M);
    const bool uni = argc > 7 && strcmp(argv[7], "uni") == 0;
    const bool noweights = argc > 9 && strcmp(argv[9], "nowusage") == 0; // mimique officiel : pas de flag WEIGHT
    g_uniform = uni;
    std::mt19937 rng(argc > 8 ? (unsigned) atoi(argv[8]) : g_seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int r = 0; r < M; r++)
        for (int e = 0; e < K; e++)
            w[(size_t) r * K + e] = uni ? dist(rng) : (((r * 31 + e * 7) % 17) - 8) / 8.0f;

    const size_t row_bytes = ggml_row_size(wtype, K);
    std::vector<uint8_t> wq(row_bytes * M);
    ggml_quantize_chunk(wtype, w.data(), wq.data(), 0, M, K, nullptr);

    std::vector<float> bhost((size_t) K * N);
    if (identity) {
        std::fill(bhost.begin(), bhost.end(), 0.0f);
        for (int i = 0; i < K && i < N; i++) bhost[(size_t) i * K + i] = 1.0f;
    } else if (uni) {
        for (size_t i = 0; i < bhost.size(); i++) bhost[i] = dist(rng);
    } else {
        for (size_t i = 0; i < bhost.size(); i++)
            bhost[i] = ((int) (i * 131 + 7) % 23 - 11) / 11.0f;
    }

    const size_t out_bytes = sizeof(float) * (size_t) M * N;
    std::vector<float> out_htp((size_t) M * N), out_cpu((size_t) M * N);

    // ---- HTP ----
    {
        struct ggml_init_params ip1 = { 64 * 1024 * 1024, nullptr, true };
        ggml_context * cw = ggml_init(ip1);
        ggml_tensor * a = ggml_new_tensor_2d(cw, wtype, K, M);
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, be_htp);
        if (!noweights) ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(a, wq.data(), 0, row_bytes * M);

        struct ggml_init_params ip = { 256 * 1024 * 1024, nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * b = ggml_new_tensor_2d(c, GGML_TYPE_F32, K, N);
        ggml_tensor * o = ggml_mul_mat(c, a, b);
        ggml_backend_buffer_t bb = ggml_backend_alloc_ctx_tensors(c, be_htp);
        ggml_backend_tensor_set(b, bhost.data(), 0, sizeof(float) * bhost.size());
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, o);
        ggml_backend_graph_compute(be_htp, gf);
        ggml_backend_tensor_get(o, out_htp.data(), 0, out_bytes);
        ggml_backend_buffer_free(bb);
        ggml_backend_buffer_free(bw);
        ggml_free(c);
        ggml_free(cw);
    }
    // ---- CPU ----
    {
        struct ggml_init_params ip1 = { 64 * 1024 * 1024, nullptr, true };
        ggml_context * cw = ggml_init(ip1);
        ggml_tensor * a = ggml_new_tensor_2d(cw, wtype, K, M);
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, be_cpu);
        ggml_backend_tensor_set(a, wq.data(), 0, row_bytes * M);

        struct ggml_init_params ip = { 256 * 1024 * 1024, nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * b = ggml_new_tensor_2d(c, GGML_TYPE_F32, K, N);
        ggml_tensor * o = ggml_mul_mat(c, a, b);
        ggml_backend_buffer_t bb = ggml_backend_alloc_ctx_tensors(c, be_cpu);
        ggml_backend_tensor_set(b, bhost.data(), 0, sizeof(float) * bhost.size());
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, o);
        ggml_backend_graph_compute(be_cpu, gf);
        ggml_backend_tensor_get(o, out_cpu.data(), 0, out_bytes);
        ggml_backend_buffer_free(bb);
        ggml_backend_buffer_free(bw);
        ggml_free(c);
        ggml_free(cw);
    }

    // ---- reference exacte : produit fp64 des poids dequantises x activations brutes ----
    std::vector<float> wde((size_t) K * M);
    for (int r = 0; r < M; r++) {
        const uint8_t * rp = wq.data() + (size_t) r * row_bytes;
        if (wtype == GGML_TYPE_Q1_0) {
            const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) rp);
            const uint8_t * qs = rp + 2;
            for (int e = 0; e < K; e++)
                wde[(size_t) r * K + e] = ((qs[e / 8] >> (e % 8)) & 1) ? d : -d;
        } else {
            const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) rp);
            const uint8_t * qs = rp + 2;
            for (int e = 0; e < K; e++) {
                const uint8_t nib = (e < 16) ? (qs[e] & 0xF) : (qs[e - 16] >> 4);
                wde[(size_t) r * K + e] = (nib - 8) * d;
            }
        }
    }
    double max_cpu_exact = 0.0, max_htp_exact = 0.0, max_htp_cpu = 0.0;
    int nbad = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double exact = 0.0;
            for (int k = 0; k < K; k++)
                exact += (double) wde[(size_t) m * K + k] * bhost[(size_t) n * K + k];
            const double cv = out_cpu[(size_t) n * M + m];
            const double hv = out_htp[(size_t) n * M + m];
            if (fabs(cv - exact) > max_cpu_exact) max_cpu_exact = fabs(cv - exact);
            if (fabs(hv - exact) > max_htp_exact) max_htp_exact = fabs(hv - exact);
            if (fabs(hv - cv) > max_htp_cpu) max_htp_cpu = fabs(hv - cv);
            if (fabs(hv - cv) > 1e-2) nbad++;
        }
    }
    printf("[%s K=%d M=%d N=%d %s] |htp-cpu|=%.6f |cpu-exact|=%.6f |htp-exact|=%.6f bad=%d/%d %s\n",
           ty, K, M, N, mode, max_htp_cpu, max_cpu_exact, max_htp_exact, nbad, M * N,
           nbad ? "FAIL" : "PASS");
    // detail des 5 pires mismatches
    for (int rep = 0; rep < 5; rep++) {
        int bm = -1, bn = -1; double be = 0.0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                const double e = fabs(out_htp[(size_t) n * M + m] - out_cpu[(size_t) n * M + m]);
                bool used = false;
                for (int q = 0; q < rep; q++) {} // (pas de memo des precedents : approximation)
                if (e > be) {
                    // eviter de reprendre le meme element (simple : marquer)
                    be = e; bm = m; bn = n;
                }
            }
        if (bm >= 0 && be > 1e-3) {
            double exact = 0.0;
            for (int k = 0; k < K; k++)
                exact += (double) wde[(size_t) bm * K + k] * bhost[(size_t) bn * K + k];
            printf("  pire m=%d n=%d : cpu=%+.5f htp=%+.5f exact=%+.5f\n",
                   bm, bn, out_cpu[(size_t) bn * M + bm], out_htp[(size_t) bn * M + bm], exact);
            out_cpu[(size_t) bn * M + bm] = out_htp[(size_t) bn * M + bm]; // masquer pour le prochain pire
        }
    }

    ggml_backend_free(be_htp);
    ggml_backend_free(be_cpu);
    return 0;
}

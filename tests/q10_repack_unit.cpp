// Test unitaire host (sans DSP) :
//  (1) controle : repack_q4_0_tiled (upstream) -> repack_tiled_q4_0 (upstream inverse) = identite
//  (2) Q1_0 : repack_q1_0_as_q4_0_tiled (mon patch) -> repack_tiled_q4_0 (inverse upstream,
//      valide par la communaute) -> dequant doit donner exactement +-d du bloc q1_0 couvrant.
// Si (2) echoue, le repack Q1_0 est bugue ; si (2) passe, le bug est ailleurs (DSP/params).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

typedef unsigned short ggml_fp16_t;

static float fp16_to_fp32(ggml_fp16_t h) {
    const unsigned sign = (h >> 15) & 1u;
    const unsigned exp  = (h >> 10) & 0x1fu;
    const unsigned man  = h & 0x3ffu;
    float v;
    if (exp == 0) {
        v = (man == 0) ? 0.0f : std::ldexp((float) man, -24);
    } else if (exp == 31) {
        v = INFINITY;
    } else {
        v = std::ldexp(1.0f + man / 1024.0f, (int) exp - 15);
    }
    return sign ? -v : v;
}
static ggml_fp16_t fp32_to_fp16(float f) {
    // conversion paresseuse suffisante pour le test : via float bits
    // (on ne quantifie que des valeurs simples)
    unsigned x; memcpy(&x, &f, 4);
    unsigned sign = (x >> 16) & 0x8000u;
    int e = (int) ((x >> 23) & 0xff) - 127 + 15;
    unsigned m = x & 0x7fffffu;
    if (e <= 0) return (ggml_fp16_t) sign;
    if (e >= 31) return (ggml_fp16_t) (sign | 0x7c00u);
    return (ggml_fp16_t) (sign | ((unsigned) e << 10) | (m >> 13));
}

static const int QK1_0 = 128;
static const int QK4_0 = 32;
static const size_t HTP_MM_WEIGHT_TILE_SIZE_Q4_0 = 576;

struct block_q1_0 { ggml_fp16_t d; uint8_t qs[16]; };   // 18 o, 128 elts
struct block_q4_0 { ggml_fp16_t d; uint8_t qs[16]; };   // 18 o,  32 elts

struct fake_tensor {
    int64_t ne[4];
    void * data;
};

static int64_t hex_round_up(int64_t n, int64_t m) { return ((n + m - 1) / m) * m; }
static size_t row_size_q4_0(int64_t ne0) { return (size_t) (ne0 / 32) * 18; }
static size_t row_size_q1_0(int64_t ne0) { return (size_t) (ne0 / 128) * 18; }

// ---- verbatim ggml-hexagon.cpp : unpack/pack q4_0 quants ----
static void unpack_q4_0_quants(uint8_t * qs, const block_q4_0 * x, unsigned int bi) {
    static const int qk = QK4_0;
    for (unsigned int i = 0; i < qk / 2; ++i) {
        const int x0             = (x->qs[i] & 0x0F);
        const int x1             = (x->qs[i] >> 4);
        qs[bi * qk + i + 0]      = x0;
        qs[bi * qk + i + qk / 2] = x1;
    }
}
static void pack_q4_0_quants(block_q4_0 * x, const uint8_t * qs, unsigned int bi) {
    static const int qk = QK4_0;
    for (unsigned int i = 0; i < qk / 2; ++i) {
        const uint8_t x0 = qs[bi * qk + i + 0];
        const uint8_t x1 = qs[bi * qk + i + qk / 2];
        x->qs[i]         = x0 | (x1 << 4);
    }
}

// ---- verbatim ggml-hexagon.cpp : repack_q4_0_tiled (upstream forward) ----
static void repack_q4_0_tiled(fake_tensor * t, const void * data, size_t offset, size_t size) {
    const block_q4_0 * src_matrix = (const block_q4_0 *) data;
    int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);
    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;
    size_t slice_size = ne1 * row_size_q4_0(ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) end_slice = ne2 * ne3;
    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const block_q4_0 * src_slice = src_matrix + (slice_idx - start_slice) * (ne1 * (ne0 / 32));
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;
        for (int ct = 0; ct < n_col_tiles; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;
                uint8_t tile_quants[32][32];
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r < ne1 && kt < ne0 / 32) {
                        unpack_q4_0_quants(tile_quants[row], &src_slice[r * (ne0 / 32) + kt], 0);
                    } else {
                        memset(tile_quants[row], 8, 32);
                    }
                }
                for (int cp = 0; cp < 16; cp++) {
                    for (int row = 0; row < 32; row++) {
                        tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                    }
                }
                ggml_fp16_t * scale_dst = (ggml_fp16_t *) (tile_dst + 512);
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_slice[r * (ne0 / 32) + kt].d : 0;
                }
            }
        }
    }
}

// ---- verbatim ggml-hexagon.cpp : repack_tiled_q4_0 (upstream inverse) ----
static void repack_tiled_q4_0(void * data, const fake_tensor * t, size_t offset, size_t size) {
    block_q4_0 * dst_matrix = (block_q4_0 *) data;
    int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);
    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;
    size_t slice_size = ne1 * row_size_q4_0(ne0);
    size_t row_size_bytes = row_size_q4_0(ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) end_slice = ne2 * ne3;
    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        size_t cur_start_byte = (std::max)(offset, (size_t) slice_idx * slice_size);
        size_t cur_end_byte   = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_size);
        size_t slice_offset_start = cur_start_byte - (size_t) slice_idx * slice_size;
        size_t slice_offset_end   = cur_end_byte - (size_t) slice_idx * slice_size;
        int64_t start_row = slice_offset_start / row_size_bytes;
        int64_t end_row   = (slice_offset_end + row_size_bytes - 1) / row_size_bytes;
        end_row = (std::min)(end_row, ne1);
        int start_ct = start_row / 32;
        int end_ct   = (end_row + 31) / 32;
        end_ct = (std::min)(end_ct, n_col_tiles);
        block_q4_0 * dst_slice = dst_matrix + (cur_start_byte - offset) / sizeof(block_q4_0);
        const uint8_t * matrix_src = (const uint8_t *) t->data + slice_idx * matrix_size;
        for (int ct = start_ct; ct < end_ct; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;
                uint8_t tile_quants[32][32];
                for (int cp = 0; cp < 16; cp++) {
                    for (int row = 0; row < 32; row++) {
                        uint8_t val = tile_src[cp * 32 + row];
                        tile_quants[row][2 * cp + 0] = val & 0x0F;
                        tile_quants[row][2 * cp + 1] = val >> 4;
                    }
                }
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / 32) {
                        pack_q4_0_quants(&dst_slice[(r - start_row) * (ne0 / 32) + kt], tile_quants[row], 0);
                    }
                }
                const ggml_fp16_t * scale_src = (const ggml_fp16_t *) (tile_src + 512);
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / 32) {
                        dst_slice[(r - start_row) * (ne0 / 32) + kt].d = scale_src[row];
                    }
                }
            }
        }
    }
}

// ---- verbatim MON patch : repack_q1_0_as_q4_0_tiled ----
static void repack_q1_0_as_q4_0_tiled(fake_tensor * t, const void * data, size_t offset, size_t size) {
    const block_q1_0 * src_matrix = (const block_q1_0 *) data;
    int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);
    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;
    size_t slice_size = ne1 * row_size_q1_0(ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) end_slice = ne2 * ne3;
    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const block_q1_0 * src_slice = src_matrix + (slice_idx - start_slice) * (ne1 * (ne0 / QK1_0));
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;
        for (int ct = 0; ct < n_col_tiles; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;
                uint8_t tile_quants[32][32];
                ggml_fp16_t tile_scales[32];
                const int64_t qb  = kt / 4;
                const int64_t bofs = (kt % 4) * 32;
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r < ne1 && qb < ne0 / QK1_0) {
                        const block_q1_0 * blk = &src_slice[r * (ne0 / QK1_0) + qb];
                        const uint8_t * qsb = blk->qs;
                        for (int j = 0; j < 32; j++) {
                            const int64_t e = bofs + j;
                            tile_quants[row][j] = (uint8_t)(7 + 2 * ((qsb[e / 8] >> (e % 8)) & 1));
                        }
                        tile_scales[row] = blk->d;
                    } else {
                        memset(tile_quants[row], 8, 32);
                        tile_scales[row] = 0;
                    }
                }
                for (int cp = 0; cp < 16; cp++) {
                    for (int row = 0; row < 32; row++) {
                        tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                    }
                }
                ggml_fp16_t * scale_dst = (ggml_fp16_t *) (tile_dst + 512);
                for (int row = 0; row < 32; row++) {
                    scale_dst[row] = tile_scales[row];
                }
            }
        }
    }
}

// dequant q4_0 reference ggml : byte i = elt i (low nibble) | elt i+16 (high nibble)
static void dequant_q4_0(const block_q4_0 * b, float * out) {
    const float d = fp16_to_fp32(b->d);
    for (int e = 0; e < 32; e++) {
        const uint8_t nib = (e < 16) ? (b->qs[e] & 0xF) : (b->qs[e - 16] >> 4);
        out[e] = ((int) nib - 8) * d;
    }
}

static uint32_t rng_state = 42;
static float frand() { rng_state = rng_state * 1664525u + 1013904223u; return (float) (rng_state >> 8) / (float) (1 << 24) - 0.5f; }

// dequant q1_0 : la reference CPU (vec_dot_q1_0_q8_0 + quantize_row_q1_0_ref)
static void dequant_q1_0(const block_q1_0 * b, float * out) {
    const float d = fp16_to_fp32(b->d);
    for (int e = 0; e < 128; e++) {
        const int bit = (b->qs[e / 8] >> (e % 8)) & 1;
        out[e] = bit ? d : -d;
    }
}

// inverse Q1_0 avec le FIX7 (memset seulement au premier k-tile)
static void repack_tiled_q1_0_fix7(void * data, const fake_tensor * t, size_t offset, size_t size) {
    block_q1_0 * dst_matrix = (block_q1_0 *) data;
    int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);
    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;
    size_t slice_size = ne1 * row_size_q1_0(ne0);
    size_t row_size_bytes = row_size_q1_0(ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) end_slice = ne2 * ne3;
    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        size_t cur_start_byte = (std::max)(offset, (size_t) slice_idx * slice_size);
        size_t cur_end_byte   = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_size);
        size_t slice_offset_start = cur_start_byte - (size_t) slice_idx * slice_size;
        size_t slice_offset_end   = cur_end_byte - (size_t) slice_idx * slice_size;
        int64_t start_row = slice_offset_start / row_size_bytes;
        int64_t end_row   = (slice_offset_end + row_size_bytes - 1) / row_size_bytes;
        end_row = (std::min)(end_row, ne1);
        int start_ct = start_row / 32;
        int end_ct   = (end_row + 31) / 32;
        end_ct = (std::min)(end_ct, n_col_tiles);
        block_q1_0 * dst_slice = dst_matrix + (cur_start_byte - offset) / sizeof(block_q1_0);
        const uint8_t * matrix_src = (const uint8_t *) t->data + slice_idx * matrix_size;
        for (int ct = start_ct; ct < end_ct; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;
                uint8_t tile_quants[32][32];
                for (int cp = 0; cp < 16; cp++) {
                    for (int row = 0; row < 32; row++) {
                        uint8_t val = tile_src[cp * 32 + row];
                        tile_quants[row][2 * cp + 0] = val & 0x0F;
                        tile_quants[row][2 * cp + 1] = val >> 4;
                    }
                }
                const ggml_fp16_t * scale_src = (const ggml_fp16_t *) (tile_src + 512);
                const int64_t qb = kt / 4;
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / QK1_0) {
                        block_q1_0 * blk = &dst_slice[(r - start_row) * (ne0 / QK1_0) + qb];
                        if (kt % 4 == 0) memset(blk->qs, 0, sizeof(blk->qs));
                        for (int j = 0; j < 32; j++) {
                            const uint8_t nib = tile_quants[row][j];
                            if (nib & 8) {
                                const int64_t e = (kt % 4) * 32 + j;
                                blk->qs[e / 8] |= (uint8_t) (1u << (e % 8));
                            }
                        }
                        blk->d = scale_src[row];
                    }
                }
            }
        }
    }
}

// flow officiel : set_tensor(repack) -> get_tensor(inverse fix7) -> dequant vs attendu
static int run_official_flow(int K, int M) {
    const size_t row_q1 = row_size_q1_0(K);
    const size_t src_bytes = row_q1 * M;
    std::vector<uint8_t> src(src_bytes);
    std::vector<float> expect((size_t) K * M);
    for (int r = 0; r < M; r++) {
        block_q1_0 * blks = (block_q1_0 *) (src.data() + (size_t) r * row_q1);
        for (int qb = 0; qb < K / 128; qb++) {
            block_q1_0 * b = &blks[qb];
            float sum = 0.0f;
            for (int j = 0; j < 128; j++) sum += fabsf(frand());
            const float d = sum / 128.0f;
            b->d = fp32_to_fp16(d);
            memset(b->qs, 0, 16);
            for (int j = 0; j < 128; j++) {
                const int bit = frand() >= 0.0f ? 1 : 0;
                expect[(size_t) r * K + qb * 128 + j] = bit ? fp16_to_fp32(b->d) : -fp16_to_fp32(b->d);
                if (bit) b->qs[j / 8] |= (uint8_t) (1u << (j % 8));
            }
        }
    }
    const int64_t ne0p = hex_round_up(K, 32), ne1p = hex_round_up(M, 32);
    const size_t tiled_bytes = (size_t) (ne0p / 32) * (ne1p / 32) * 576;
    std::vector<uint8_t> tiled(tiled_bytes);
    fake_tensor t = { { K, M, 1, 1 }, tiled.data() };
    repack_q1_0_as_q4_0_tiled(&t, src.data(), 0, src_bytes);          // set_tensor
    std::vector<uint8_t> back(src_bytes);
    repack_tiled_q1_0_fix7(back.data(), &t, 0, src_bytes);             // get_tensor (fix7)
    int nbad = 0; double maxerr = 0.0;
    std::vector<float> dq(128);
    for (int r = 0; r < M; r++) {
        const block_q1_0 * blks = (const block_q1_0 *) (back.data() + (size_t) r * row_q1);
        for (int qb = 0; qb < K / 128; qb++) {
            dequant_q1_0(&blks[qb], dq.data());
            for (int j = 0; j < 128; j++) {
                const int e = qb * 128 + j;
                const double err = fabs((double) dq[j] - expect[(size_t) r * K + e]);
                if (err > maxerr) maxerr = err;
                if (err > 1e-4 && nbad < 6)
                    printf("  [flow K=%d M=%d] r=%d e=%3d attendu=%+.5f obtenu=%+.5f\n",
                           K, M, r, e, expect[(size_t) r * K + e], dq[j]);
                if (err > 1e-4) nbad++;
            }
        }
    }
    printf("[flow K=%d M=%d] max_err=%.6f bad=%d %s\n", K, M, maxerr, nbad, nbad ? "FAIL" : "PASS");
    return nbad;
}

static int run_case(int K, int M, bool control_q4_0) {
    const size_t row_src = control_q4_0 ? row_size_q4_0(K) : row_size_q1_0(K);
    const size_t src_bytes = row_src * M;

    std::vector<uint8_t> src(src_bytes);
    std::vector<float> expect((size_t) K * M);   // valeur attendue apres dequant

    for (int r = 0; r < M; r++) {
        uint8_t * rp = src.data() + (size_t) r * row_src;
        if (control_q4_0) {
            block_q4_0 * blks = (block_q4_0 *) rp;
            for (int qb = 0; qb < K / 32; qb++) {
                block_q4_0 * b = &blks[qb];
                float amax = 0.0f;
                float x[32];
                for (int j = 0; j < 32; j++) { x[j] = frand(); amax = std::max(amax, fabsf(x[j])); }
                const float d = amax / 8.0f;
                b->d = fp32_to_fp16(d);
                memset(b->qs, 0, 16);
                for (int j = 0; j < 32; j++) {
                    const int q = d > 0 ? (int) roundf(x[j] / d) : 0;
                    const uint8_t nib = (uint8_t) std::min(15, std::max(0, q + 8));
                    expect[(size_t) r * K + qb * 32 + j] = ((int) nib - 8) * fp16_to_fp32(b->d);
                    if (j < 16) b->qs[j] |= (nib & 0xF);
                    else        b->qs[j - 16] |= (nib << 4);
                }
            }
        } else {
            block_q1_0 * blks = (block_q1_0 *) rp;
            for (int qb = 0; qb < K / 128; qb++) {
                block_q1_0 * b = &blks[qb];
                float sum = 0.0f;
                float x[128];
                for (int j = 0; j < 128; j++) { x[j] = frand(); sum += fabsf(x[j]); }
                const float d = sum / 128.0f;
                b->d = fp32_to_fp16(d);
                memset(b->qs, 0, 16);
                for (int j = 0; j < 128; j++) {
                    const int bit = x[j] >= 0.0f ? 1 : 0;
                    expect[(size_t) r * K + qb * 128 + j] = bit ? fp16_to_fp32(b->d) : -fp16_to_fp32(b->d);
                    if (bit) b->qs[j / 8] |= (uint8_t) (1u << (j % 8));
                }
            }
        }
    }

    // forward -> tiled
    const int64_t ne0p = hex_round_up(K, 32), ne1p = hex_round_up(M, 32);
    const size_t tiled_bytes = (size_t) (ne0p / 32) * (ne1p / 32) * 576;
    std::vector<uint8_t> tiled(tiled_bytes, 0xAA);
    fake_tensor t = { { K, M, 1, 1 }, tiled.data() };
    if (control_q4_0) repack_q4_0_tiled(&t, src.data(), 0, src_bytes);
    else              repack_q1_0_as_q4_0_tiled(&t, src.data(), 0, src_bytes);

    // inverse upstream -> blocs q4_0
    std::vector<uint8_t> back(row_size_q4_0(K) * M);
    fake_tensor t2 = { { K, M, 1, 1 }, tiled.data() };
    repack_tiled_q4_0(back.data(), &t2, 0, back.size());

    // dequant et comparaison
    int nbad = 0; double maxerr = 0.0;
    std::vector<float> dq(32);
    for (int r = 0; r < M; r++) {
        const block_q4_0 * blks = (const block_q4_0 *) (back.data() + (size_t) r * row_size_q4_0(K));
        for (int qb = 0; qb < K / 32; qb++) {
            dequant_q4_0(&blks[qb], dq.data());
            for (int j = 0; j < 32; j++) {
                const int e = qb * 32 + j;
                const double err = fabs((double) dq[j] - expect[(size_t) r * K + e]);
                if (err > maxerr) maxerr = err;
                if (err > 1e-4 && nbad < 8) {
                    printf("  [%s K=%d M=%d] r=%d e=%3d attendu=%+.5f obtenu=%+.5f\n",
                           control_q4_0 ? "ctrl" : "q1_0", K, M, r, e, expect[(size_t) r * K + e], dq[j]);
                    nbad++;
                }
            }
        }
    }
    printf("[%s K=%d M=%d] max_err=%.6f bad=%d %s\n",
           control_q4_0 ? "ctrl" : "q1_0", K, M, maxerr, nbad, nbad ? "FAIL" : "PASS");
    return nbad;
}

int main() {
    int total = 0;
    // flow officiel d'abord (set -> get -> dequant)
    total += run_official_flow(256, 16);
    total += run_official_flow(128, 16);
    total += run_official_flow(512, 5);
    // controle : round-trip upstream q4_0 doit etre exact
    total += run_case(256, 16, true);
    total += run_case(256, 5,  true);   // ne1 non multiple de 32 (padding)
    // Q1_0 via mon repack + inverse upstream
    total += run_case(256, 16, false);
    total += run_case(128, 16, false);  // 1 seul bloc q1_0 par ligne
    total += run_case(512, 5,  false);  // padding + 4 blocs
    total += run_case(256, 33, false);  // 2 col-tiles
    // ---- test decisif : repack_q1_0(Q1_0) == repack_q4_0(matrice q4_0 equivalente), octet a octet
    {
        const int K = 256, M = 33;
        const size_t row_q1 = row_size_q1_0(K);
        std::vector<uint8_t> src_q1(row_q1 * M);
        // matrice q4_0 equivalente : chaque bloc 32 = +-d du bloc q1_0 couvrant
        const size_t row_q4 = row_size_q4_0(K);
        std::vector<uint8_t> src_eq(row_q4 * M);
        for (int r = 0; r < M; r++) {
            block_q1_0 * b1 = (block_q1_0 *) (src_q1.data() + (size_t) r * row_q1);
            block_q4_0 * b4 = (block_q4_0 *) (src_eq.data() + (size_t) r * row_q4);
            for (int qb = 0; qb < K / 128; qb++) {
                float sum = 0.0f;
                for (int j = 0; j < 128; j++) sum += fabsf(frand());
                const float d = sum / 128.0f;
                b1[qb].d = fp32_to_fp16(d);
                memset(b1[qb].qs, 0, 16);
                for (int k32 = 0; k32 < 4; k32++) {
                    b4[qb * 4 + k32].d = b1[qb].d;
                    memset(b4[qb * 4 + k32].qs, 0, 16);
                }
                for (int j = 0; j < 128; j++) {
                    const int bit = frand() >= 0.0f ? 1 : 0;
                    if (bit) b1[qb].qs[j / 8] |= (uint8_t) (1u << (j % 8));
                    const int k32 = j / 32, jj = j % 32;
                    const uint8_t nib = bit ? 9 : 7;
                    if (jj < 16) b4[qb * 4 + k32].qs[jj] |= nib;
                    else         b4[qb * 4 + k32].qs[jj - 16] |= (uint8_t) (nib << 4);
                }
            }
        }
        const int64_t ne0p = hex_round_up(K, 32), ne1p = hex_round_up(M, 32);
        const size_t tiled_bytes = (size_t) (ne0p / 32) * (ne1p / 32) * 576;
        std::vector<uint8_t> tiled_a(tiled_bytes), tiled_b(tiled_bytes);
        fake_tensor t = { { K, M, 1, 1 }, tiled_a.data() };
        repack_q1_0_as_q4_0_tiled(&t, src_q1.data(), 0, src_q1.size());
        fake_tensor t2 = { { K, M, 1, 1 }, tiled_b.data() };
        repack_q4_0_tiled(&t2, src_eq.data(), 0, src_eq.size());
        int ndiff = 0;
        for (size_t i = 0; i < tiled_bytes && ndiff < 8; i++) {
            if (tiled_a[i] != tiled_b[i]) {
                printf("  [egalite] offset %zu : q1_0=%02x q4_0_eq=%02x\n", i, tiled_a[i], tiled_b[i]);
                ndiff++;
            }
        }
        printf("[egalite octets K=%d M=%d] %s (%d octets differents)\n",
               K, M, ndiff == 0 ? "IDENTIQUE - plomberie host validee" : "DIFFERE", ndiff);
        total += ndiff;
    }

    printf(total == 0 ? "\nTOUT OK\n" : "\nECHECS: %d\n", total);
    return total != 0;
}

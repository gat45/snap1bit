#!/usr/bin/env python3
# Insertion de repack_q1_0_as_q4_0_tiled dans ggml-hexagon-fastrpc.cpp
# (apres repack_q6k_as_q4_0_tiled). Conversion lossless directe :
#   bit=1 -> nibble 9 (+1), bit=0 -> nibble 7 (-1)
#   scale q4_0 (par 32) = d du super-bloc q1_0 (garde d-uniforme cote host)
# Layout tile Q4_0 32x32 : 512 octets nibbles (16 cp x 32 rows) + 64 octets scales.
# offset/size decoupent le fichier en slices ne1 x row_bytes (meme contrat
# que repack_q4_0_tiled, tire de la lecture de son code).
import sys

FASTRPC = "ggml/src/ggml-hexagon/ggml-hexagon-fastrpc.cpp"
FUNC = """
// Q1_0 weights are converted to Q4_0 (LOSSLESS, no dequant roundtrip):
// bit=1 -> nibble 9 (+1), bit=0 -> nibble 7 (-1); per-32 scale = super-block d.
// dequant(q4_0) = (nibble - 8) * d = +/-d exactly, so the NPU Q4_0 matmul
// kernels compute the exact Q1_0 dot product (see supported_mul_mat guard:
// conversion is exact only for uniform d across the 4 sub-blocks; host gate
// rejects non-uniform d before any repack happens).
// Tile layout is identical to repack_q4_0_tiled: 512B nibbles + 64B F16 scales.
static void repack_q1_0_as_q4_0_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    const size_t slice_size = (size_t)ne1 * ggml_row_size(t->type, ne0);
    int64_t start_slice = (int64_t)(offset / slice_size);
    int64_t end_slice   = (int64_t)((offset + size + slice_size - 1) / slice_size);
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const block_q1_0 * src_slice = (const block_q1_0 *) data;
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = 0; ct < n_col_tiles; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                uint8_t tile_quants[32][32];
                ggml_half tile_scales[32];
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r < ne1 && kt < ne0 / QK1_0) {
                        const block_q1_0 * blk = &src_slice[r * (ne0 / QK1_0) + kt];
                        // expand 128 bits -> 32 nibbles {7,9}
                        for (int cp = 0; cp < 16; cp++) {
                            const uint8_t byte = blk->qs[cp];
                            tile_quants[row][2 * cp + 0] = 7 + 2 * ((byte >> 0) & 1);
                            tile_quants[row][2 * cp + 1] = 7 + 2 * ((byte >> 1) & 1);
                            tile_quants[row][2 * cp + 2] = 7 + 2 * ((byte >> 2) & 1);
                            tile_quants[row][2 * cp + 3] = 7 + 2 * ((byte >> 3) & 1);
                            tile_quants[row][2 * cp + 4] = 7 + 2 * ((byte >> 4) & 1);
                            tile_quants[row][2 * cp + 5] = 7 + 2 * ((byte >> 5) & 1);
                            tile_quants[row][2 * cp + 6] = 7 + 2 * ((byte >> 6) & 1);
                            tile_quants[row][2 * cp + 7] = 7 + 2 * ((byte >> 7) & 1);
                        }
                        tile_scales[row] = blk->d;
                    } else {
                        memset(tile_quants[row], 8, 32);  // zero nibble
                        tile_scales[row] = 0;
                    }
                }

                for (int cp = 0; cp < 16; cp++) {
                    for (int row = 0; row < 32; row++) {
                        tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                    }
                }

                ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                for (int row = 0; row < 32; row++) {
                    scale_dst[row] = tile_scales[row];
                }
            }
        }
    }
}

"""
with open(FASTRPC, "r", encoding="utf-8") as f:
    src = f.read()

ANCHOR = """// Q5_K weights are converted to Q4_0 (dequant Q5_K -> f32 -> requant Q4_0)"""
if FUNC not in src:
    if ANCHOR not in src:
        print("[FAIL] ancre Q5_K introuvable"); sys.exit(1)
    src = src.replace(ANCHOR, FUNC + ANCHOR, 1)
    with open(FASTRPC, "w", encoding="utf-8") as f:
        f.write(src)
    print("[OK] repack_q1_0_as_q4_0_tiled insere")
else:
    print("[SKIP] deja present")

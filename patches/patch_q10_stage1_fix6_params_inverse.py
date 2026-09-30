#!/usr/bin/env python3
# Fix 6 (dspqueue, ggml-hexagon.cpp) - deux corrections :
#
# (A) CAUSE RACINE du 0/32 : les helpers de params HVX/HMX recoivent
#     src0->type BRUT (Q1_0=41) alors que le DSP stocke en Q4_0 tiled.
#     htp_mm_get_weight_tile_size(41) -> default: 0  =>  kparams->tile_size = 0
#     => kernels parametres a zero => garbage pour TOUTES les tailles.
#     Q4_K passe car HTP_TYPE_Q4_K a son cas dans les switches.
#     Fix : presenter le type de STOCKAGE (Q4_0) a toute la machinerie de
#     params via un helper ggml_hexagon_weight_storage_type(), applique aux
#     deux sites (precompute_matmul_params_impl + precompute_fused_mmnx_params).
#
# (B) get_tensor (inverse) : le switch n'a pas de case Q1_0 -> relecture
#     garbage apres le test (get-tensor a flags 0x1). Ajout de
#     repack_tiled_q1_0 (inverse de repack_q1_0_as_q4_0_tiled) + case.
import sys

def patch(path, edits):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    for old, new, cnt in edits:
        n = src.count(old)
        if n == 0:
            print("[FAIL] %s: %r" % (path, old[:70])); sys.exit(1)
        if cnt != -1 and n != cnt:
            print("[FAIL] %s: x%d (attendu %d)" % (path, n, cnt)); sys.exit(1)
        src = src.replace(old, new, 1 if cnt == 1 else -1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("[OK] %s: %d editions" % (path, len(edits)))

HEX = "ggml/src/ggml-hexagon/ggml-hexagon.cpp"

HELPER = """
// Type de STOCKAGE DSP pour les types convertis au repack (Q1_0 -> Q4_0 tiled).
// Toute la machinerie de params HVX/HMX (tile sizes, strides, layouts) doit
// voir le type de stockage, pas le type logique ggml : sinon les switches
// htp_mm_get_weight_tile_size etc. tombent dans default -> tile_size = 0.
static inline enum ggml_type ggml_hexagon_weight_storage_type(enum ggml_type type) {
    return type == GGML_TYPE_Q1_0 ? GGML_TYPE_Q4_0 : type;
}
"""

INVERSE = """
// repack q4_0_tiled tensor into q1_0 data (inverse pour get_tensor, 28/09)
static void repack_tiled_q1_0(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    block_q1_0 * dst_matrix = (block_q1_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    size_t slice_size = ne1 * ggml_row_size(t->type, ne0);
    size_t row_size_bytes = ggml_row_size(t->type, ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

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

                const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
                const int64_t qb = kt / 4;
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / QK1_0) {
                        block_q1_0 * blk = &dst_slice[(r - start_row) * (ne0 / QK1_0) + qb];
                        memset(blk->qs, 0, sizeof(blk->qs));
                        for (int j = 0; j < 32; j++) {
                            const uint8_t nib = tile_quants[row][j];
                            // nibble 7 (-1) -> bit 0, nibble 9 (+1) -> bit 1
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
"""

patch(HEX, [
    # (A1) helper juste apres is_repack_type
    ("""           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q1_0; // custom : stocke en Q4_0 tiled (lossless)
}
""",
     """           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q1_0; // custom : stocke en Q4_0 tiled (lossless)
}
""" + HELPER, 1),

    # (A2) les DEUX sites de params : wtype logique -> wtype de stockage
    ("""    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);
""",
     """    // Q1_0 custom : presenter le type de STOCKAGE (Q4_0) aux helpers de
    // params (sinon tile_size = 0 -> kernels HVX/HMX parametres a zero).
    const int wtype = (int) ggml_hexagon_weight_storage_type(src0->type);
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);
""", -1),

    # (B1) fonction inverse apres repack_q1_0_as_q4_0_tiled
    ("""                    } else {
                        memset(tile_quants[row], 8, 32);       // element nul
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
""",
     """                    } else {
                        memset(tile_quants[row], 8, 32);       // element nul
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
""" + INVERSE, 1),

    # (B2) case Q1_0 dans le switch get_tensor (inverse)
    ("""        case GGML_TYPE_IQ4_NL:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q4_0(data, tensor, offset, size);
            break;
""",
     """        case GGML_TYPE_IQ4_NL:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q4_0(data, tensor, offset, size);
            break;

        case GGML_TYPE_Q1_0: // custom : inverse Q4_0 tiled -> Q1_0
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q1_0(data, tensor, offset, size);
            break;
""", 1),
])
print("FIX6_OK")

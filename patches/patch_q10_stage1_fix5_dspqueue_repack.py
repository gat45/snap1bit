#!/usr/bin/env python3
# Fix 5 : chemin dspqueue - les repacks se font via des fonctions LOCALES de
# ggml-hexagon.cpp (differente de fastrpc.cpp !). Ajout de :
#   - repack_q1_0_as_q4_0_tiled (locale, apres repack_q4_0_tiled :880)
#   - case Q1_0 dans repack_tensor_tiled
#   - case Q1_0 dans le switch get_tensor (inverse)
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

FUNC = """
// Q1_0 -> Q4_0 tiled, LOSSLESS (un element = un nibble : 7 + 2*bit, i.e.
// +1/-1 bipolaire ; la scale de la tuile 32-wide = d du bloc q1_0 couvrant,
// chaque tuile de 32 etant contenue dans un seul bloc de 128 puisque
// 128 % 32 == 0). Les padding rows/k-tiles sont remplis de nibbles 8
// (zero) avec scale 0. Layout identique a repack_q4_0_tiled.
static void repack_q1_0_as_q4_0_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    const block_q1_0 * src_matrix = (const block_q1_0 *) data;
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
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const block_q1_0 * src_slice = src_matrix + (slice_idx - start_slice) * (ne1 * (ne0 / QK1_0));
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = 0; ct < n_col_tiles; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                uint8_t tile_quants[32][32];
                ggml_half tile_scales[32];
                const int64_t qb  = kt / 4;                   // bloc 128 couvrant ce k-tile
                const int64_t bofs = (kt % 4) * 32;            // premier element du k-tile dans le bloc
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r < ne1 && qb < ne0 / QK1_0) {
                        const block_q1_0 * blk = &src_slice[r * (ne0 / QK1_0) + qb];
                        const uint8_t * qsb = blk->qs;
                        for (int j = 0; j < 32; j++) {
                            const int64_t e = bofs + j;        // element global dans le bloc
                            tile_quants[row][j] = (uint8_t)(7 + 2 * ((qsb[e / 8] >> (e % 8)) & 1));
                        }
                        tile_scales[row] = blk->d;
                    } else {
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

"""

patch(HEX, [
    # 1) inserer la fonction apres repack_q4_0_tiled (fin : avant la fonction suivante)
    ("""static void repack_q4_1_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {""",
     FUNC + """static void repack_q4_1_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {""", 1),

    # 2) case Q1_0 dans repack_tensor_tiled
    ("""        case GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, data, 0, size);
            break;

        case GGML_TYPE_MXFP4:""",
     """        case GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, data, 0, size);
            break;

        case GGML_TYPE_Q1_0: // custom : lossless vers tuiles Q4_0
            repack_q1_0_as_q4_0_tiled(tensor, data, 0, size);
            break;

        case GGML_TYPE_MXFP4:""", 1),
])
print("FIX5_OK")

#!/usr/bin/env python3
# Correction 2 du repack Q1_0 :
#  1. un element Q1_0 = UN nibble entier (7 + 2*bit) : nibble=9 -> +d, 7 -> -d.
#     (l'ancienne version appairait 2 elements par nibble : FAUX pour q4_0,
#     qui stocke 1 element par nibble, 2 elements par octet)
#  2. scale de chaque tuile 32-wide = d du bloc q1_0 couvrant (kt/4) :
#     128 % 32 == 0, donc chaque k-tile est contenu dans UN seul bloc -> la
#     conversion est exacte pour TOUT fichier Q1_0, sans garde d-uniforme.
import sys

def patch(path, edits):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    for old, new, cnt in edits:
        n = src.count(old)
        if n == 0:
            print("[FAIL] %s: pattern introuvable: %r" % (path, old[:70])); sys.exit(1)
        if cnt != -1 and n != cnt:
            print("[FAIL] %s: pattern x%d (attendu %d)" % (path, n, cnt)); sys.exit(1)
        src = src.replace(old, new, 1 if cnt == 1 else -1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("[OK] %s: %d editions" % (path, len(edits)))

HEX = "ggml/src/ggml-hexagon/ggml-hexagon.cpp"
FASTRPC = "ggml/src/ggml-hexagon/ggml-hexagon-fastrpc.cpp"

# --- ggml-hexagon.cpp : le garde d-uniformite est inutile (scale par tuile
# --= d du bloc couvrant) -> remplace par un commentaire de contrat.
patch(HEX, [
    ("""    // Q1_0 custom (2026-09-28, plan kernel 8-bis) : le chemin repack convertit
    // lossless Q1_0 -> Q4_0 (nibbles {7,9}, scale d conserve par bloc de 32 :
    // dequant q4_0 = (nibble-8)*d = +d ou -d exactement). Cette conversion est
    // EXACTE seulement si d est constant sur les 4 sous-blocs q4_0 de chaque
    // super-bloc q1_0 (cas des GGUF quantises directement en Q1_0). Sinon,
    // repli CPU explicite (jamais un faux offload).
    if (src0->type == GGML_TYPE_Q1_0) {
#ifndef Q1_0_KEEPQ40
        const int64_t nb1 = src0->ne[0] / QK1_0;
        const block_q1_0 * bq = (const block_q1_0 *) src0->data;
        for (int64_t b = 0; b < nb1; b += 4) {
            const ggml_half d0 = bq[b].d;
            for (int64_t sb = 1; sb < 4 && b + sb < nb1; sb++) {
                if (bq[b + sb].d != d0) {
                    HEX_VERBOSE("ggml-hex: %s reject Q1_0 non-uniform d at block %lld\\n",
                                sess->c_name(), (long long)(b + sb));
                    return false;
                }
            }
        }
#endif
    }

    return true;""",
     """    // Q1_0 custom (2026-09-28, plan kernel 8-bis) : le chemin repack convertit
    // lossless Q1_0 -> Q4_0. Exactitude par construction : chaque tuile 32x32
    // (k-range [kt*32, kt*32+32[) est contenue dans UN SEUL bloc q1_0 de 128
    // (128 % 32 == 0), la scale q4_0 de la tuile = d de ce bloc, et
    // dequant(q4_0) = (nibble-8)*d = +d ou -d exactement (nibble 9 = +1,
    // 7 = -1). Valable pour TOUT fichier Q1_0, aucune condition d-uniforme
    // (voir repack_q1_0_as_q4_0_tiled).

    return true;""", 1),
])

# --- fastrpc.cpp : header comment, dsp_type comment, corps du mapping ---
patch(FASTRPC, [
    ("""// Q1_0 weights are converted to Q4_0 (LOSSLESS, no dequant roundtrip):
// bit=1 -> nibble 9 (+1), bit=0 -> nibble 7 (-1); per-32 scale = super-block d.
// dequant(q4_0) = (nibble - 8) * d = +/-d exactly, so the NPU Q4_0 matmul
// kernels compute the exact Q1_0 dot product (see supported_mul_mat guard:
// conversion is exact only for uniform d across the 4 sub-blocks; host gate
// rejects non-uniform d before any repack happens).
// Tile layout is identical to repack_q4_0_tiled: 512B nibbles + 64B F16 scales.""",
     """// Q1_0 weights are converted to Q4_0 (LOSSLESS, no dequant roundtrip):
// bit=1 -> nibble 9 (+1), bit=0 -> nibble 7 (-1), ONE element per nibble.
// The scale of each 32-wide k-tile = d of the covering 128-bit q1_0 block
// (every 32-tile lies inside a single block since 128 % 32 == 0).
// dequant(q4_0) = (nibble - 8) * d = +/-d exactly, so the NPU Q4_0 matmul
// kernels compute the exact Q1_0 dot product for ANY q1_0 file (no
// uniformity assumption).
// Tile layout is identical to repack_q4_0_tiled: 512B nibbles + 64B F16 scales.""", 1),

    ("""                uint8_t tile_quants[32][32];
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
                }""",
     """                uint8_t tile_quants[32][32];
                ggml_half tile_scales[32];
                const int64_t nb1_row = ne0 / QK1_0;       // blocs q1_0 par ligne
                const int64_t qb  = kt / 4;                // bloc couvrant ce k-tile de 32
                const int32_t bofs = (int32_t)(kt % 4) * 32;  // bit de depart dans qs
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r < ne1 && qb < nb1_row) {
                        const block_q1_0 * blk = &src_slice[r * nb1_row + qb];
                        const uint8_t * qsb = blk->qs;
                        // 1 element = 1 nibble : nibble = 7 + 2*bit (+1 / -1)
                        for (int j = 0; j < 32; j++) {
                            const int e = bofs + j;
                            tile_quants[row][j] = (uint8_t)(7 + 2 * ((qsb[e / 8] >> (e % 8)) & 1));
                        }
                        tile_scales[row] = blk->d;         // scale exacte de la tuile
                    } else {
                        memset(tile_quants[row], 8, 32);   // zero nibble
                        tile_scales[row] = 0;
                    }
                }""", 1),

    ("""    if (type == GGML_TYPE_Q1_0) return GGML_TYPE_Q4_0; // custom : lossless (nibbles {7,9}, d constant)""",
     """    if (type == GGML_TYPE_Q1_0) return GGML_TYPE_Q4_0; // custom : lossless (nibbles {7,9}, scale = d du bloc couvrant)""", 1),
])
print("CORRECTION2_OK")

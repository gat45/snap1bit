#!/usr/bin/env python3
# Fix 7 : bug de l'inverse repack_tiled_q1_0 (fix6) - le memset du bloc q1_0
# etait execute pour CHAQUE k-tile alors qu'un bloc q1_0 couvre 4 k-tiles
# (128 % 32 == 0) -> chaque memset effacait les bits poses par les k-tiles
# precedentes. Seul le dernier quart du bloc survivait dans la copie CPU.
# Le memset ne doit se faire qu'au premier k-tile du bloc (kt % 4 == 0),
# et le scale doit n'etre reecrit qu'une seule fois aussi (inoffensif sinon).
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

HEX = "/home/videl/up-b11855/ggml/src/ggml-hexagon/ggml-hexagon.cpp"

OLD = """                const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
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
                }"""

NEW = """                const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
                const int64_t qb = kt / 4;
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / QK1_0) {
                        block_q1_0 * blk = &dst_slice[(r - start_row) * (ne0 / QK1_0) + qb];
                        // un bloc q1_0 couvre 4 k-tiles : initialisation uniquement
                        // au premier k-tile, sinon les bits des k-tiles precedentes
                        // seraient effaces (fix7)
                        if (kt % 4 == 0) {
                            memset(blk->qs, 0, sizeof(blk->qs));
                        }
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
                }"""

patch(HEX, [(OLD, NEW, 1)])
print("FIX7_OK")

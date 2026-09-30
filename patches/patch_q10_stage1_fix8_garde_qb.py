#!/usr/bin/env python3
# Fix 8 : dans repack_tiled_q1_0 (fix6), la garde `kt < ne0 / QK1_0` est
# fausse : kt est l'index de TUtile (32 elems) alors que ne0/QK1_0 compte des
# BLOCS de 128. Seuls les premiers k-tiles etaient convertis (e.g. K=256 :
# kt<2 -> elements 0-63 OK, 64-255 corrompus). La bonne garde est sur le
# bloc couvrant : qb = kt/4 < ne0/QK1_0 (meme semantique que le forward).
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

OLD = """                const int64_t qb = kt / 4;
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / QK1_0) {
                        block_q1_0 * blk = &dst_slice[(r - start_row) * (ne0 / QK1_0) + qb];
                        // un bloc q1_0 couvre 4 k-tiles : initialisation uniquement
                        // au premier k-tile, sinon les bits des k-tiles precedentes
                        // seraient effaces (fix7)
                        if (kt % 4 == 0) {
                            memset(blk->qs, 0, sizeof(blk->qs));
                        }"""

NEW = """                const int64_t qb = kt / 4;
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    // garde sur le BLOC couvrant (fix8) : kt est une tuile de 32,
                    // ne0/QK1_0 compte des blocs de 128 -> qb, pas kt
                    if (r >= start_row && r < end_row && qb < ne0 / QK1_0) {
                        block_q1_0 * blk = &dst_slice[(r - start_row) * (ne0 / QK1_0) + qb];
                        // un bloc q1_0 couvre 4 k-tiles : initialisation uniquement
                        // au premier k-tile, sinon les bits des k-tiles precedentes
                        // seraient effaces (fix7)
                        if (kt % 4 == 0) {
                            memset(blk->qs, 0, sizeof(blk->qs));
                        }"""

patch(HEX, [(OLD, NEW, 1)])
print("FIX8_OK")

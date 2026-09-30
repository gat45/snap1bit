#!/usr/bin/env python3
# Fix 4 : brancher Q1_0 sur la machinerie REPACK du chemin dspqueue
#  (a) is_repack_type (+Q1_0) -> REPACK flag + alloc repackee
#  (b) tiled_row_size (cas Q1_0 = geometrie tuile Q4_0 : (ne0/32)*18)
#  (c) same_shape : comparer le type de STOCKAGE (Q1_0 stocke en Q4_0)
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
patch(HEX, [
    # (a) is_repack_type : Q1_0 suit le chemin repack (stocke en Q4_0 tiled)
    ("""static inline bool ggml_hexagon_is_repack_type(enum ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
           type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q6_K ||
           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K;
}""",
     """static inline bool ggml_hexagon_is_repack_type(enum ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
           type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q6_K ||
           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q1_0; // custom : stocke en Q4_0 tiled (lossless)
}""", 1),

    # (b) tiled_row_size : Q1_0 utilise la geometrie de tuile Q4_0
    ("""static inline size_t ggml_hexagon_tiled_row_size(enum ggml_type type, int64_t ne0) {
    if (type == GGML_TYPE_Q6_K) {
        return (size_t) (ne0 / 32) * (HTP_MM_WEIGHT_TILE_SIZE_Q6_K / 32);
    }""",
     """static inline size_t ggml_hexagon_tiled_row_size(enum ggml_type type, int64_t ne0) {
    if (type == GGML_TYPE_Q1_0) {
        // custom : stocke en tuiles Q4_0 (576 B par tuile 32x32)
        return (size_t) (ne0 / 32) * (HTP_MM_WEIGHT_TILE_SIZE_Q4_0 / 32);
    }
    if (type == GGML_TYPE_Q6_K) {
        return (size_t) (ne0 / 32) * (HTP_MM_WEIGHT_TILE_SIZE_Q6_K / 32);
    }""", 1),

    # (c) same_shape : comparaison sur le type de stockage
    ("""        return (h->type == t->type) &&""",
     """        // Q1_0 custom : h->type porte le type de STOCKAGE (Q4_0)
        const enum ggml_type t_stor = (t->type == GGML_TYPE_Q1_0) ? GGML_TYPE_Q4_0 : t->type;
        return (h->type == t_stor) &&""", 1),
])
print("FIX4_OK")

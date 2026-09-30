#!/usr/bin/env python3
# Patch Stage 1 kernel Q1_0 HTP : chemin conversion lossless Q1_0 -> Q4_0 tiled
# (plan §8 bis, 2026-09-28). Aucune apostrophe dans ce fichier pour bash-safe.
import sys

def patch(path, edits):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    for old, new, cnt in edits:
        n = src.count(old)
        if n == 0:
            print("[FAIL] %s: pattern introuvable: %r" % (path, old[:60])); sys.exit(1)
        if cnt != -1 and n != cnt:
            print("[FAIL] %s: pattern x%d (attendu %d): %r" % (path, n, cnt, old[:60])); sys.exit(1)
        src = src.replace(old, new, 1 if cnt == 1 else -1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("[OK] %s: %d editions" % (path, len(edits)))

HEX = "ggml/src/ggml-hexagon/ggml-hexagon.cpp"

GUARD = """    // Q1_0 custom (2026-09-28, plan kernel 8-bis) : le chemin repack convertit
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

    return true;
}

static bool ggml_hexagon_supported_mul_mat_id(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {"""

patch(HEX, [
    ("""    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:""",
     """    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q1_0: // custom Q1_0 : conversion lossless Q4_0 au repack (28/09)
        case GGML_TYPE_Q4_K:""", 1),

    ("""        case GGML_TYPE_Q1_0: // custom Q1_0 : conversion lossless Q4_0 au repack (28/09)
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            if (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                return false;
            }

            if (src0->ne[0] % ((src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q4_K) ? QK_K : 32)) {
                return false;
            }""",
     """        case GGML_TYPE_Q1_0: // custom Q1_0 : conversion lossless Q4_0 au repack (28/09)
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            if (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                return false;
            }

            if (src0->ne[0] % ((src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q4_K) ? QK_K :
                               (src0->type == GGML_TYPE_Q1_0 ? QK1_0 : 32))) {
                return false;
            }""", 1),

    ("""    return true;
}

static bool ggml_hexagon_supported_mul_mat_id(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {""",
     GUARD, 1),

])

FASTRPC = "ggml/src/ggml-hexagon/ggml-hexagon-fastrpc.cpp"
patch(FASTRPC, [
    ("""static inline enum ggml_type ggml_hexagon_weight_dsp_type(enum ggml_type type) {
    if (type == GGML_TYPE_BF16) return GGML_TYPE_F16;
    if (type == GGML_TYPE_Q4_K) return GGML_TYPE_Q4_0;""",
     """static inline enum ggml_type ggml_hexagon_weight_dsp_type(enum ggml_type type) {
    if (type == GGML_TYPE_BF16) return GGML_TYPE_F16;
    if (type == GGML_TYPE_Q1_0) return GGML_TYPE_Q4_0; // custom : lossless (nibbles {7,9}, d constant)
    if (type == GGML_TYPE_Q4_K) return GGML_TYPE_Q4_0;""", 1),

    # set_tensor : repack Q1_0 -> Q4_0 tiled (stockage DSP)
    ("""            case GGML_TYPE_MXFP4:
                if (dp >= base && dp < end) {
                    GGMLHEXAGON_LOG_DEBUG("repack");
                    repack_mxfp4_tiled(tensor, data, offset, size);
                } else {
                    GGMLHEXAGON_LOG_DEBUG("cpu buffer");
                    memcpy(tensor->data, data, size);
                }
                break;""",
     """            case GGML_TYPE_MXFP4:
                if (dp >= base && dp < end) {
                    GGMLHEXAGON_LOG_DEBUG("repack");
                    repack_mxfp4_tiled(tensor, data, offset, size);
                } else {
                    GGMLHEXAGON_LOG_DEBUG("cpu buffer");
                    memcpy(tensor->data, data, size);
                }
                break;
            case GGML_TYPE_Q1_0: // custom (2026-09-28) : conversion lossless Q1_0 -> Q4_0
                if (dp >= base && dp < end) {
                    GGMLHEXAGON_LOG_DEBUG("repack q1_0 as q4_0");
                    repack_q1_0_as_q4_0_tiled(tensor, data, offset, size);
                } else {
                    GGMLHEXAGON_LOG_DEBUG("cpu buffer");
                    memcpy(tensor->data, data, size);
                }
                break;""", 1),

    # get_alloc_size : allouer la taille repackee Q4_0 (x4 brut) pour Q1_0
    ("""    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: {
            size_t repacked = ggml_hexagon_repacked_size(tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);""",
     """    switch (tensor->type) {
        case GGML_TYPE_Q1_0: // custom : stocke en Q4_0 tiled (x4 brut)
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: {
            size_t repacked = ggml_hexagon_repacked_size(tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);""", 1),

    # supports_op (MUL_MAT check) : Q1_0 doit franchir le check scheduler
    ("""    switch (src0->type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        {
            if (src0->ne[0] % ((src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q6_K) ? QK_K : 32)) {
                return false;
            }""",
     """    switch (src0->type) {
        case GGML_TYPE_Q1_0: // custom : lossless Q4_0 repack (voir supported_mul_mat)
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        {
            if (src0->ne[0] % ((src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q6_K) ? QK_K :
                               (src0->type == GGML_TYPE_Q1_0 ? QK1_0 : 32))) {
                return false;
            }""", 1),
])
print("PATCH_COMPLET")

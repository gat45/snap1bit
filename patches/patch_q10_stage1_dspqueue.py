#!/usr/bin/env python3
# Patch 3 : chemin dspqueue (add_tensor) - le type envoye au DSP doit etre le
# type de STOCKAGE (Q4_0 pour un poids Q1_0 repacke), pas le type logique.
import sys

def patch(path, edits):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    for old, new, cnt in edits:
        n = src.count(old)
        if n == 0:
            print("[FAIL] %s: pattern introuvable: %r" % (path, old[:70])); sys.exit(1)
        if cnt != -1 and n != cnt:
            print("[FAIL] %s: x%d (attendu %d)" % (path, n, cnt)); sys.exit(1)
        src = src.replace(old, new, 1 if cnt == 1 else -1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("[OK] %s: %d editions" % (path, len(edits)))

HEX = "ggml/src/ggml-hexagon/ggml-hexagon.cpp"
patch(HEX, [
    ("""        htp_tensor &h = h_tens[ti];
        h.bi    = add_buffer(sbuf);
        h.ti    = ti;
        h.data  = t_offset;
        h.type  = t->type;""",
     """        htp_tensor &h = h_tens[ti];
        h.bi    = add_buffer(sbuf);
        h.ti    = ti;
        h.data  = t_offset;
        // Q1_0 custom : le DSP doit voir le type de STOCKAGE (Q4_0 repacke),
        // pas le type logique ggml (sinon dsp-error NO-SUPPORT sur le type 41).
        h.type  = ggml_hexagon_weight_dsp_type(t->type);""", 1),
])
print("PATCH3_OK")

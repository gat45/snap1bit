#!/usr/bin/env python3
# Fix compilation : mapping local minimal (la fonction est declaree dans
# fastrpc.cpp, invisible ici). Seul Q1_0 doit etre remappe dans ce chemin.
import sys

HEX = "ggml/src/ggml-hexagon/ggml-hexagon.cpp"
with open(HEX, "r", encoding="utf-8") as f:
    src = f.read()

OLD = """        // Q1_0 custom : le DSP doit voir le type de STOCKAGE (Q4_0 repacke),
        // pas le type logique ggml (sinon dsp-error NO-SUPPORT sur le type 41).
        h.type  = ggml_hexagon_weight_dsp_type(t->type);"""
NEW = """        // Q1_0 custom : le DSP doit voir le type de STOCKAGE (Q4_0 repacke),
        // pas le type logique ggml (sinon dsp-error NO-SUPPORT sur le type 41).
        h.type  = (t->type == GGML_TYPE_Q1_0) ? GGML_TYPE_Q4_0 : t->type;"""

if OLD in src:
    src = src.replace(OLD, NEW, 1)
    with open(HEX, "w", encoding="utf-8") as f:
        f.write(src)
    print("[OK] mapping local applique")
elif NEW in src:
    print("[SKIP] deja applique")
else:
    print("[FAIL] pattern introuvable"); sys.exit(1)

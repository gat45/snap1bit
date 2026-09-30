#!/usr/bin/env python3
# Fix 3 : 1 element Q1_0 = 1 NIBBLE (4 bits), pas 1 bit !
# bit b du fichier -> element b+bofs : nibble = 7 + 2*b (9=+1, 7=-1)
# Ancien code lisait qsb[e/8]>>(e%8) : element e = bit e -> facteur 4 trop lent
# et melange de nibbles. Correct : nibble e -> bits 4e..4e+3 ; pour un element
# binaire, les 4 bits du nibble e sont les bits 4e..4e+3 du bloc (element e*4..e*4+3)
# NON. La bonne lecture :
#   element binaire j (0..127) vit dans le NIBBLE j//4, position j%4
#   nibble j vaut 7 + 2*bit_j  (bit_j = bit de poids (j%4) de qs[j/4])
#   -> element j = signe (7 + 2*bit_{j}) et son bit source est (qs[j/4] >> (j%4))
import sys

FASTRPC = "ggml/src/ggml-hexagon/ggml-hexagon-fastrpc.cpp"
with open(FASTRPC, "r", encoding="utf-8") as f:
    src = f.read()

OLD = """                        // 1 element = 1 nibble : nibble = 7 + 2*bit (+1 / -1)
                        for (int j = 0; j < 32; j++) {
                            const int e = bofs + j;
                            bofs_unused_guard = 0;
                            tile_quants[row][j] = (uint8_t)(7 + 2 * ((qsb[e / 8] >> (e % 8)) & 1));
                        }"""
NEW = """                        // 1 element = 1 nibble : nibble e = 7 + 2*bit_e
                        // bit_e = bit (e%8) de qs[e/8] (e = bofs + j)
                        for (int j = 0; j < 32; j++) {
                            const int e = bofs + j;
                            tile_quants[row][j] = (uint8_t)(7 + 2 * ((qsb[e / 8] >> (e % 8)) & 1));
                        }"""

# (le code actuel n'a pas de garde ; pattern reel :)
OLD2 = """                        // 1 element = 1 nibble : nibble = 7 + 2*bit (+1 / -1)
                        for (int j = 0; j < 32; j++) {
                            const int e = bofs + j;
                            tile_quants[row][j] = (uint8_t)(7 + 2 * ((qsb[e / 8] >> (e % 8)) & 1));
                        }"""
NEW2 = """                        // 1 element = 1 nibble : nibble e = 7 + 2*bit_e, 1 bit source
                        // par element : bit_e = bit (e%8) de qs[e/8].
                        // ATTENTION : cette lecture est celle d'un fichier Q1_0
                        // où 1 element = 1 BIT. Le test qui suit re-verifie
                        // l'exactitude au vec_dot ; si le desaccord bipolaire
                        // persiste, c'est le contrat du FILE, pas ce mapping.
                        for (int j = 0; j < 2 * 16; j++) {
                            const int e = bofs + j;
                            tile_quants[row][j] = (uint8_t)(7 + 2 * ((qsb[e / 8] >> (e % 8)) & 1));
                        }"""
print("[SKIP] fix3 annule : le mapping bit->element est en realite CORRECT (1 bit = 1 element).")

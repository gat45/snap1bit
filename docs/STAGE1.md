# Q1_0 Stage 1 — kernel HTP v81 (Snapdragon 8 Elite Gen 5, Hexagon HTP v81)

Implémentation du support `GGML_TYPE_Q1_0` (type 41) pour le backend
`ggml-hexagon` (chemin dspqueue **et** fastrpc), par conversion lossless
Q1_0 → tuiles Q4_0 du repack existant. Validation : **32/32 tests
officiels MUL_MAT Q1_0**, **796/796 suite MUL_MAT complète**, bench modèle
réel sur device.

Base : 37f752b (`self-build-jz-b11855`).

## 1. Conception

Format Q1_0 : blocs de 128 éléments, `d` FP16 + 16 octets de bits de signe
(1,125 bpw). Référence CPU (`quantize_row_q1_0_ref`, ggml-quants.c:40) :
`d = Σ|x|/128`, bits LSB-first, valeurs **±d bipolaires**.

Mapping lossless vers tuiles Q4_0 (576 o = 512 nibbles + 64 o scales F16) :

- bit=1 → nibble 9 (+1), bit=0 → nibble 7 (−1), 1 élément = 1 nibble ;
- scale de chaque tuile 32-wide = `d` du bloc q1_0 **couvrant**
  (128 % 32 = 0 → chaque k-tile appartient à UN seul bloc → exact pour tout
  fichier Q1_0, sans garde d'uniformité) ;
- padding rows/k-tiles : nibbles 8 + scale 0.

## 2. Patches

### ggml-hexagon.cpp (chemin dspqueue)

| Élément | Modification |
|---|---|
| `ggml_hexagon_supported_mul_mat` | +case Q1_0, alignement QK1_0 |
| `ggml_hexagon_is_repack_type` | +Q1_0 |
| `ggml_hexagon_tiled_row_size` | cas Q1_0 = (ne0/32) × 18 |
| buffer type `get_alloc_size` | alloc repackée (×4) pour Q1_0 |
| `init_tensor` / `set_tensor` | flag REPACK (needs_repack ou usage WEIGHTS) |
| `add_tensor` | `h.type` = **type de stockage Q4_0** + géométrie repackée |
| `repack_tensor_tiled` | +case Q1_0 → `repack_q1_0_as_q4_0_tiled` |
| `get_tensor` (switch inverse) | +case Q1_0 → `repack_tiled_q1_0` (fix6) |
| `same_shape` | comparaison sur le type de stockage |
| `precompute_matmul_params_impl`, `precompute_fused_mmnx_params` | `wtype` = type de **stockage** via `ggml_hexagon_weight_storage_type()` (fix6) |

### ggml-hexagon-fastrpc.cpp (chemin MEMPOOL)

`weight_dsp_type` (+Q1_0→Q4_0), `supports_op` (+Q1_0, align QK1_0),
`get_alloc_size` (+Q1_0), `set_tensor` (+case Q1_0),
`repack_q1_0_as_q4_0_tiled`.

## 3. Historique des fixes (causes racines)

1. **NO-SUPPORT type 41** : `h.type` dspqueue non mappé → le DSP voyait le
   type brut. Corrigé (mapping local, la fonction fastrpc n'est pas visible).
2. **Garbage (ERR 1,3e9)** : `htp_mm_get_weight_tile_size(41)` →
   `default: 0` ; les params HVX/HMX recevaient le type brut.
   → **fix6** : helper `ggml_hexagon_weight_storage_type()` aux 2 sites de
   params + inverse `repack_tiled_q1_0`.
3. **CPU du test corrompu (ERR ~3,9 NMSE, non corrélé)** : le harnais
   officiel alimente son backend CPU de référence via `get_tensor` du backend
   testé → l'inverse était en jeu.
   - **fix7** : `memset(blk->qs)` exécuté pour chaque k-tile alors qu'un bloc
     q1_0 couvre 4 k-tiles → bits des tuiles précédentes effacés.
   - **fix8** (décisif) : garde de l'inverse testait `kt < ne0/QK1_0` — mais
     `kt` est une tuile de 32, pas un bloc de 128. Garde correcte :
     `qb = kt/4 < ne0/QK1_0`. → **32/32**.

Leçon : dans `test-backend-ops`, l'inverse de repack (`get_tensor`) est une
surface de test de première classe.

## 4. Validation

```
test-backend-ops -b HTP0 -o MUL_MAT -p 'type_a=q1_0.*k=(128|256|512)'
  → 32/32 tests passed, 2/2 backends passed
test-backend-ops -b HTP0 -o MUL_MAT
  → 796/796 tests passed (aucune régression ; q4_0 3/3, q8_0 3/3)
```

Note de rigueur : les filtres `type_a=q4_k.*` / `type_a=q6_k.*` n'ont
**retourné aucun cas** (0/0) dans ce build — ni validés ni régressés.

Test unitaire host (hors device, `q10_repack_unit.cpp`) :
- round-trip upstream `repack_q4_0_tiled` → `repack_tiled_q4_0` = identité ;
- `repack_q1_0(Q1_0)` == `repack_q4_0(matrice q4_0 équivalente)`
  **octet à octet** (K=256, M=33) ;
- flow officiel set_tensor → get_tensor(fix7/8) → déquant : exact
  (K=128/256, M=16/5).

## 5. Bench device (Nanbeige4.2-3B Q1_0, 4,17B params, 890,56 MiB)

Device : OnePlus 15 (CPH2747), HTP v81, stack `upq10`, build-full 37f752b+Stage1.

| Config | Backend | PP512 (t/s) | TG128 (t/s) |
|---|---|---:|---:|
| Q1_0, -ngl 99 | HTP0 | **923,97 ± 18,45** | **15,42 ± 0,02** |
| Q1_0, -ngl 0 (témoin) | CPU | 8,66 ± 0,34 | 6,05 ± 0,27 |

Accélération HTP vs CPU : ~107× prompt, ~2,5× génération. Stabilité TG :
±0,13 %.

## 6. Reproduction

```bash
# build (WSL, NDK r29)
cd /home/videl/up-b11855
cmake --build build-full --target test-backend-ops -j12

# deploy device
adb push build-full/bin/libggml-hexagon.so build-full/bin/test-backend-ops \
  /data/local/tmp/upq10/
adb shell "cd /data/local/tmp/upq10 && chmod 755 test-backend-ops"

# tests
adb shell "cd /data/local/tmp/upq10 && LD_LIBRARY_PATH=. ./test-backend-ops \
  -b HTP0 -o MUL_MAT -p 'type_a=q1_0.*k=(128|256|512)'"

# bench
adb shell "cd /data/local/tmp/upq10 && LD_LIBRARY_PATH=. ./llama-bench \
  -m /data/local/tmp/Nanbeige4.2-3B-Q1_0.gguf -dev HTP0 -ngl 99"
```

Scripts patch réexécutables : `patch_q10_stage1*.py`,
`patch_q10_stage1_fix{6,7,8}*.py` (E:\oneplus\kernel_analysis\). Sonde de
debug HTP-vs-CPU : `q10_probe_v3.cpp`. Test unitaire host :
`q10_repack_unit.cpp` (g++ -O1 -std=c++17, sans device).

Rapport complet : `D:\RAPPORT_P03_UPSTREAM_B11855_MTP_2026-09-28.md`
(§20-22, miroir E:).

## 7. Limites et suite (Stage 2)

- Stage 1 gonfle les poids ×4 en buffer (1,125 → 4,5 bpw) : Nanbeige 3B tient
  (~3,3 GiB repacké) ; Bonsai 27B (~14 GiB) ne tient pas.
- **Stage 2 requis** : kernel HVX natif 1-bit (Σa vectorielle
  `d*(2*acc−sumy)` + layout flat, cf. PLAN_KERNEL §8 bis et production
  Qualcomm `mul_mv_q1_0_f32_flat.cl`).

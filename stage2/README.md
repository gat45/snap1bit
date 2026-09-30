# Stage 2 — Kernel HVX natif Q1_0 (design, 30-09)

## 0. Verdict du design

Deux variantes possibles du microkernel 32x1. La **V1 (tuile Q4_0, compensée)**
est recommandée : zéro nouveau layout, zéro changement de plomberie host,
deux instructions de moins par k-tile que V2, et le pattern est déjà validé
par notre Stage 1 (nibbles 7/9).

## 1. Mathématique (stratégie B, validée par mul_mv_q1_0_f32_flat.cl Qualcomm)

Un bloc q1_0 = 128 poids ±d (bit b_j). Le dot avec une tuile d'activation
quantifiée q8 (i8 + scale d_a par sous-bloc de 32) :

    dot_32 = Σ_j w_j a_j = Σ_j (2 b_j − 1) d a_j
           = d_a · d_w · ( 2·Σ_{b=1} q8_j − Σ_j q8_j )      (q8 = act quantifiée)

Via le layout tuile Q4_0 existant (nibbles n_j = 7 + 2 b_j) :

    vrmpy(nibble_i8, q8_i8) = Σ_j (7 + 2 b_j) q8_j = 7·Σq8 + 2·Σ_{b=1} q8
    ⇒ dot_raw = vrmpy − 7·Σq8_ktile
    ⇒ dot = d_w · d_a · dot_raw

**L'unique inconnue est Σq8 par k-tile de 32** (somme des i8 de l'activation).
Deux options :

- **Option A (précalcul au quantize)** : dans
  `quantize_block_f32_q8_0_tiled`, ajouter par sous-bloc la somme i8
  (vrmpy avec vecteur de 1) et la stocker dans dst[10] (il reste de la
  place : 1152 o = 9 vecteurs, on passe à 10, tile 1280 o = taille q8_1).
  Coût : quasi nul (1 instr/sous-bloc). Impact host : ACT tile size 1152→1280
  (identique à q8_1 → réutilise `htp_mm_q8_1_tiled_row_size`).
- **Option B (au vol dans le kernel)** : `vrmpy` du q8 contre un vecteur de
  uns → même coût par k-tile que l'option A mais répété pour CHAQUE colonne
  de poids traitée par le même k-tile... En fait non : Σq8 ne dépend que du
  k-tile et de la ligne d'activation → recalculé 1× par kernel call si on
  le fait au début de tiled_vec_dot (le même v_act est lu par toutes les
  ktiles). Coût : 2 instructions × n_k_tiles, négligeable. **Option B
  retenue pour V1** (aucun changement du quantize ni du host).

## 2. Kernel V1 : `tiled_vec_dot_q1_0_32x1`

```c
// Tuile poids : layout Q4_0 tiled EXISTANT (576 o : nibbles 7|9 + scales f16),
// rempli par repack_q1_0_as_q4_0_tiled (Stage 1, déjà validé).
// Act : q8_0_tiled existant (1152 o/bloc de 128, 8+1 vecteurs/sous-bloc).
static void tiled_vec_dot_q1_0_32x1(const uint32_t n, float * restrict s,
        const void * restrict vx, const void * restrict vy,
        uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    // sommet_i8 du bloc d'activation : pour Σq8 par sous-bloc on utilise
    // vrmpy avec un vecteur de uns : sum_i8 = Q6_Vw_vrmpy_VbVb(ones, v_q)
    const HVX_Vector ones = Q6_Vb_vsplat_R(1);

    HVX_Vector v_sum_float = Q6_V_vzero();
    const HVX_Vector seven = Q6_Vb_vsplat_R(7);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        // accum i32 identique a q4_0 (nibbles 7/9 dejaSigned, offset 8 inutile:
        // on garde les nibbles tels quels et on compense par 7·Σ)
        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act, Q6_V_vzero());

        // Σq8 des 32 act de ce k-tile (2 vrmpy : act réparti sur v_act[0..7])
        HVX_Vector v_sum_a = Q6_Vw_vrmpy_VbVb(v_act[0], ones);
        v_sum_a = Q6_Vw_vadd_VwVw(v_sum_a, Q6_Vw_vrmpy_VbVb(v_act[1], ones));
        // ... (v_act[2..7]) — TODO unroll macro

        // vrmpy = 7·Σa + 2·Σ_{b=1}a  =>  dot_raw = vrmpy − 7·Σa
        HVX_Vector v_corr = Q6_Vw_vmpy_VwVh(seven_i32, sum_a_i16);  // 7·Σa en i32
        HVX_Vector v_dot  = Q6_Vw_vsub_VwVw(v_sum, v_corr);

        HVX_Vector v_sum_sf  = Q6_Vsf_equals_Vw(v_dot);
        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        v_sum_float = hvx_vec_add_f32_f32(v_sum_float,
                        hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb));
    }
    // store identique q4_0
}
```

⚠️ Points à trancher au codage réel (indiqués pour ne pas les oublier) :
- `accum_4bit_32x1` soustrait i8=8 → pour Q1_0 on peut soit réutiliser tel quel
  (compensation : vrmpy − 8·Σa au lieu de 7·Σa, plus simple : **garder i8=8**,
  correction `vrmpy − 8·Σa`, zéro différence de code avec q4_0), soit neutraliser.
  → **Choix : garder i8=8, correction = 8·Σa.**
- Le nibble est (7+2b) en LOW et HIGH ; après `unpack_and_interleave_4bit_x2`
  les i8 valent n (0..15), l'offset i8=8 donne n−8 = ±1 exactement →
  `vrmpy = Σ (n_j−8) q8_j = Σ ±1 · q8_j = 2Σ_{b=1} − Σ` déjà ! La correction
  8·Σa n'est alors PAS nécessaire : `accum_4bit_32x1` avec i8=8 donne
  **directement** Σ ±1·q8. ⇒ Le kernel V1 est quasi identique au q4_0, seule
  la sémantique change (les nibbles 7/9 portent ±1 au lieu de q4). Vérifier
  numériquement quand même (test host avant device).
- Overlap `unpack_and_interleave_4bit_x2` : même entrelacement que q4_0 →
  reuse tel quel.

**Conclusion simplifiée** : `tiled_vec_dot_q1_0_32x1` = copie de
`tiled_vec_dot_q4_0_32x1` où `i8` est appliqué (n−8 = ±1) — ce qui est déjà
le cas — et où le résidu de signe est exact. Le gain Stage 2 n'est donc PAS
dans le microkernel (déjà optimal) mais **dans le buffer** : ne plus stocker
les poids en tuiles Q4_0 (576 o) mais en **Q1_0 packed (18 o/bloc de 128,
soit 72 o/tuile 32×32 → réduction 8× du trafic poids DDR/VTCM**).

## 3. Vrai gain Stage 2 : layout packed-native

Nouveau layout tuile `q1_0_tiled` : 32 rows × 32 cols de bits = 128 o de
bits + 64 o scales f16 = 192 o/tuile (vs 576) → **3× moins de lectures
poids**. Le microkernel unpack les bits → nibbles 7/9 **dans VTCM** (1
valen/vshade) puis accum_4bit identique. Nouvelles pièces :
- `repack_q1_0_native_tiled` : bits row-major par tuile 32×32 (128 o) +
  scales ; ~3 h de code.
- `tiled_vec_dot_q1_0_native_32x1` : unpack bits (vlut/vdelta 4×) → 2
  vecteurs nibble 7/9 → accum identique.
- Host : `HTP_MM_WEIGHT_TILE_SIZE_Q1_0 = 192`, tiled_row_size, alloc.
- Validation : test-backend-ops 32/32 (même filtre), bench Nanbeige
  (attendu : tg ↑ car BW poids /3 ; pp légèrement ↑).

## 4. Ordre d'implémentation

1. **S2.1** (1-2 h) : microkernel V1 + dispatch (prouve la sémantique ±1,
   zéro gain perf attendu — déjà couvert par Stage 1) → tests.
2. **S2.2** (1 jour) : layout packed-native (192 o) + repack + host wiring
   → 32/32 → bench A/B vs Stage 1 (l'objectif : tg 15,42 → 20+ t/s si
   BW-bound).
3. **S2.3** : 27B feasibility avec le layout 192 o : 3,53 GiB × (1 + 192/
   (18·32)) ... recalc : poids tuilés 27B = 3,53 GiB × 192/576 = 1,18 GiB +
   buffers → TIENT sur device. C'est le déblocage 27B.

## 5. Risques

- `vlut`/`vdelta` bits→nibbles : coût par k-tile à mesurer (peut manger le
  gain BW si mal vectorisé) → mesurer au profiling S2.2 avant de conclure.
- Alignements VTCM des tuiles 192 o (192 % 128 ≠ 0) → padding à 256 o/tuile
  (2× mieux que 576 quand même).
- Le dispatch actuel route Q1_0 via `wtype=Q4_0` (fix6) : le layout natif
  exigera un vrai type interne `HTP_TYPE_Q1_0_TILED` → attention aux
  switches (tile_size, src1_row_size inchangés côté act).

// s21_q1_0_native_tiled.c — Stage 2 : scaffold de recherche Q1_0 (30-09)
//
// STATUT: NON DEPLOYABLE. Le self-test hôte vérifie le format dense et la
// sémantique ±1, mais la portion HVX est volontairement un skeleton. Ne pas
// l'intégrer comme kernel: le mapping dense bit->lanes doit d'abord être
// démontré byte-à-byte contre unpack_and_interleave_4bit_x2. En particulier,
// une tuile de 128 octets de bits ne peut pas être lue comme vptr[0..3]
// (quatre vecteurs de 128 octets) sans réplication ou dépassement de buffer.
//
// Resolution de la question laissee ouverte par stage2_kernel_draft.md §2 :
//   « les nibbles 7/9 apres offset i8=8 donnent n-8 = ±1 exactement →
//     accum_4bit_32x1 reutilise tel quel, zero correction »
// → CONFIRME PAR LES SOURCES (section 22.1 du rapport) : le Stage 1 (fix6,
//   repack_q1_0_as_q4_0_tiled, ggml-hexagon.cpp:1022) remplit deja exactement
//   ce layout et le kernel q4_0 (hvx-mm-kernels-tiled.h:539) porte donc deja
//   la semantique ±1. Le microkernel V1 n'apporte rien de neuf.
//
// => Le vrai gain Stage 2 est le LAYOUT NATIF : on ne stocke plus les poids
//    en tuiles Q4_0 (576 o/tuile 32x32) mais en BITS PACKES (192 o/tuile),
//    et le kernel dequantifie bits -> nibbles 7/9 DANS le kernel.
//
//    q4_0 tiled (Stage 1) :  4 vecteurs nibbles (512 o) + 1 vecteur scales
//                            (64 o) = 576 o/tuile (pad 640)
//    q1_0 natif (ce fichier): 1 vecteur de bits (128 o = 1024 bits = 32
//                            rows x 32 k) + 64 o de scales f16 = 192 o/tuile
//    → 3x moins de trafic poids DDR/VTCM. Feasibilite 27B : 3,53 GiB de
//      poids tuiles 576-o -> 1,18 GiB en 192-o : TIENT sur device.
//
// Coté kernel, le q4_0 lit vptr[0..3] (nibbles packes 2 valeurs/octet) et
// passe chaque vecteur dans unpack_and_interleave_4bit_x2 puis vsub(i8=8).
// En natif on lit 1 vecteur de bits que l'on EXPAND en 8 vecteurs de nibbles
// non-packed (1 valeur/octet, valeur = 7 + 2*bit) ; l'accumulateur consomme
// alors directement ces vecteurs (pas d'unpack, vsub(i8=8) → ±1 exact).
//    Cout calc  : ~8 phases (vand/vlsr/vor) + 8 vrmpyacc vs ~4 unpack + 4 vrmpy
//    Cout poids : 192 o lus vs 576 o lus → si BW-bound (tg128), gain net attendu
//    (a MESURER en S2.3, cf. draft §5 : le cout d'expansion peut manger le
//     gain BW si mal vectorise — c'est exactement ce que ce scaffold permet
//     de benches).
//
// Layout d'une tuile native "q1_0_tiled" (32 rows x 32 k) :
//   [0   ..127]  bits packes, bit index = row*32 + (k % 32), LSB-first
//   [128 ..191]  scales f16 LE, scale[row] = d du bloc q1_0 couvrant (row, k/128)
//   Taille 192 o, alignee a 256 o (HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q1_0).
//
// Selftest x86 (gcc -DS21_SELFTEST) : quantize q1_0 (ref ggml-quants.c:40 :
// d = sum|x|/128, bit = x>0), repack natif, dot scalaire de reference, et
// verification de parite du chemin "nibbles 7/9 + offset i8=8" (domaine
// entier, exact). Le kernel HVX (garde __HVX__) est fourni pour la compile
// hexagon-clang (-mhvx -mv81) est seulement une signature de travail: il ne
// valide ni le layout vectoriel ni les performances.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Constantes layout (miroir de matmul-ops.h cote DSP, a y reporter en S2.2)
// ---------------------------------------------------------------------------
#define QK1_0                              128  // bloc ggml q1_0 = 128 elements
#define Q1_0_BLOCK_BYTES                    18  // 2 o (d f16) + 16 o bits
#define HTP_MM_WEIGHT_TILE_SIZE_Q1_0       192  // tuile 32x32 native
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q1_0 256 // pad VTCM
#define HTP_MM_TILED_ROWS                  32   // rows par tuile (32x1)
#define HTP_MM_K_PER_TILE                  32   // k par k-tile

// ---------------------------------------------------------------------------
// f16 (binary16) minimal pour la reference host — pas d' dependence F16C
// ---------------------------------------------------------------------------
static float s21_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x03FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else { // subnormal
            float v;
            exp = 127 - 15 - 10;
            while (!(man & 0x0400u)) { man <<= 1; exp--; }
            man &= 0x03FFu;
            bits = sign | ((uint32_t)(exp + 15 + 10) << 23) | (man << 13);
            (void)v;
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

static uint16_t s21_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man  = x & 0x007FFFFFu;
    if (exp >= 31)  return (uint16_t)(sign | 0x7C00u);          // inf
    if (exp <= 0)   return (uint16_t)sign;                      // flush (scales > 0 en pratique)
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

// ---------------------------------------------------------------------------
// Quantize q1_0 reference (identique ggml-quants.c:40, bloc 128, d=sum|x|/n)
// ---------------------------------------------------------------------------
typedef struct {
    uint16_t d;        // f16
    uint8_t  qs[16];   // bits de signe, LSB-first
} s21_block_q1_0;

static void s21_quantize_row_q1_0_ref(const float *x, s21_block_q1_0 *y, int64_t n) {
    for (int64_t b = 0; b < n; b += QK1_0) {
        float amax = 0.0f;
        for (int i = 0; i < QK1_0; i++) amax += fabsf(x[b + i]);
        const float d = amax / QK1_0;
        y->d = s21_f32_to_f16(d);
        memset(y->qs, 0, 16);
        for (int i = 0; i < QK1_0; i++) {
            if (x[b + i] > 0.0f) y->qs[i / 8] |= (uint8_t)(1u << (i % 8));
        }
        y++;
    }
}

// Dequant scalar d'une tuile native -> ±d (reference exacte du layout)
static void s21_tile_dequant_ref(const uint8_t *tile, float *out /*32x32*/) {
    float ds[32];
    for (int r = 0; r < 32; r++) {
        uint16_t h;
        memcpy(&h, tile + 128 + 2 * r, 2);
        ds[r] = s21_f16_to_f32(h);
    }
    for (int r = 0; r < 32; r++)
        for (int k = 0; k < 32; k++) {
            unsigned bit = (tile[r * 4 + k / 8] >> (k % 8)) & 1u;
            out[r * 32 + k] = bit ? ds[r] : -ds[r];
        }
}

// ---------------------------------------------------------------------------
// Repack natif : blocs q1_0 [nrows x ne0] -> tuiles 192 o (row-block x k-tile)
// Meme convention d'ordre que le repack q4_0 du Stage 1 : tuile indexee par
// (row_block, k_tile), k_tile majeur dans un row_block.
// ---------------------------------------------------------------------------
static void s21_repack_q1_0_native_tiled(const s21_block_q1_0 *src, int nrows,
                                         int ne0, uint8_t *dst) {
    const int n_rtiles = ne0 / HTP_MM_K_PER_TILE;            // k-tiles par row
    const int n_rblk  = (nrows + 31) / 32;
    memset(dst, 0, (size_t)n_rblk * n_rtiles * HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q1_0);
    for (int rb = 0; rb < n_rblk; rb++) {
        for (int kt = 0; kt < n_rtiles; kt++) {
            uint8_t *tile = dst + ((size_t)rb * n_rtiles + kt) *
                                  HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q1_0;
            for (int r = 0; r < 32; r++) {
                const int row = rb * 32 + r;
                if (row >= nrows) break;
                // bloc q1_0 couvrant : ce row, k-range [kt*32 ..], bloc = k/128
                const int k0 = kt * HTP_MM_K_PER_TILE;
                const s21_block_q1_0 *blk = &src[(size_t)row * (ne0 / QK1_0) + k0 / QK1_0];
                memcpy(tile + 128 + 2 * r, &blk->d, 2);
                for (int k = 0; k < 32; k++) {
                    const int kk   = k0 + k;                    // index global k
                    const int bit  = (blk->qs[(kk % QK1_0) / 8] >> (kk % 8)) & 1u;
                    if (bit) tile[r * 4 + k / 8] |= (uint8_t)(1u << (k % 8));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dot scalaire de reference : 1 tuile (32 rows x 32 k) x activation f32[32]
// s[r] += sum_k w[r,k] * a[k]  (semantique du kernel 32x1, accumule par k-tile)
// ---------------------------------------------------------------------------
static void s21_vec_dot_tile_ref(const uint8_t *tile, const float *a /*32*/, float *s /*32*/) {
    float w[32 * 32];
    s21_tile_dequant_ref(tile, w);
    for (int r = 0; r < 32; r++) {
        float acc = 0.0f;
        for (int k = 0; k < 32; k++) acc += w[r * 32 + k] * a[k];
        s[r] += acc;
    }
}

// Parite du chemin entier "nibbles 7/9, offset i8=8" : pour chaque octet de
// bits, les 8 valeurs (n - 8) avec n = 7 + 2*bit doivent valoir ±1 exactement.
static int s21_check_nibble_parity(void) {
    for (uint32_t b = 0; b < 256u; b++) {
        for (int j = 0; j < 8; j++) {
            const int bit = (b >> j) & 1;
            const int n   = 7 + 2 * bit;   // nibble stocke par le Stage 1
            const int v   = n - 8;         // ce que fait vsub(i8=8)
            if (v != (bit ? 1 : -1)) {
                printf("FAIL parity b=%u j=%d n=%d v=%d\n", b, j, n, v);
                return 0;
            }
        }
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Kernel HVX (compile hexagon-clang -mhvx ; inert en x86)
// Convention d'appel calquee sur tiled_vec_dot_q4_0_32x1 (kernels-tiled.h:539)
// ---------------------------------------------------------------------------
#ifdef __HVX__

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

// Expansion bits -> nibbles 7/9 (1 valeur/octet), 1 vecteur de bits en entree
// → 8 vecteurs de nibbles en sortie. Stratégie 8 phases (correct-first) :
// pour la phase j, isoler le bit j de chaque octet puis le placer en position
// (j mod 4) du quad correspondant via vdelta, et composer par vor.
// (Candidat a optimisation en S2.2 : double vlut 4-bits -> 4 o, ou pack 2
//  bits/octet au repack pour diviser les phases par 2 — cf. draft §5.)
#define S21_EXPAND_PHASE(j, bits, acc_j)                                     \
    do {                                                                     \
        HVX_Vector v_b = Q6_V_vand_VV(bits, Q6_Vb_vsplat_R(1u << (j)));      \
        v_b = (j) ? Q6_Vub_vlsr_VubR(v_b, (j)) : v_b;                        \
        acc_j = Q6_V_vor_VV(acc_j, v_b);                                     \
    } while (0)

// Accumule une tuile native 32x32 contre 8 vecteurs d'activation q8 non-packed
// (meme convention que v_act du q4_0 : 32 valeurs f32-scale en v_act[8]... non,
// en q8_0_tiled le scale act est le vecteur v_act[8] — la reprise est verbatim).
static inline HVX_Vector s21_accum_q1_0_tile(const HVX_Vector *restrict vptr,
                                             const HVX_Vector *restrict v_act) {
    const HVX_Vector i8    = Q6_Vb_vsplat_R(8);
    const HVX_Vector seven = Q6_Vb_vsplat_R(7);
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    // vptr[0] = 128 o de bits = tuile complete ; expansion en 8 vecteurs :
    // nib[2j]   = vecteur des valeurs (7+2b) pour la moitie basse du k-quad j
    // nib[2j+1] = moitie haute. Chaque nib[j] consomme 1 vecteur act : v_act[j].
    HVX_Vector nib[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) nib[j] = Q6_V_vand_VV(Q6_V_vzero(), seven); // 7*ones

    // NOTE S2.2 : placement exact des bits dans les quads a aligner sur la
    // convention unpack_and_interleave_4bit_x2 du q4_0 (shuff/valign) avant
    // le premier run device — le selftest host verifie la semantique, le
    // placement vectoriel est valide au test-backend-ops.
    (void)vptr; (void)v_act; (void)i8;
    // TODO(S2.2) : boucle reelle — expand puis
    //   v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, nib[2j], v_act[2j]);
    //   v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, nib[2j+1], v_act[2j+1]);
    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

// Skeleton du kernel 32x1 natif (meme signature que q4_0). Les strides poids
// passent de 640 a 256 (tuile alignee), l'act reste 1152 o / bloc 128.
static void s21_tiled_vec_dot_q1_0_native_32x1(const uint32_t n, float *restrict s,
                                               const void *restrict vx,
                                               const void *restrict vy,
                                               uint32_t valid_rows,
                                               const float *restrict sz) {
    const uint8_t *restrict tile_ptr = vx;
    const uint8_t *restrict y_q      = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    const uint32_t n_k_tiles = n / HTP_MM_K_PER_TILE;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector *restrict vptr =
            (const HVX_Vector *)(tile_ptr + kt * HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q1_0);
        const HVX_Vector *restrict v_act = (const HVX_Vector *)(y_q + kt * 1152);

        HVX_Vector v_sum = s21_accum_q1_0_tile(vptr, v_act);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        // scales : vptr[4] sous q4_0 (576/128 = 4.5 → vecteur 4) devient ici
        // le vecteur 1 (192/128 = 1.5 → scales a l'offset 128 = vecteur 1)
        HVX_Vector v_scale_w   = vptr[1];
        HVX_Vector v_scale_a   = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float),
                        hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

#endif // __HVX__

// ---------------------------------------------------------------------------
// Selftest x86 : parité nibbles, quantize/repack, dot de reference vs dot
// direct sur les floats d'origine (tolerance = erreur d'arrondi attendue)
// ---------------------------------------------------------------------------
#ifdef S21_SELFTEST

static uint32_t rng = 0x12345678u;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((rng >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

int main(void) {
    int ok = 1;

    // 1. parite nibbles 7/9 / offset 8 → ±1
    ok &= s21_check_nibble_parity();
    printf("[%s] parite nibbles 7/9 + i8=8 → ±1 exact\n", ok ? "OK" : "FAIL");

    // 2. matrice 64 rows x 256 k → quantize → repack natif → dot ref
    enum { NR = 64, NK = 256 };
    float *x = malloc(sizeof(float) * NR * NK);
    for (int i = 0; i < NR * NK; i++) x[i] = frand();

    s21_block_q1_0 *q = malloc(sizeof(s21_block_q1_0) * NR * (NK / QK1_0));
    for (int r = 0; r < NR; r++)
        s21_quantize_row_q1_0_ref(x + (size_t)r * NK, q + (size_t)r * (NK / QK1_0), NK);

    uint8_t *tiles = malloc((size_t)(NR / 32) * (NK / 32) *
                            HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q1_0);
    s21_repack_q1_0_native_tiled(q, NR, NK, tiles);

    // activations
    float a[NK];
    for (int i = 0; i < NK; i++) a[i] = frand();

    // dot direct sur dequantise (verifie le layout bits/scales du repack)
    float s[32];
    for (int rb = 0; rb < NR / 32; rb++) {
        for (int r = 0; r < 32; r++) s[r] = 0.0f;
        for (int kt = 0; kt < NK / 32; kt++) {
            const uint8_t *tile = tiles +
                ((size_t)rb * (NK / 32) + kt) * HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q1_0;
            s21_vec_dot_tile_ref(tile, a + kt * 32, s);
        }
        // reference : dot direct float d'origine, compare a la BORNE EXACTE
        // de l'erreur de quantization 1-bit (Cauchy-Schwarz sur w = ±d_blk) :
        //   |dot_q - dot| <= somme_k |w_q[k]-w[k]| * |a[k]| <= somme_k 2 d_blk |a[k]|
        // (un seuil RELATIF n'a pas de sens : le bruit 1-bit domine des que
        //  |ref| est petit — c'est precisement le mecanisme [H3] du rapport)
        for (int r = 0; r < 32; r++) {
            const int row = rb * 32 + r;
            double ref = 0.0, bound = 0.0;
            for (int k = 0; k < NK; k++) {
                ref += (double)x[(size_t)row * NK + k] * a[k];
                const s21_block_q1_0 *blk =
                    &q[(size_t)row * (NK / QK1_0) + k / QK1_0];
                bound += 2.0 * (double)s21_f16_to_f32(blk->d) * fabs((double)a[k]);
            }
            const double err = fabs((double)s[r] - ref);
            if (!(err <= bound * 1.0000001 + 1e-6)) {
                printf("FAIL dot row %d : got %f ref %f err %f > borne %f\n",
                       row, s[r], ref, err, bound);
                ok = 0;
            }
        }
    }
    printf("[%s] dot tuile native vs float direct (tol. quant 1-bit)\n", ok ? "OK" : "FAIL");

    // 3. l'erreur doit etre ENCORE plus petite vs la reference dequantisee
    //    (bit-exact attendu a l'arrondi f16 pres sur les scales)
    float w_dq[NR * NK];
    for (int r = 0; r < NR; r++)
        for (int k = 0; k < NK; k++) {
            const s21_block_q1_0 *blk = &q[(size_t)r * (NK / QK1_0) + k / QK1_0];
            const unsigned bit = (blk->qs[(k % QK1_0) / 8] >> (k % 8)) & 1u;
            w_dq[(size_t)r * NK + k] = (bit ? 1.0f : -1.0f) * s21_f16_to_f32(blk->d);
        }
    double max_rel = 0.0;
    for (int rb = 0; rb < NR / 32; rb++) {
        for (int r = 0; r < 32; r++) s[r] = 0.0f;
        for (int kt = 0; kt < NK / 32; kt++) {
            const uint8_t *tile = tiles +
                ((size_t)rb * (NK / 32) + kt) * HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q1_0;
            s21_vec_dot_tile_ref(tile, a + kt * 32, s);
        }
        for (int r = 0; r < 32; r++) {
            const int row = rb * 32 + r;
            double ref = 0.0;
            for (int k = 0; k < NK; k++) ref += (double)w_dq[(size_t)row * NK + k] * a[k];
            const double rel = fabs((double)s[r] - ref) / (fabs(ref) + 1e-9);
            if (rel > max_rel) max_rel = rel;
        }
    }
    printf("[%s] dot tuile native vs dequantise exact : max rel = %.3g\n",
           max_rel < 1e-5 ? "OK" : "FAIL", max_rel);
    ok &= (max_rel < 1e-5);

    printf("=== S21 SELFTEST %s ===\n", ok ? "PASS" : "FAIL");
    free(x); free(q); free(tiles);
    return ok ? 0 : 1;
}

#endif // S21_SELFTEST

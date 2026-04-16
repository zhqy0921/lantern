#include "lantern_poly.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

/*
 * This file is intentionally self-contained for learning:
 * 1) scalar reference implementation
 * 2) optional AVX512 kernels
 * 3) runtime dispatch that picks the best path
 */

static inline uint32_t mod_q_u64(uint64_t x) {
  return (uint32_t)(x % LANTERN_Q);
}

/* ----------------------------- scalar backend ----------------------------- */

static void lantern_poly_zero_scalar(lantern_poly *p) {
  size_t i;
  for (i = 0; i < LANTERN_DEG; i++) {
    p->c[i] = 0;
  }
}

static void lantern_poly_add_scalar(lantern_poly *r, const lantern_poly *a, const lantern_poly *b) {
  size_t i;
  for (i = 0; i < LANTERN_DEG; i++) {
    uint64_t s = (uint64_t)a->c[i] + (uint64_t)b->c[i];
    r->c[i] = mod_q_u64(s);
  }
}

static void lantern_poly_sub_scalar(lantern_poly *r, const lantern_poly *a, const lantern_poly *b) {
  size_t i;
  for (i = 0; i < LANTERN_DEG; i++) {
    uint64_t ai = a->c[i];
    uint64_t bi = b->c[i];
    uint64_t d = (ai >= bi) ? (ai - bi) : (ai + LANTERN_Q - bi);
    r->c[i] = (uint32_t)d;
  }
}

static void lantern_poly_mul_scalar(lantern_poly *r, const lantern_poly *a, const lantern_poly *b) {
  __int128_t acc[LANTERN_DEG];
  size_t i;
  size_t j;

  for (i = 0; i < LANTERN_DEG; i++) {
    acc[i] = 0;
  }

  /* Negacyclic convolution in Z_q[X]/(X^N + 1). */
  for (i = 0; i < LANTERN_DEG; i++) {
    for (j = 0; j < LANTERN_DEG; j++) {
      __int128_t term = ((__int128_t)a->c[i]) * ((__int128_t)b->c[j]);
      size_t k = i + j;
      if (k < LANTERN_DEG) {
        acc[k] += term;
      } else {
        acc[k - LANTERN_DEG] -= term;
      }
    }
  }

  for (i = 0; i < LANTERN_DEG; i++) {
    __int128_t m = acc[i] % (__int128_t)LANTERN_Q;
    if (m < 0) {
      m += (__int128_t)LANTERN_Q;
    }
    r->c[i] = (uint32_t)m;
  }
}

/* ---------------------------- runtime feature ---------------------------- */

#if defined(__x86_64__) || defined(__i386__)
static int lantern_cpu_has_avx512(void) {
  static int cached = -1;
  if (cached >= 0) {
    return cached;
  }

#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  cached = (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
            __builtin_cpu_supports("avx512vl"))
               ? 1
               : 0;
#else
  cached = 0;
#endif

  return cached;
}
#else
static int lantern_cpu_has_avx512(void) {
  return 0;
}
#endif

/* ----------------------------- AVX512 backend ----------------------------- */

#if defined(__x86_64__) || defined(__i386__)
#define LANTERN_AVX512_ATTR __attribute__((target("avx512f,avx512dq,avx512vl")))

LANTERN_AVX512_ATTR
static void lantern_poly_add_avx512(lantern_poly *r, const lantern_poly *a, const lantern_poly *b) {
  const __m512i q = _mm512_set1_epi64((long long)LANTERN_Q);
  size_t i = 0;

  for (; i + 8 <= LANTERN_DEG; i += 8) {
    __m256i a32 = _mm256_loadu_si256((const __m256i *)(a->c + i));
    __m256i b32 = _mm256_loadu_si256((const __m256i *)(b->c + i));

    __m512i a64 = _mm512_cvtepu32_epi64(a32);
    __m512i b64 = _mm512_cvtepu32_epi64(b32);
    __m512i sum = _mm512_add_epi64(a64, b64);
    __m512i sub = _mm512_sub_epi64(sum, q);

    __mmask8 geq_q = _mm512_cmpge_epu64_mask(sum, q);
    __m512i red = _mm512_mask_mov_epi64(sum, geq_q, sub);

    __m256i out = _mm512_cvtepi64_epi32(red);
    _mm256_storeu_si256((__m256i *)(r->c + i), out);
  }

  for (; i < LANTERN_DEG; i++) {
    uint64_t s = (uint64_t)a->c[i] + (uint64_t)b->c[i];
    r->c[i] = mod_q_u64(s);
  }
}

LANTERN_AVX512_ATTR
static void lantern_poly_sub_avx512(lantern_poly *r, const lantern_poly *a, const lantern_poly *b) {
  const __m512i q = _mm512_set1_epi64((long long)LANTERN_Q);
  size_t i = 0;

  for (; i + 8 <= LANTERN_DEG; i += 8) {
    __m256i a32 = _mm256_loadu_si256((const __m256i *)(a->c + i));
    __m256i b32 = _mm256_loadu_si256((const __m256i *)(b->c + i));

    __m512i a64 = _mm512_cvtepu32_epi64(a32);
    __m512i b64 = _mm512_cvtepu32_epi64(b32);

    __m512i no_borrow = _mm512_sub_epi64(a64, b64);
    __m512i with_q = _mm512_sub_epi64(_mm512_add_epi64(a64, q), b64);

    __mmask8 geq = _mm512_cmpge_epu64_mask(a64, b64);
    __m512i red = _mm512_mask_mov_epi64(with_q, geq, no_borrow);

    __m256i out = _mm512_cvtepi64_epi32(red);
    _mm256_storeu_si256((__m256i *)(r->c + i), out);
  }

  for (; i < LANTERN_DEG; i++) {
    uint64_t ai = a->c[i];
    uint64_t bi = b->c[i];
    uint64_t d = (ai >= bi) ? (ai - bi) : (ai + LANTERN_Q - bi);
    r->c[i] = (uint32_t)d;
  }
}

#else
static void lantern_poly_add_avx512(lantern_poly *r, const lantern_poly *a, const lantern_poly *b) {
  lantern_poly_add_scalar(r, a, b);
}

static void lantern_poly_sub_avx512(lantern_poly *r, const lantern_poly *a, const lantern_poly *b) {
  lantern_poly_sub_scalar(r, a, b);
}
#endif

/* ------------------------------- public API ------------------------------- */

void lantern_poly_zero(lantern_poly *p) {
  lantern_poly_zero_scalar(p);
}

void lantern_poly_add(lantern_poly *r, const lantern_poly *a, const lantern_poly *b) {
  if (lantern_cpu_has_avx512()) {
    lantern_poly_add_avx512(r, a, b);
    return;
  }
  lantern_poly_add_scalar(r, a, b);
}

void lantern_poly_sub(lantern_poly *r, const lantern_poly *a, const lantern_poly *b) {
  if (lantern_cpu_has_avx512()) {
    lantern_poly_sub_avx512(r, a, b);
    return;
  }
  lantern_poly_sub_scalar(r, a, b);
}

void lantern_poly_mul(lantern_poly *r, const lantern_poly *a, const lantern_poly *b) {
  /* Keep mul scalar for now: clearer and easier to validate while learning. */
  lantern_poly_mul_scalar(r, a, b);
}

const char *lantern_poly_backend_name(void) {
  return lantern_cpu_has_avx512() ? "avx512(add/sub)+scalar(mul)" : "scalar";
}

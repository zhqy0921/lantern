#include "../src/lantern_poly.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t rng_state = 1u;

static uint32_t next_u32(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state;
}

static void random_poly(lantern_poly *p) {
  size_t i;
  for (i = 0; i < LANTERN_DEG; i++) {
    p->c[i] = next_u32() % LANTERN_Q;
  }
}

static int equal_poly(const lantern_poly *a, const lantern_poly *b) {
  return memcmp(a->c, b->c, sizeof(a->c)) == 0;
}

int main(void) {
  lantern_poly a;
  lantern_poly b;
  lantern_poly c;
  lantern_poly t1;
  lantern_poly t2;
  lantern_poly t3;
  int i;

  printf("backend: %s\n", lantern_poly_backend_name());

  for (i = 0; i < 100; i++) {
    random_poly(&a);
    random_poly(&b);
    random_poly(&c);

    /* (a + b) - b == a */
    lantern_poly_add(&t1, &a, &b);
    lantern_poly_sub(&t2, &t1, &b);
    if (!equal_poly(&t2, &a)) {
      fprintf(stderr, "round-trip add/sub failed at test %d\n", i);
      return 1;
    }

    /* a + b == b + a */
    lantern_poly_add(&t1, &a, &b);
    lantern_poly_add(&t2, &b, &a);
    if (!equal_poly(&t1, &t2)) {
      fprintf(stderr, "add commutativity failed at test %d\n", i);
      return 1;
    }

    /* a*(b + c) == a*b + a*c */
    lantern_poly_add(&t1, &b, &c);
    lantern_poly_mul(&t1, &a, &t1);
    lantern_poly_mul(&t2, &a, &b);
    lantern_poly_mul(&t3, &a, &c);
    lantern_poly_add(&t2, &t2, &t3);
    if (!equal_poly(&t1, &t2)) {
      fprintf(stderr, "mul distributivity failed at test %d\n", i);
      return 1;
    }
  }

  puts("ok");
  return 0;
}

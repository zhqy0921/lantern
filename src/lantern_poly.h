#ifndef LANTERN_POLY_H
#define LANTERN_POLY_H

#include <stdint.h>

#define LANTERN_DEG 128u
#define LANTERN_Q 4294967197u

typedef struct {
  uint32_t c[LANTERN_DEG];
} lantern_poly;

void lantern_poly_zero(lantern_poly *p);
void lantern_poly_add(lantern_poly *r, const lantern_poly *a, const lantern_poly *b);
void lantern_poly_sub(lantern_poly *r, const lantern_poly *a, const lantern_poly *b);
void lantern_poly_mul(lantern_poly *r, const lantern_poly *a, const lantern_poly *b);
const char *lantern_poly_backend_name(void);

#endif

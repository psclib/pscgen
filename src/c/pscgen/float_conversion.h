#ifndef FLOAT_CONVERSION_H
#define FLOAT_CONVERSION_H

#include <stdio.h>
#include <inttypes.h>

typedef uint16_t half;

/* ----- prototypes ------ */
float half_to_float(half);
half float_to_half(float);

#endif /*FLOAT_CONVERSION_H*/

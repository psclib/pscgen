/**   IEEE 758-2008 Half-precision Floating Point Format
**    --------------------------------------------------
**    | Field    | Last | First | Note
**    |----------|------|-------|----------
**    | Sign     | 15   | 15    |
**    | Exponent | 14   | 10    | Bias = 15
**    | Fraction | 9    | 0     |
*/

#ifndef HALF_H
#define HALF_H

#include <stdio.h>
#include <inttypes.h>

typedef uint16_t half;

/* ----- prototypes ------ */
float half_to_float(half);
static uint32_t half_to_float_I(half);
half float_to_half(float);
static half float_to_half_I(uint32_t);

#endif /*HALF_H*/

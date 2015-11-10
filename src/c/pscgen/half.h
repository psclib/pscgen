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
half float_to_half(float);

#endif /*HALF_H*/

#include "half.h"

float half_to_float(half y)
{
    union { float f; uint32_t i; } v;
    v.i = half_to_float_I(y);
    return v.f;
}

uint32_t static half_to_float_I(half y)
{
    int s = (y >> 15) & 0x00000001;  // sign
    int e = (y >> 10) & 0x0000001f;  // exponent
    int f =  y        & 0x000003ff;  // fraction

    // need to handle 7c00 INF and fc00 -INF?
    if (e == 0) {
        // need to handle +-0 case f==0 or f=0x8000?
        if (f == 0)  // Plus or minus zero
            return s << 31;
        else {       // Denormalized number -- renormalize it
            while (!(f & 0x00000400)) {
                f <<= 1;
                e -=  1;
            }
            e += 1;
            f &= ~0x00000400;
        }
    } else if (e == 31) {
        // Inf
        if (f == 0)
            return (s << 31) | 0x7f800000;
        // NaN
        else
            return (s << 31) | 0x7f800000 | (f << 13);
    }

    e = e + (127 - 15);
    f = f << 13;

    return ((s << 31) | (e << 23) | f);
}

half float_to_half(float i)
{
    union { float f; uint32_t i; } v;
    v.f = i;
    return float_to_half_I(v.i);
}

half static float_to_half_I(uint32_t i)
{
    register int s =  (i >> 16) & 0x00008000;                   // sign
    register int e = ((i >> 23) & 0x000000ff) - (127 - 15);     // exponent
    register int f =   i        & 0x007fffff;                   // fraction

    // need to handle NaNs and Inf?
    if (e <= 0) {
        if (e < -10) {
            if (s)                                              // handle -0.0
               return 0x8000;
            else
               return 0;
        }
        f = (f | 0x00800000) >> (1 - e);
        return s | (f >> 13);
    } else if (e == 0xff - (127 - 15)) {
        if (f == 0)                                             // Inf
            return s | 0x7c00;
        else {                                                  // NAN
            f >>= 13;
            return s | 0x7c00 | f | (f == 0);
        }
    } else {
        if (e > 30)                                             // Overflow
            return s | 0x7c00;
        return s | (e << 10) | (f >> 13);
    }
}

#ifndef STANDALONE
#include "nnu_storage.h"
#endif

int storage_gamma_pow(Storage_Scheme s)
{
    switch (s) {
        case half: return 16;
        case mini: return 8;
        case micro: return 4;
        case nano: return 2;
        case two_mini: return 16;
        case four_micro: return 16;
    }

    return -1;

}

int storage_stride(Storage_Scheme s)
{
    switch (s) {
        case half: return 1;
        case mini: return 1;
        case micro: return 1;
        case nano: return 1;
        case two_mini: return 2;
        case four_micro: return 4;
    }

    return -1;
}

uint16_t float_to_storage(float i, Storage_Scheme s)
{
    switch (s) {
        case half: return float_to_half(i);
        case mini: return float_to_mini(i);
        case micro: return float_to_micro(i);
        case nano: return float_to_nano(i);
        case two_mini: return float_to_mini(i);
        case four_micro: return float_to_micro(i);
    }

    return -1;
}

void storage_to_float(float *i, uint16_t y, Storage_Scheme s)
{
    switch (s) {
        case half:
            i[0] = half_to_float(y);
            break;
        case mini:
            i[0] = mini_to_float(y);
            break;
        case micro:
            i[0] = micro_to_float(y);
            break;
        case nano:
            i[0] = nano_to_float(y);
            break;
        case two_mini: 
            i[0] = mini_to_float(y >> 8);
            i[1] = mini_to_float(y);
            break;
        case four_micro: 
            i[0] = micro_to_float(y >> 12);
            i[1] = micro_to_float(y >> 8);
            i[2] = micro_to_float(y >> 4);
            i[3] = micro_to_float(y);
            break;
    }
}

uint8_t float_to_nano(float i)
{
    uint8_t ret = 0;
    int mask = 0x007f0000;
    int fi, *fi_ptr;
    if(i < 0) {
        ret |= 1 << 1;
        i = -i;
    }

    i += 1;
    fi_ptr = (int *)&i;
    fi = *fi_ptr;
    ret |= (mask & fi) >> 22;

    return ret;
}

float nano_to_float(uint8_t y)
{
    float *f;
    int s;
    int b = 0x3f800000;

    if(y == 0) {
        return 0.0;
    }

    s = (y >> 1) << 31;
    b |= ((y << 1) >> 1) << 22;
    f = (float *)&b;
    *f = *f - 1;

    if(s == 0){
        return *f;    
    }else{
        return -*f;    
    }
}

uint8_t float_to_micro(float i)
{
    uint8_t ret = 0;
    int mask = 0x007f0000;
    int fi, *fi_ptr;

    if(i < 0){
        ret |= 1 << 3;
        i = -i;
    }

    i += 1;
    fi_ptr = (int *)&i;
    fi = *fi_ptr;
    ret |= (mask & fi) >> 20;

    return ret;
}

float micro_to_float(uint8_t y)
{
    float *f;
    int b = 0x3f800000;
    int s;

    if(y == 0) {
        return 0.0;
    }

    s = (y >> 3) << 31;
    b |= ((y << 1) >> 1) << 20;
    f = (float *)&b;
    *f = *f - 1;

    if(s == 0){
        return *f;    
    }else{
        return -*f;    
    }
}



uint8_t float_to_mini(float i)
{
    int fi, *fi_ptr;
    int mask = 0x007f0000;
    uint8_t ret = 0;

    if(i < 0) {
        ret |= 1 << 7;
        i = -i;
    }

    i += 1;
    fi_ptr = (int *)&i;
    fi = *fi_ptr;

    ret |= (mask & fi) >> 16;

    return ret;
}

float mini_to_float(uint8_t y)
{
    float *f;
    int s;
    int b = 0x3f800000;

    if(y == 0) {
        return 0.0;
    }

    s = (y >> 7) << 31;
    b |= ((y << 1) >> 1) << 16;
    f = (float *)&b;
    *f = *f - 1;

    if(s == 0) {
      return *f;    
    }
    else{
      return -*f;    
    }
}

static uint32_t  half_to_float_I(uint16_t y)
{
    int s = (y >> 15) & 0x00000001;  /* sign */
    int e = (y >> 10) & 0x0000001f;  /* exponent */
    int f =  y        & 0x000003ff;  /* fraction */

    /* need to handle 7c00 INF and fc00 -INF? */
    if (e == 0) {
        /* need to handle +-0 case f==0 or f=0x8000? */
        if (f == 0)  /* Plus or minus zero */
            return s << 31;
        else {       /* Denormalized number -- renormalize it */
            while (!(f & 0x00000400)) {
                f <<= 1;
                e -=  1;
            }
            e += 1;
            f &= ~0x00000400;
        }
    } else if (e == 31) {
        /* Inf */
        if (f == 0)
            return (s << 31) | 0x7f800000;
        /* NaN */
        else
            return (s << 31) | 0x7f800000 | (f << 13);
    }

    e = e + (127 - 15);
    f = f << 13;

    return ((s << 31) | (e << 23) | f);
}

static uint16_t float_to_half_I(uint32_t i)
{
    int s =  (i >> 16) & 0x00008000;                   /* sign */
    int e = ((i >> 23) & 0x000000ff) - (127 - 15);     /* exponent */
    int f =   i        & 0x007fffff;                   /* fraction */

    /* need to handle NaNs and Inf? */
    if (e <= 0) {
        if (e < -10) {
            if (s)                                             /* handle -0.0 */
               return 0x8000;
            else
               return 0;
        }
        f = (f | 0x00800000) >> (1 - e);
        return s | (f >> 13);
    } else if (e == 0xff - (127 - 15)) {
        if (f == 0)                                             /* Inf */
            return s | 0x7c00;
        else {                                                  /* NAN */
            f >>= 13;
            return s | 0x7c00 | f | (f == 0);
        }
    } else {
        if (e > 30)                                             /* Overflow */
            return s | 0x7c00;
        return s | (e << 10) | (f >> 13);
    }
}

float half_to_float(uint16_t y)
{
    union { float f; uint32_t i; } v;
    v.i = half_to_float_I(y);
    return v.f;
}

uint16_t float_to_half(float i)
{
    union { float f; uint32_t i; } v;
    v.f = i;
    return float_to_half_I(v.i);
}

#ifndef NNU_STORAGE_H
#define NNU_STORAGE_H

#include <stdio.h>
#include <inttypes.h>

typedef enum
{
    half,
    mini,
    micro,
    nano,
    two_mini,
    four_micro
} Storage_Scheme;

int storage_gamma_pow(Storage_Scheme);
int storage_stride(Storage_Scheme);
uint16_t float_to_storage(float, Storage_Scheme);
void storage_to_float(float*, uint16_t, Storage_Scheme);

uint8_t float_to_nano(float i);
float nano_to_float(uint8_t y);
uint8_t float_to_micro(float i);
float micro_to_float(uint8_t y);
float half_to_float(uint16_t);
uint16_t float_to_half(float);
uint8_t float_to_mini(float i);
float mini_to_float(uint8_t y);

#endif /* NNU_STORAGE_H */

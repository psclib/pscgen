#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

// bit returned at location
int bit_return(int a, int loc)   
{
    int buf = a & 1<<loc;

    if (buf == 0) return 0;
    else return 1; 
}

void print_bit(int k){
  int tt;
  for(tt = 31; tt >= 0; tt--)
      printf("%d", bit_return(k, tt));  
  printf("\n");
}

uint8_t float_to_mini(float i)
{
  uint8_t ret = 0;
  if(i < 0){
    ret |= 1 << 7;
    i = -i;
  }
  i += 1;
  int fi = *(int *)&i;
  int mask = 0x007f0000;
  ret |= (mask & fi) >> 16;
  return ret;
}

float mini_to_float(uint8_t y)
{
  float *f;
  int b = 0x3f800000;
  if(y == 0) {
      return 0.0;
  }
  int s = (y >> 7) << 31;
  b |= ((y << 1) >> 1) << 16;
  f = (float *)&b;
  *f = *f - 1;
  if(s == 0){
    return *f;    
  }else{
    return -*f;    
  }
}

int main_mini() 
{
  int i, j, tt, k;
  float f;

  for(j = 0; j < 256; j++) {
    f = mini_to_float(float_to_mini(mini_to_float(j)));
    printf("%f\n", f);
    k = *(int*)&f;
    print_bit(k);
  }

  return 0;
}

uint8_t float_to_micro(float i)
{
  uint8_t ret = 0;
  if(i < 0){
    ret |= 1 << 3;
    i = -i;
  }
  i += 1;
  int fi = *(int *)&i;
  int mask = 0x007f0000;
  ret |= (mask & fi) >> 20;
  return ret;
}

float micro_to_float(uint8_t y)
{
  float *f;
  int b = 0x3f800000;
  if(y == 0) {
      return 0.0;
  }
  int s = (y >> 3) << 31;
  b |= ((y << 1) >> 1) << 20;
  f = (float *)&b;
  *f = *f - 1;
  if(s == 0){
    return *f;    
  }else{
    return -*f;    
  }
}

int main_micro() 
{
  int i, j, tt, k;
  float f;

  for(j = 0; j < 16; j++) {
    f = micro_to_float(float_to_micro(micro_to_float(j)));
    // f = micro_to_float(j);
    printf("%f\n", f);
    k = *(int*)&f;
    print_bit(k);
  }

  return 0;
}

uint8_t float_to_nano(float i)
{
  uint8_t ret = 0;
  if(i < 0){
    ret |= 1 << 1;
    i = -i;
  }
  i += 1;
  int fi = *(int *)&i;
  int mask = 0x007f0000;
  ret |= (mask & fi) >> 22;
  return ret;
}

float nano_to_float(uint8_t y)
{
  float *f;
  int b = 0x3f800000;
  if(y == 0) {
      return 0.0;
  }
  int s = (y >> 1) << 31;
  b |= ((y << 1) >> 1) << 22;
  f = (float *)&b;
  *f = *f - 1;
  if(s == 0){
    return *f;    
  }else{
    return -*f;    
  }
}

int main_nano() 
{
  int i, j, tt, k;
  float f;

  for(j = 0; j < 4; j++) {
    f = nano_to_float(float_to_nano(nano_to_float(j)));
    // f = micro_to_float(j);
    printf("%f\n", f);
    k = *(int*)&f;
    print_bit(k);
  }

  return 0;
}


int main() 
{
  return main_nano();
}
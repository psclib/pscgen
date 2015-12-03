#include <stdio.h>
#include "wav.h"

float int16_audio_to_float(int16_t i){
  float f = (float)i / 32768;
  if(f > 1){
    f = 1;
  }else if(f < -1){
    f = -1;
  }
  return f;
}

main()
{
    int16_t *samples = NULL;

    wavread("test.wav", &samples);

    printf("No. of channels: %d\n",     header->num_channels);
    printf("Sample rate:     %d\n",     header->sample_rate);
    printf("Bit rate:        %dkbps\n", header->byte_rate*8 / 1000);
    printf("Bits per sample: %d\n\n",     header->bps);
    printf("# Samples: %d\n\n",     header->datachunk_size/header->num_channels);

    for (int i =0; i < header->datachunk_size/header->num_channels; i+=header->num_channels) {
        printf("%.7f\n", int16_audio_to_float(samples[i]));
    }

    // for (size_t i = 0; i < 100000; i++) {
    //   printf("%d ", samples[i]);
    //   /* code */
    // }
    // Modify the header values & samples before writing the new file
    wavwrite("test2.wav", samples);

    free(header);
    free(samples);
}

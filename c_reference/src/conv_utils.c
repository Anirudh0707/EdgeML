#include"conv_utils.h"
#include <math.h>

int prepareLowRankConvMat(float* out, float* W1, float* W2, unsigned rank, unsigned I, unsigned J){
  for(int i = 0 ; i < I; i++){
    for(int j = 0 ; j < J; j++){
        float sum = 0;
        for(int k = 0; k < rank ; k++){
            sum += (W1[i * rank + k] * W2[k * J + j]);
        }
        out[i * J + j] = sum;
    }
  }
  return 0;
}

int TanhGate(float* output_signal, float* input_signal, unsigned in_T, unsigned in_channels){
  unsigned int piv = in_channels>>1;
  for(int t = 0 ; t < in_T ; t++){
    for(int d = 0 ; d < piv ; d++){
      output_signal[t * piv + d] = sigmoid(input_signal[t * in_channels + d]) * tanh(input_signal[t * in_channels + (d + piv)]);
    }
  }
  return 0;
}

float relu(float x) {
  if (x < 0.0) return 0.0;
  else return x;
}

float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-1.0f * x));
}
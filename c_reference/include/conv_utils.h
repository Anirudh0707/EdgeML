#ifndef __CONVLAYER_UTILS__
#define __CONVLAYER_UTILS__

#include <math.h>

int prepareLowRankConvMat(float* out, float* W1, float* W2, unsigned rank, unsigned I, unsigned J);
int TanhGate(float* output_signal, float* input_signal, unsigned in_T, unsigned in_channels);
float sigmoid(float x);
float relu(float x);

#endif
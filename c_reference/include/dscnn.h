#ifndef __DSCNN__
#define __DSCNN__

#include"conv1d.h"
#include"conv_utils.h"
#include<stdlib.h>
#include<math.h>

int DSCNN_LR(float* output_signal, float* input_signal, unsigned in_T, unsigned in_channels, float* mean, float* var,
    unsigned affine, float* gamma, float* beta, unsigned in_place, unsigned cnn_hidden, int cnn_padding, unsigned cnn_kernel_size,
    const void* cnn_params, int cnn_activations);


#endif
#ifndef __DSCNN__
#define __DSCNN__

#include"conv1d.h"
#include"conv_utils.h"
#include<stdlib.h>
#include<math.h>

int DSCNN_LR(float* output_signal, float* input_signal, unsigned in_T, unsigned in_channels, float* mean, float* var,
    unsigned affine, float* gamma, float* beta, unsigned in_place, unsigned cnn_hidden, int cnn_padding, unsigned cnn_kernel_size,
    const void* cnn_params, int cnn_activations);

int DSCNN_LR_Point_Depth(float* output_signal, float* input_signal, unsigned in_T, unsigned in_channels, float* mean, float* var,
    unsigned affine, float* gamma, float* beta, unsigned in_place, unsigned depth_cnn_hidden, int depth_cnn_padding, 
    unsigned depth_cnn_kernel_size, const void* depth_cnn_params, int depth_cnn_activations, unsigned point_cnn_hidden, 
    int point_cnn_padding, unsigned point_cnn_kernel_size, const void* point_cnn_params, int point_cnn_activations, 
    int pool_padding, unsigned pool_kernel_size, int pool_activation);

#endif
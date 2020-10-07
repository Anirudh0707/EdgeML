#include"dscnn.h"
#include"conv1d.h"
#include"conv_utils.h"

int DSCNN_LR(float* output_signal, float* input_signal, unsigned in_T, unsigned in_channels, float* mean, float* var,
    unsigned affine, float* gamma, float* beta, unsigned in_place, unsigned cnn_hidden, int cnn_padding, unsigned cnn_kernel_size,
    const void* cnn_params, int cnn_activations){
    
    float out_T;
    
    // BatchNorm
    float* norm_out = (float*)malloc(in_T*in_channels*sizeof(float));
    BatchNorm1d(norm_out, input_signal, in_T, in_channels, 
    mean, var, affine, gamma, beta, in_place);

    // CNN
    out_T = in_T - cnn_kernel_size + 2*cnn_padding + 1;
    Conv1D_LR(output_signal, out_T, cnn_hidden, norm_out, 
    in_T, in_channels, cnn_padding, cnn_kernel_size, 
    cnn_params, cnn_activations);
    free(norm_out);
    return 0;
}
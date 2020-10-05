#include<stdlib.h>
#ifndef __CONVLAYERS_H__
#define __CONVLAYERS_H__

#define ERR_INTERMIDIATE_NOT_INIT -1
#define ERR_TEMPW_NOT_INIT -2
#define ERR_TEMPLRU_NOT_INIT -3
#define ERR_NORMFEATURES_NOT_INIT -4

/**
 * @brief Model paramters for the 1D Convolution Layer
 * @var       mean         pointer to mean of input vector for normalization, size inputDims
 * @var       stdDev       pointer to standard dev of input for normalization, size inputDims*steps
 * @var       W            pointer to convolutional weights W
 * @var       B            pointer to the bias vector for the convolution
 */
typedef struct ConvLayers_Params{
    float* W;
    float* B;
} ConvLayers_Params;


int Conv1D(float *output_signal, unsigned out_T, unsigned out_channels, const float *input_signal, 
    unsigned N, unsigned in_T, unsigned in_channels, int padding, unsigned kernel_size, 
    const void* params, int activations);

int Conv1D_Depth(float *output_signal, unsigned out_T, const float *input_signal, 
    unsigned N, unsigned in_T, unsigned in_channels, int padding, unsigned kernel_size, 
    const void* params, int activations);

// Low Rank
/**
 * @brief Model paramters for the Low Rank 1D Convolution Layer
 * @var       mean         pointer to mean of input vector for normalization, size inputDims
 * @var       stdDev       pointer to standard dev of input for normalization, size inputDims
 * @var       W1           pointer to first low-rank component of the convolutional weight W
 * @var       W2           pointer to second low-rank component of the convolutional weight W
 * @var       Rank         rank of W matrix
 * @var       B            pointer to the bias vector for the convolution
 */
typedef struct ConvLayers_LR_Params{
    float* W1;
    float* W2;
    float* B;
    unsigned rank;
} ConvLayers_LR_Params;

int Conv1D_LR(float *output_signal, unsigned out_T, unsigned out_channels, const float *input_signal, 
    unsigned N, unsigned in_T, unsigned in_channels, int padding, unsigned kernel_size, 
    const void* params, int activations);

int Conv1D_Depth_LR(float *output_signal, unsigned out_T, const float *input_signal, 
    unsigned N, unsigned in_T, unsigned in_channels, int padding, unsigned kernel_size, 
    const void* params, int activations);

//Pool
int AvgPool1D(float *output_signal, unsigned out_T, const float *input_signal, unsigned N, unsigned in_T, unsigned in_channels, 
    int padding, unsigned kernel_size, int activations);

#endif
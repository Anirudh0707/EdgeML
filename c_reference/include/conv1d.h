// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __CONV1D_H__
#define __CONV1D_H__

/**
 * @brief Model parameters for the 1D Convolution Layer
 * @var   W    pointer to convolutional weights W, size for regular = out_channels*in_channels*kernel_size, size for depth based = out_channels*kernel_size
 * @var   B    pointer to the bias vector for the convolution, shape = [out_channels]
 */
typedef struct ConvLayers_Params {
  float* W;
  float* B;
} ConvLayers_Params;

/**
 * @brief Model definition for the 1D Convolution Layer
 * @param[out]   output_signal    pointer to the output signal, size = out_time * out_channels
 * @param[in]    out_time            number of time steps in the output
 * @param[in]    out_channels     number of output channels for the output of the conv layer
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time             number of time steps in the input
 * @param[in]    in_channels      number of input channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(teim step in_time).
 * 
 * @param[in]    kernel_size      kernel size of the conv filter
 * @param[in]    params           weights, bias and other essential parameters used to describe the layer
 * @param[in]    activations      an integer to choose the type of activation function.
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 */
int conv1d(float *output_signal, unsigned out_time, unsigned out_channels, const float *input_signal, 
  unsigned in_time, unsigned in_channels, int padding, unsigned kernel_size, 
  const void* params, int activations);

/**
 * @brief Model definition for the 1D Depthwise Convolution Layer
 * @param[out]   output_signal    pointer to the output signal, size = out_time * in_channels
 * @param[in]    out_time            number of time steps in the output
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time             number of time steps in the input
 * @param[in]    in_channels      number of input channels. The output will have the same number of channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(teim step in_time).
 * 
 * @param[in]    kernel_size      kernel size of the conv filter
 * @param[in]    params           weights, bias and other essential parameters used to describe the layer
 * @param[in]    activations      an integer to choose the type of activation function.
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 */
int conv1d_depth(float *output_signal, unsigned out_time, const float *input_signal, 
  unsigned in_time, unsigned in_channels, int padding, unsigned kernel_size, 
  const void* params, int activations);


/**
 * @brief Model parameters for the 1D Low Rank Convolution Layer
 * @var    W1      pointer to the 1st low-rank component of the weights, size = out_channels * rank
 * @var    W2      pointer to the 2nd low-rank component of the weights, size for regular = rank * in_channels * kernel_size, size for depthwise = rank * kernel_size
 * @var    B       pointer to the bias vector for the convolution, shape = [out_channels]
 * @var    rank    rank of the weight tensor. A low rank decomposition typically used to reduce computation and storage
 */
typedef struct ConvLayers_LR_Params {
  float* W1;
  float* W2;
  float* B;
  unsigned rank;
} ConvLayers_LR_Params;

/**
 * @brief Model definition for the 1D Low Rank Convolution Layer
 * @brief Identical to the non-low-rank form. One modification is the mulitplication of the weights handeled within the layer
 * @param[out]   output_signal    pointer to the output signal, size = out_time * out_channels
 * @param[in]    out_time            number of time steps in the output
 * @param[in]    out_channels     number of output channels for the ouput of the conv layer
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time             number of time steps in the input
 * @param[in]    in_channels      number of input channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(teim step in_time).
 * 
 * @param[in]    kernel_size      kernel size of the conv filter
 * @param[in]    params           weights, bias and other essential parameters used to describe the layer
 * @param[in]    activations      an integer to choose the type of activation function.
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 */
int conv1d_lr(float *output_signal, unsigned out_time, unsigned out_channels, const float *input_signal, 
  unsigned in_time, unsigned in_channels, int padding, unsigned kernel_size, 
  const void* params, int activations);

/**
 * @brief Model definition for the 1D Depthwise Convolution Layer
 * @brief Identical to the non-low-rank form. One modification is the mulitplication of the weights handeled witin the layer
 * @param[out]   output_signal    pointer to the output signal, size = out_time * in_channels
 * @param[in]    out_time            number of time steps in the output
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time             number of time steps in the input
 * @param[in]    in_channels      number of input channels. The output will have the same number of channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(teim step in_time).
 * 
 * @param[in]    kernel_size      kernel size of the conv filter
 * @param[in]    params           weights, bias and other essential parameters used to describe the layer
 * @param[in]    activations      an integer to choose the type of activation function.
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 */
int conv1d_depth_lr(float *output_signal, unsigned out_time, const float *input_signal, 
  unsigned in_time, unsigned in_channels, int padding, unsigned kernel_size, 
  const void* params, int activations);

// Auxillary Layers
/**
 * @brief Model definition for the 1D Average Pooling Layer
 * @param[out]   output_signal    pointer to the output signal, size = out_time * in_channels. Provide Null/0 incase of in-place computation
 * @param[in]    out_time            number of time steps in the output
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time             number of time steps in the input
 * @param[in]    in_channels      number of input channels. The output will have the same number of channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(teim step in_time).
 * 
 * @param[in]    kernel_size      kernel size of the pool filter
 * @param[in]    activations      an integer to choose the type of activation function.
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 */
int avgpool1d(float *output_signal, unsigned out_time, const float *input_signal, unsigned in_time, unsigned in_channels, 
  int padding, unsigned kernel_size, int activations);

/**
 * @brief Model definition for the 1D batch Normalization Layer
 * @param[out]   output_signal    pointer to the output signal, size = out_time * in_channels. Provide Null/0 incase of in-place computation
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time             number of time steps in the input
 * @param[in]    in_channels      number of input channels. The output will have the same number of channels
 * @param[in]    mean             pointer to the mean for the batch normalization, size = in_channels
 * @param[in]    var              pointer to the variance for the batch normalization, size = in_channels
 * @param[in]    affine           whether the affine operations are applied
 * @param[in]    gamma            pointer to the scaling factors for the post-norm affine operation, size = in_channels
 * @param[in]    beta             pointer to the scalar offsets for the post-norm affine operation, size = in_channels
 * @param[in]    in_place         in-place computation of the batchnorm i.e. the output is stored in-place of the input signal. Storage efficient
 * @param[in]    eps              a very small +ve value to avoid division by 0. For the default value, assign = 0.00001
 */
int batchnorm1d(float* output_signal, float* input_signal, unsigned in_time, unsigned in_channels, 
  float* mean, float* var, unsigned affine, float* gamma , float * beta, unsigned in_place, float eps);

#endif

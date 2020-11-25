// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __CONV1D_H__
#define __CONV1D_H__

// Currently dilation is not supported. We have coded separate functions for regular and depthwise conv1d(and low-rank versions). They currently do not support the use of groups
// We use a custom matVec with offset (in utils) for our conv codes. This allows us to decompose the operation using the low-rank property and decrease the run-time
// The unoptimized version would be to first compute the weights and then perform the convolution

/**
 * @brief Model parameters for the 1D Convolution Layer
 * @var   W           pointer to the flattened conv weights, original shape for regular = [out_channels, kernel_size, in_channels], shape for depthwise = [in_channels, kernel_size, 1]
 * @var   B           pointer to the bias vector, original shape = [out_channels]
 * @var   depthwise   flag for deciding between regular(=0) and depthwise(=1) conv
 */
typedef struct ConvLayers_Params {
  const float* const W;
  const float* const B;
  unsigned depthwise;
} ConvLayers_Params;

/**
 * @brief Model definition for the 1D Convolution Layer. Currently only for dilation = 1
 * @param[out]   output_signal    pointer to the output signal, size = out_time * out_channels
 * @param[in]    out_time         number of time steps in the output
 * @param[in]    out_channels     number of output channels for the output of the conv layer
 *                                NOTE: out_channels == in_channels for depthwise. This is set manually in the function
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time          number of time steps in the input
 * @param[in]    in_channels      number of input channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(time step in_time).
 * @param[in]    kernel_size      kernel size of the conv filter
 * @param[in]    params           weights, bias and other essential parameters used to describe the layer
 * @param[in]    stride           stride length for the layer. input_time_iterator += stride for output_time_iterator +=1
 * @param[in]    activation       an integer to choose the type of activation function.
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 */
int conv1d(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation);

/**
 * @brief Model parameters for the 1D Convolution Layer
 * @var   W           pointer to the flattened conv weights, original shape for regular = [out_channels, kernel_size, in_channels], shape for depthwise = [in_channels, kernel_size, 1]
 * @var   B           pointer to the bias vector, original shape = [out_channels]
 * @var   block_size  block/tile size for the cache. Used for tiled MatMul
 */
typedef struct ConvLayers_Parallel_Params {
  const float* const W;
  const float* const B;
  unsigned block_size;
} ConvLayers_Parallel_Params;

/**
 * @brief Model definition for the 1D Parallel Convolution Layer. Currently only for dilation = 1. No depthwise.
 * @param[out]   output_signal    pointer to the output signal, size = out_time * in_channels
 * @param[in]    out_time         number of time steps in the output
 * @param[in]    out_channels     number of output channels for the output of the conv layer
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time          number of time steps in the input
 * @param[in]    in_channels      number of input channels. The output will have the same number of channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(time step in_time).
 * @param[in]    kernel_size      kernel size of the conv filter
 * @param[in]    params           weights, bias and other essential parameters used to describe the layer
 * @param[in]    stride           stride length for the layer. input_time_iterator += stride for output_time_iterator +=1
 * @param[in]    activation       an integer to choose the type of activation function.
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 */
int conv1d_parallel(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation);

/**
 * @brief Model parameters for the 1D Low Rank Convolution Layer.
 * @var    W1      pointer to the flattened 1st low-rank component of the weights, original shape = [out_channels, rank]. For depthwise out_channels = in_channels
 * @var    W2      pointer to the flattened 2nd low-rank component of the weights, original shape for regular = [rank, kernel_size, in_channels], shape for depthwise = [rank, kernel_size, 1]
 * @var    B       pointer to the flattened bias vector for the convolution, original shape = [out_channels]
 * @var    rank    rank of the weight tensor. A low-rank decomposition typically used to reduce computation and storage
 */
typedef struct ConvLayers_LR_Params {
  const float* const W1;
  const float* const W2;
  const float* const B;
  unsigned rank;
} ConvLayers_LR_Params;

/**
 * @brief Model definition for the 1D Low-Rank Convolution Layer. Currently only for dilation = 1. 
 * @brief Low-Rank and depthwise are incompatible as the low-rank decomposition of the weight matrix violates the depthwise conditions (out_channels % groups = 0, where groups = in_channels)
 * @param[out]   output_signal    pointer to the output signal, size = out_time * out_channels
 * @param[in]    out_time         number of time steps in the output
 * @param[in]    out_channels     number of output channels for the ouput of the conv layer
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time          number of time steps in the input
 * @param[in]    in_channels      number of input channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(time step in_time).
 * @param[in]    kernel_size      kernel size of the conv filter
 * @param[in]    params           weights, bias and other essential parameters used to describe the layer
 * @param[in]    stride           stride length for the layer. input_time_iterator += stride for output_time_iterator +=1
 * @param[in]    activation       an integer to choose the type of activation function.
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 */
int conv1d_lr(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation);

/**
 * @brief Model definition for the 1D Low-Rank Depthwise Convolution Layer. Currently only for dilation = 1
 * @param[out]   output_signal    pointer to the output signal, size = out_time * in_channels
 *                                NOTE: out_channels == in_channels for depthwise conv
 * @param[in]    out_time         number of time steps in the output
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time          number of time steps in the input
 * @param[in]    in_channels      number of input channels. The output will have the same number of channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(time step in_time).
 * @param[in]    kernel_size      kernel size of the conv filter
 * @param[in]    params           weights, bias and other essential parameters used to describe the layer
 * @param[in]    stride           stride length for the layer. input_time_iterator += stride for output_time_iterator +=1
 * @param[in]    activation       an integer to choose the type of activation function.
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 Note for the usage of conv1d_depth_lr:
 The depthwise with low-rank conv1d code currently uses an unoptimized implementation based on the computation of the conv weights, followed by the depthwise convolution
 The reason for using the unoptimized implementation for the depthwise with low-rank conv1d is due to the violation of the depthwise constraints when the low-rank decomposition is applied
 The use of depthwise convolution imposes a constraint on the out_channels of the weight matrix. When the low-rank decomposition is applied on top of this matrix, these constraints will be violated
 The decomposition converts the depthwise conv into a fully-connected layer and a convolution layer with weight [rank, kernel_size, 1]
 The new smaller weight matrix resembles a depthwise conv. But here, typically, in_channels > rank. This causes a violation in the matrix constraints for depthwise convolution
 Hence, due to the violation, we cannot split the opeartion and would need to use the unoptimized solution with full-rank weight computation followed by convolution

 The depthwise with low-rank code is recommended for extreme storage constraints with no major constraints on the computation cost
 For all other puposes, we recommend the use of a combinantion of depthwise conv, low-rank conv and regular conv
 */
int conv1d_depth_lr(float* output_signal, unsigned out_time, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation);

// Auxiliary Layers
/**
 * @brief Model definition for the 1D Average Pooling Layer. Currently only for dilation = 1
 * @param[out]   output_signal    pointer to the output signal, size = out_time * in_channels. Provide Null/0 in case of in-place computation
 *                                NOTE: out_channels == in_channels for avgpool
 * @param[in]    out_time         number of time steps in the output
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time          number of time steps in the input
 * @param[in]    in_channels      number of input channels. The output will have the same number of channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(time step in_time).
 * @param[in]    kernel_size      kernel size of the pool filter
 * @param[in]    stride           stride length for the layer. input_time_iterator += stride for output_time_iterator +=1
 * @param[in]    activation       an integer to choose the type of activation function.
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 */
int avgpool1d(float* output_signal, unsigned out_time, const float* input_signal,
  unsigned in_time, unsigned in_channels,
  unsigned padding, unsigned kernel_size, unsigned stride, unsigned activation);

/**
 * @brief Model definition for the 1D batch Normalization Layer
 * @param[out]   output_signal    pointer to the output signal, size = out_time * in_channels. Provide Null/0 in case of in-place computation
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time          number of time steps in the input
 * @param[in]    in_channels      number of input channels. The output will have the same number of channels
 * @param[in]    mean             pointer to the mean for the batch normalization, size = in_channels. if affine_config  = 2, then pass a NULL/0
 * @param[in]    var              pointer to the variance for the batch normalization, size = in_channels. if affine_config  = 2, then pass a NULL/0
 * @param[in]    affine_config    whether the affine operations are applied
 *                                if affine_config = 0, then only mean and var are used
 *                                if affine_config = 1, then mean, var, gamma and beta are used for the final computation.
 *                                if affine_config = 2, then only the gamma and beta are used. gamma = original_gamma/sqrt(var), beta = original_beta - gamma * mean/sqrt(var)
 *                                Note: Use affine_config = 2 for faster calculations. The new gamma and beta would need to be pre-computed, stored and passed
 * @param[in]    gamma            pointer to the scaling factors for the post-norm affine operation, size = in_channels. Provide Null/0 if affine_config is 0
 * @param[in]    beta             pointer to the offsets for the post-norm affine operation, size = in_channels. Provide Null/0 if affine_config is 0
 * @param[in]    in_place         in-place computation of the batchnorm i.e. the output is stored in-place of the input signal. Storage efficient
 * @param[in]    eps              a very small +ve value to avoid division by 0. For the default value, assign = 0.00001
 */
int batchnorm1d(float* output_signal, float* input_signal,
  unsigned in_time, unsigned in_channels,
  const float* const mean, const float* const var, 
  unsigned affine_config, const float* const gamma , const float* const beta,
  unsigned in_place, float eps);

#endif

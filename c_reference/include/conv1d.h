// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __CONV1D_H__
#define __CONV1D_H__

/**
 * @brief Model parameters for the 1D Convolution Layer
 * @var   W    pointer to convolution weights W, size for regular = out_channels * in_channels * kernel_size, size for depth based = out_channels * kernel_size
 * @var   B    pointer to the bias vector for the convolution, size = out_channels
 */
typedef struct ConvLayers_Params {
  const float* const W;
  const float* const B;
} ConvLayers_Params;

/**
 * @brief Model definition for the 1D Convolution Layer. Currently only for dilation = 1
 * @param[out]   output_signal    pointer to the output signal, size = out_time * out_channels
 * @param[in]    out_time         number of time steps in the output
 * @param[in]    out_channels     number of output channels for the output of the conv layer
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
 * @brief Model definition for the 1D Depthwise Convolution Layer. Currently only for dilation = 1
 * @param[out]   output_signal    pointer to the output signal, size = out_time * in_channels
 *                                NOTE: out_channels == in_channels for depthwise
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
 */
int conv1d_depth(float* output_signal, unsigned out_time, const float* input_signal, 
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size, 
  const void* params, unsigned stride, unsigned activation);

/**
 * @brief Model parameters for the 1D Low Rank Convolution Layer.
 * @var    W1      pointer to the 1st low-rank component of the weights, size = out_channels * rank
 * @var    W2      pointer to the 2nd low-rank component of the weights, size for regular = rank * in_channels * kernel_size, size for depthwise = rank * kernel_size
 * @var    B       pointer to the bias vector for the convolution, shape = [out_channels]
 * @var    rank    rank of the weight tensor. A low-rank decomposition typically used to reduce computation and storage
 */
typedef struct ConvLayers_LR_Params {
  const float* const W1;
  const float* const W2;
  const float* const B;
  unsigned rank;
} ConvLayers_LR_Params;

/**
 * @brief Model definition for the 1D Low-Rank Convolution Layer. Currently only for dilation = 1
 * @brief Identical to the non-low-rank form. One modification is the multiplication of the weights handled within the layer
 * @brief The Weights W1 and W2 are multiplied within the layer using a matmul function from utils. Operation : W1 * W2
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
 * @brief Identical to the non-low-rank form. One modification is the multiplication of the weights handled within the layer
 * @brief The Weights W1 and W2 are multiplied within the layer using a matmul function from utils. Operation : W1 * W2
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
 * @param[in]    mean             pointer to the mean for the batch normalization, size = in_channels
 * @param[in]    var              pointer to the variance for the batch normalization, size = in_channels
 * @param[in]    affine           whether the affine operations are applied
 * @param[in]    gamma            pointer to the scaling factors for the post-norm affine operation, size = in_channels. Provide Null/0 if affine is False(non-zero)
 * @param[in]    beta             pointer to the offsets for the post-norm affine operation, size = in_channels. Provide Null/0 if affine is False(non-zero)
 * @param[in]    in_place         in-place computation of the batchnorm i.e. the output is stored in-place of the input signal. Storage efficient
 * @param[in]    eps              a very small +ve value to avoid division by 0. For the default value, assign = 0.00001
 */
int batchnorm1d(float* output_signal, float* input_signal,
  unsigned in_time, unsigned in_channels,
  const float* const mean, const float* const var, 
  unsigned affine, const float* const gamma , const float* const beta,
  unsigned in_place, float eps);

#endif

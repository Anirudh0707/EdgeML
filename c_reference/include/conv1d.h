// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __CONV1D_H__
#define __CONV1D_H__

/*  All the matrices/tensors are stored in the row major format

   NOTES for the conv layers
-> For the non-depthwise cases, store the matrices as described below. Permutation might be necessary
-> The low-rank conv layers don't support depthwise computation. This is due to the out_channels/in_channels = 0 constarint. 
   For full-rank this is satisfied since out_channels = in_channels
   When the weight matrix is decomposed, the constarint is violated (since rank < out_channels ; and out_channels = in_channels for depthwise)
-> For the parallel cases, the non-overlapping cases of the convolution are computed parallelly using MatMul (since the blocked MatMul is faster)
   This howver is only valid for when the filter is fully in the input. There would be no-overlapping filters for the edge cases
   Hence the MatVec code(regular code) is used to calculate these cases

   Constraint
-> Due to the above reason, the parallel layers have to be used only for large in_time inputs
   This should typically be for in_time (without the padding) greater than 3 times the kernel_size
   For such short input cases, the code will either yield index-mismatched output or display a segmentration fault
-> This constraint is due to a lack of time steps to parallelize into a matrix
   For such cases, the MatVec would need to be used
*/

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
 *                                NOTE: out_channels = in_channels for depthwise. This is set manually in the function
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time          number of time steps in the input
 * @param[in]    in_channels      number of input channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(time step in_time-1)
 * @param[in]    kernel_size      kernel size of the conv filter
 * @param[in]    params           weights, bias and other essential parameters used to describe the layer
 * @param[in]    stride           stride length for the layer. input_time_iterator += stride for output_time_iterator +=1
 * @param[in]    activation       an integer to choose the type of activation function. More can be added as per the necessity
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 */
int conv1d(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation);

/**
 * @brief Model parameters for the 1D Parallel Convolution Layer
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
 * @param[out]   output_signal    pointer to the output signal, size = out_time * out_channels
 * @param[in]    out_time         number of time steps in the output
 * @param[in]    out_channels     number of output channels for the output of the conv layer
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time          number of time steps in the input
 * @param[in]    in_channels      number of input channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(time step in_time-1)
 * @param[in]    kernel_size      kernel size of the conv filter
 * @param[in]    params           weights, bias and other essential parameters used to describe the layer
 * @param[in]    stride           stride length for the layer. input_time_iterator += stride for output_time_iterator +=1
 * @param[in]    activation       an integer to choose the type of activation function. More can be added as per the necessity
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
 * @param[in]    out_channels     number of output channels for the output of the conv layer
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time          number of time steps in the input
 * @param[in]    in_channels      number of input channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(time step in_time-1)
 * @param[in]    kernel_size      kernel size of the conv filter
 * @param[in]    params           weights, bias and other essential parameters used to describe the layer
 * @param[in]    stride           stride length for the layer. input_time_iterator += stride for output_time_iterator +=1
 * @param[in]    activation       an integer to choose the type of activation function. More can be added as per the necessity
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 */
int conv1d_lr(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation);

/**
 * @brief Model parameters for the 1D Low Rank Parallel Convolution Layer.
 * @var    W1                  pointer to the flattened 1st low-rank component of the weights, original shape = [out_channels, rank]. For depthwise out_channels = in_channels
 * @var    W2                  pointer to the flattened 2nd low-rank component of the weights, original shape for regular = [rank, kernel_size, in_channels], shape for depthwise = [rank, kernel_size, 1]
 * @var    B                   pointer to the flattened bias vector for the convolution, original shape = [out_channels]
 * @var    rank                rank of the weight tensor. A low-rank decomposition typically used to reduce computation and storage
 * @var    block_size_to_lr    block/tile size for the cache. Used for tiled MatMul. Used for the input -> low-rank computation
 * @var    block_size_from_lr  block/tile size for the cache. Used for tiled MatMul. Used for the low-rank -> output computation
 */
typedef struct ConvLayers_LR_Parallel_Params {
  const float* const W1;
  const float* const W2;
  const float* const B;
  unsigned rank;
  unsigned block_size_to_lr;
  unsigned block_size_from_lr;
} ConvLayers_LR_Parallel_Params;

/**
 * @brief Model definition for the 1D Low-Rank Parallel Convolution Layer. Currently only for dilation = 1. 
 * @brief Low-Rank and depthwise are incompatible as the low-rank decomposition of the weight matrix violates the depthwise conditions (out_channels % groups = 0, where groups = in_channels)
 * @param[out]   output_signal    pointer to the output signal, size = out_time * out_channels
 * @param[in]    out_time         number of time steps in the output
 * @param[in]    out_channels     number of output channels for the output of the conv layer
 * @param[in]    input_signal     pointer to the input signal. size = in_time * in_channels
 * @param[in]    in_time          number of time steps in the input
 * @param[in]    in_channels      number of input channels
 * @param[in]    padding          padding applied to the input before the conv is performed.
 *                                Note: padding is applied to both the starting and ending of the input, along the time axis
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(time step in_time-1)
 * @param[in]    kernel_size      kernel size of the conv filter
 * @param[in]    params           weights, bias and other essential parameters used to describe the layer
 * @param[in]    stride           stride length for the layer. input_time_iterator += stride for output_time_iterator +=1
 * @param[in]    activation       an integer to choose the type of activation function. More can be added as per the necessity
 *                                0: none
 *                                1: sigmoid
 *                                2: tanh
 *                                3: relu
 */
int conv1d_lr_parallel(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
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
 *                                E.g : padding = 3, the input is padded with zeros(for 3 time steps), both before the input_signal(time step 0) and after the input_signal(time step in_time-1)
 * @param[in]    kernel_size      kernel size of the pool filter
 * @param[in]    stride           stride length for the layer. input_time_iterator += stride for output_time_iterator +=1
 * @param[in]    activation       an integer to choose the type of activation function. More can be added as per the necessity
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

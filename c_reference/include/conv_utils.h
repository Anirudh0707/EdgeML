// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __CONVLAYER_UTILS__
#define __CONVLAYER_UTILS__

/**
 * @brief Definition for the Utility Fucntion for Preparing the Low Rank Conv Weights
 * @param[out]   out     pointer to the output signal, size (regular) = out_T * in_channels * kernel_size, size (depthwise) = out_t * kernel_size
 * @param[in]    W1      1st component of the low rank weight tensor. size = out_channels * rank
 * @param[in]    W2      2nd component of the low rank weight tensor. size (regular) = rank * in_channels * kernel_size, size (depthwise)  = rank * kernel_size
 * @param[in]    rank    rank of the weight tensor. Low Rank
 * @param[in]    I       dim 0 for W1, value = out_channels
 * @param[in]    J       dim 1 for W2, value = in_channels * kernel_size, (or for depthwise) = kernel_size
 */
int MatMul(float* out, float* W1, float* W2, unsigned rank, unsigned I, unsigned J);

/**
 * @brief Definition for the Custom non-linear layer : The TanhGate
 * @param[out]   output_signal    pointer to the output signal, size = out_T * in_channels
 * @param[in]    input_signal     pointer to the input signal. size = in_T * in_channels
 * @param[in]    in_T             number of time steps in the input
 * @param[in]    in_channels      number of input channels. The output will have the half the number of channels. Recommended in_channel % 2 == 0
 */
int TanhGate(float* output_signal, float* input_signal, unsigned in_T, unsigned in_channels);

#endif
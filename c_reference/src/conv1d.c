// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "conv1d.h"
#include "utils.h"

int conv1d_lr(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  const ConvLayers_LR_Params* tparams= (ConvLayers_LR_Params*) params;
  
  // Perform the convolution. Zero-pad is from 0 to padding and in_time + padding to in_time + 2 * padding
  // Buffer for W2 out
  float* temp_rank_out = (float*)malloc(tparams->rank * sizeof(float));
  // Buffer for W1 out
  float* temp_out = (float*)malloc(out_channels * sizeof(float));
  for (unsigned t_in_start = 0, t_in_end = kernel_size - 1, t_out = 0; 
        t_out < out_time; t_out++, t_in_start += stride, t_in_end += stride) {
    unsigned t_index = t_out * out_channels;

    // There are typically 5 cases
    // 1) Filter not yet inside the input
    // 2) Filter partially inside the input
    // 3) Filter fully inside the input
    // 4) Filter partly outside the input
    // 5) Filter fully outside the input
    if ((t_in_start >= padding) && (t_in_end < (in_time + padding))) {
      // Filter fully inside the input. Kept as the initial condition, since this is the most common one
      offset_matVec_conv1d(tparams->W2, input_signal + (t_in_start - padding) * in_channels,
              tparams->rank, kernel_size * in_channels,
              kernel_size * in_channels, 1, 0, temp_rank_out);
      // The row_stride and ncols are provided with the same value in the function call below and vec_stride = 1, depthwise = 0 
      // Hence, this call will be the same as a regular MatVec function call (without any scaling)
      offset_matVec_conv1d(tparams->W1, temp_rank_out, out_channels,
              tparams->rank, tparams->rank, 1, 0, temp_out);        
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    } 
    else if ((t_in_start < padding) && (t_in_end >= padding)) {
      // Filter partially entered the input
      // In this case we using only a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Since we discard the last(or first) few column elements we can't iterate regularly(else we access the unnecessary values)
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tparams->W2 + (padding - t_in_start) * in_channels, 
                input_signal, tparams->rank, (t_in_end - padding + 1) * in_channels, 
                kernel_size * in_channels, 1, 0, temp_rank_out);
      // The row_stride and ncols are provided with the same value in the function call below and vec_stride = 1, depthwise = 0 
      // Hence, this call will be the same as a regular MatVec function call (without any scaling)
      offset_matVec_conv1d(tparams->W1, temp_rank_out, out_channels,
              tparams->rank, tparams->rank, 1, 0, temp_out); 
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    }
    else if (t_in_start < (in_time + padding) && (t_in_end >= (in_time + padding))) {
      // Filter partially exited the input
      // In this case we using only a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Since we discard the last(or first) few column elements we can't iterate regularly(else we access the unnecessary values)
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tparams->W2, input_signal  + (t_in_start - padding) * in_channels, 
                tparams->rank, (in_time + padding - t_in_start) * in_channels, 
                kernel_size * in_channels, 1, 0, temp_rank_out);
      // The row_stride and ncols are provided with the same value in the function call below and vec_stride = 1, depthwise = 0 
      // Hence, this call will be the same as a regular MatVec function call (without any scaling)
      offset_matVec_conv1d(tparams->W1, temp_rank_out, out_channels,
              tparams->rank, tparams->rank, 1, 0, temp_out); 
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    }
    else {
      // Filter completely in the padding region
      // The filter is either fully outside the input or has not yet entered the input
      // Hence we can skip the multiplication/addition operations and just set those output values to zero
      memset(output_signal + t_index, 0, out_channels * sizeof(float));
    }
    for (unsigned co = 0; co < out_channels; co++) {
      // Post-Conv activation. More activation functions can be added should the necessity arise
      if (activation == 1) {
        output_signal[t_index + co] = sigmoid(output_signal[t_index + co] + tparams->B[co]);
      }
      else if (activation == 2) {
        output_signal[t_index + co] = tanh(output_signal[t_index + co] + tparams->B[co]);
      }
      else if (activation == 3) {
        output_signal[t_index + co] = relu(output_signal[t_index + co] + tparams->B[co]);
      }
      else {
        output_signal[t_index + co] += tparams->B[co];
      }
    }
  }
  free(temp_out);
  free(temp_rank_out);
  return 0;
}

int conv1d_depth_lr(float* output_signal, unsigned out_time, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  const ConvLayers_LR_Params* tparams= (ConvLayers_LR_Params*) params;
  // Pre-computation of the weights for conv
  float* tempW = (float*)malloc(in_channels * kernel_size * sizeof(float));
  matMul(tparams->W1, tparams->W2, in_channels, tparams->rank,
    kernel_size, 0, 1.0, tempW);
  // Perform the Convolution. Pad is from 0 to padding and in_time + padding to in_time + 2 * padding
  float* temp_out = (float*)malloc(in_channels * sizeof(float));
  for (unsigned t_in_start = 0, t_in_end = kernel_size - 1, t_out = 0; 
        t_out < out_time; t_out++, t_in_start += stride, t_in_end += stride) {
    unsigned t_index = t_out * in_channels;

    // There are typically 5 cases
    // 1) Filter not yet inside the input
    // 2) Filter partially inside the input
    // 3) Filter fully inside the input
    // 4) Filter partly outside the input
    // 5) Filter fully outside the input
    if ((t_in_start >= padding) && (t_in_end < (in_time + padding))) {
      // Filter fully inside the input. Kept as the initial condition, since this is the most common one
      offset_matVec_conv1d(tempW, input_signal + (t_in_start - padding) * in_channels,
              in_channels, kernel_size,
              kernel_size, in_channels, 1, temp_out);
      memcpy(output_signal + t_index, temp_out, in_channels * sizeof(float));
    } 
    else if ((t_in_start < padding) && (t_in_end >= padding)) {
      // Filter partially entered the input
      // In this case we using only a part of the weight matrix(assuming shape = in_channels, kernel_size)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Since we discard the last(or first) few column elements we can't iterate regularly(else we access the unnecessary values)
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tempW + (padding - t_in_start), 
                input_signal, in_channels, (t_in_end - padding + 1), 
                kernel_size, in_channels, 1, temp_out);
      memcpy(output_signal + t_index, temp_out, in_channels * sizeof(float));
    }
    else if (t_in_start < (in_time + padding) && (t_in_end >= (in_time + padding))) {
      // Filter partially exited the input
      // In this case we using only a part of the weight matrix(assuming shape = in_channels, kernel_size)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Since we discard the last(or first) few column elements we can't iterate regularly(else we access the unnecessary values)
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tempW, input_signal  + (t_in_start - padding) * in_channels, 
                in_channels, (in_time + padding - t_in_start), 
                kernel_size, in_channels, 1, temp_out);
      memcpy(output_signal + t_index, temp_out, in_channels * sizeof(float));
    }
    else {
      // Filter completely in the padding region
      // The filter is either fully outside the input or has not yet entered the input
      // Hence we can skip the multiplication/addition operations and just set those output values to zero
      memset(output_signal + t_index, 0, in_channels * sizeof(float));
    }
    for (unsigned co = 0; co < in_channels; co++) {
      // Post-Conv activation. More activation functions can be added should the necessity arise
      if (activation == 1) {
        output_signal[t_index + co] = sigmoid(output_signal[t_index + co] + tparams->B[co]);
      }
      else if (activation == 2) {
        output_signal[t_index + co] = tanh(output_signal[t_index + co] + tparams->B[co]);
      }
      else if (activation == 3) {
        output_signal[t_index + co] = relu(output_signal[t_index + co] + tparams->B[co]);
      }
      else {
        output_signal[t_index + co] += tparams->B[co];
      }
    }
  }
  free(temp_out);
  free(tempW);
  return 0;
}

int conv1d(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  const ConvLayers_Params* tparams= (ConvLayers_Params*) params;

  // Perform the Convolution. Pad is from 0 to padding and in_time + padding to in_time + 2 * padding
  float* temp_out = (float*)malloc(out_channels * sizeof(float));
  for (unsigned t_in_start = 0, t_in_end = kernel_size - 1, t_out = 0; 
        t_out < out_time; t_out++, t_in_start += stride, t_in_end += stride) {
    unsigned t_index = t_out * out_channels;

    // There are typically 5 cases
    // 1) Filter not yet inside the input
    // 2) Filter partially inside the input
    // 3) Filter fully inside the input
    // 4) Filter partly outside the input
    // 5) Filter fully outside the input
    if ((t_in_start >= padding) && (t_in_end < (in_time + padding))) {
      // Filter fully inside the input. Kept as the initial condition, since this is the most common one
      offset_matVec_conv1d(tparams->W, input_signal + (t_in_start - padding) * in_channels,
              out_channels, kernel_size * in_channels,
              kernel_size * in_channels, 1, 0, temp_out);
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    } 
    else if ((t_in_start < padding) && (t_in_end >= padding)) {
      // Filter partially entered the input
      // In this case we using only a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Since we discard the last(or first) few column elements we can't iterate regularly(else we access the unnecessary values)
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tparams->W + (padding - t_in_start) * in_channels, 
                input_signal, out_channels, (t_in_end - padding + 1) * in_channels, 
                kernel_size * in_channels, 1, 0, temp_out);
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    }
    else if (t_in_start < (in_time + padding) && (t_in_end >= (in_time + padding))) {
      // Filter partially exited the input
      // In this case we using only a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Since we discard the last(or first) few column elements we can't iterate regularly(else we access the unnecessary values)
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tparams->W, input_signal  + (t_in_start - padding) * in_channels, 
                out_channels, (in_time + padding - t_in_start) * in_channels, 
                kernel_size * in_channels, 1, 0, temp_out);
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    }
    else {
      // Filter completely in the padding region
      // The filter is either fully outside the input or has not yet entered the input
      // Hence we can skip the multiplication/addition operations and just set those output values to zero
      memset(output_signal + t_index, 0, out_channels * sizeof(float));
    }
    for (unsigned co = 0; co < out_channels; co++) {
      // Post-Conv activation. More activation functions can be added should the necessity arise
      if (activation == 1) {
        output_signal[t_index + co] = sigmoid(output_signal[t_index + co] + tparams->B[co]);
      }
      else if (activation == 2) {
        output_signal[t_index + co] = tanh(output_signal[t_index + co] + tparams->B[co]);
      }
      else if (activation == 3) {
        output_signal[t_index + co] = relu(output_signal[t_index + co] + tparams->B[co]);
      }
      else {
        output_signal[t_index + co] += tparams->B[co];
      }
    }
  }
  free(temp_out);
  return 0;
}

int conv1d_depth(float* output_signal, unsigned out_time, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  const ConvLayers_Params* tparams= (ConvLayers_Params*) params;

  // Perform the Convolution. Pad is from 0 to padding and in_time + padding to in_time + 2 * padding
  float* temp_out = (float*)malloc(in_channels * sizeof(float));
  for (unsigned t_in_start = 0, t_in_end = kernel_size - 1, t_out = 0; 
        t_out < out_time; t_out++, t_in_start += stride, t_in_end += stride) {
    unsigned t_index = t_out * in_channels;

    // There are typically 5 cases
    // 1) Filter not yet inside the input
    // 2) Filter partially inside the input
    // 3) Filter fully inside the input
    // 4) Filter partly outside the input
    // 5) Filter fully outside the input
    if ((t_in_start >= padding) && (t_in_end < (in_time + padding))) {
      // Filter fully inside the input. Kept as the initial condition, since this is the most common one
      offset_matVec_conv1d(tparams->W, input_signal + (t_in_start - padding) * in_channels,
              in_channels, kernel_size,
              kernel_size, in_channels, 1, temp_out);
      memcpy(output_signal + t_index, temp_out, in_channels * sizeof(float));
    } 
    else if ((t_in_start < padding) && (t_in_end >= padding)) {
      // Filter partially entered the input
      // In this case we using only a part of the weight matrix(assuming shape = in_channels, kernel_size)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Since we discard the last(or first) few column elements we can't iterate regularly(else we access the unnecessary values)
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tparams->W + (padding - t_in_start), 
                input_signal, in_channels, (t_in_end - padding + 1), 
                kernel_size, in_channels, 1, temp_out);
      memcpy(output_signal + t_index, temp_out, in_channels * sizeof(float));
    }
    else if (t_in_start < (in_time + padding) && (t_in_end >= (in_time + padding))) {
      // Filter partially exited the input
      // In this case we using only a part of the weight matrix(assuming shape = in_channels, kernel_size)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Since we discard the last(or first) few column elements we can't iterate regularly(else we access the unnecessary values)
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tparams->W, input_signal  + (t_in_start - padding) * in_channels, 
                in_channels, (in_time + padding - t_in_start), 
                kernel_size, in_channels, 1, temp_out);
      memcpy(output_signal + t_index, temp_out, in_channels * sizeof(float));
    }
    else {
      // Filter completely in the padding region
      // The filter is either fully outside the input or has not yet entered the input
      // Hence we can skip the multiplication/addition operations and just set those output values to zero
      memset(output_signal + t_index, 0, in_channels * sizeof(float));
    }
    for (unsigned co = 0; co < in_channels; co++) {
      // Post-Conv activation. More activation functions can be added should the necessity arise
      if (activation == 1) {
        output_signal[t_index + co] = sigmoid(output_signal[t_index + co] + tparams->B[co]);
      }
      else if (activation == 2) {
        output_signal[t_index + co] = tanh(output_signal[t_index + co] + tparams->B[co]);
      }
      else if (activation == 3) {
        output_signal[t_index + co] = relu(output_signal[t_index + co] + tparams->B[co]);
      }
      else {
        output_signal[t_index + co] += tparams->B[co];
      }
    }
  }
  free(temp_out);
  return 0;
}

int avgpool1d(float* output_signal, unsigned out_time, const float* input_signal,
  unsigned in_time, unsigned in_channels,
  unsigned padding, unsigned kernel_size, unsigned stride, unsigned activation) {

  // Iterate over the time steps and average them. Similar to Conv1D_Dept with a filter kernel of ones
  float scale = 1.0/(float)kernel_size;
  for (unsigned t_in = 0, t_out = 0; t_out < out_time; t_out++, t_in += stride) {
    for (unsigned ci = 0; ci < in_channels; ci++) {
      float sum = 0;
      for (unsigned tf = 0; tf < kernel_size; tf++) {
        if (((t_in + tf) < padding) || ((t_in + tf) >= (in_time + padding))) {
          continue;
        }
        else {
          sum += (input_signal[((tf + t_in) - padding) * in_channels + ci]);
        }
      }
      if (activation == 1) {
        output_signal[t_out * in_channels + ci] = sigmoid(sum * scale);
      }
      else if (activation == 2) {
        output_signal[t_out * in_channels + ci] = tanh(sum * scale);
      }
      else if (activation == 3) {
        output_signal[t_out * in_channels + ci] = relu(sum * scale);
      }
      else {
        output_signal[t_out * in_channels + ci] = sum * scale;
      }
    }
  }
  return 0;
}

int batchnorm1d(float* output_signal, float* input_signal,
  unsigned in_time, unsigned in_channels,
  const float* const mean, const float* const var,
  unsigned affine_config, const float* const gamma , const float* const beta,
  unsigned in_place, float eps) {
  // Check if affine values was learnt
  if (affine_config == 1) {
    // Check for in-place computation
    if (in_place) {
      for (unsigned t = 0; t < in_time; t++) {
        for (unsigned d = 0; d < in_channels; d++) {
          input_signal[t * in_channels + d] = gamma[d]
                                              * ((input_signal[t * in_channels + d]
                                              - mean[d]) / sqrt(var[d] + eps))
                                              + beta[d];
        }
      }
    }
    else {
      for (unsigned t = 0; t < in_time; t++) {
        for (unsigned d = 0; d < in_channels; d++) {
          output_signal[t * in_channels + d] = gamma[d]
                                               * ((input_signal[t * in_channels + d]
                                               - mean[d]) / sqrt(var[d] + eps))
                                               + beta[d];
        }
      }
    }
  }
  else if (affine_config == 2) {
    // Check for in-place computation
    if (in_place) {
      for (unsigned t = 0; t < in_time; t++) {
        for (unsigned d = 0; d < in_channels; d++) {
          input_signal[t * in_channels + d] = (gamma[d]
                                               * input_signal[t * in_channels + d])
                                               + beta[d];
        }
      }
    }
    else {
      for (unsigned t = 0; t < in_time; t++) {
        for (unsigned d = 0; d < in_channels; d++) {
          output_signal[t * in_channels + d] = (gamma[d]
                                                * input_signal[t * in_channels + d])
                                                + beta[d];
        }
      }
    }
  }
  else {
      // Check for in-place computation
    if (in_place) {
      for (unsigned t = 0; t < in_time; t++) {
        for (unsigned d = 0; d < in_channels; d++) {
          input_signal[t * in_channels + d] = (input_signal[t * in_channels + d]
                                               - mean[d]) / sqrt(var[d] + eps);
        }
      }
    }
    else {
      for (unsigned t = 0; t < in_time; t++) {
        for (unsigned d = 0; d < in_channels; d++) {
          output_signal[t * in_channels + d] = (input_signal[t * in_channels + d] 
                                                - mean[d]) / sqrt(var[d] + eps);
        }
      }
    }
  }
  return 0;
}

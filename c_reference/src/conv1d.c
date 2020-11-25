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

int conv1d_lr_parallel(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  // Compute the LCM. Necessary  for the MatMul iterations
  unsigned total_in_cols, num_iter, ncols = kernel_size * in_channels; // MatMul variables
  unsigned gcd, temp, temp_lcm_1 = kernel_size, temp_lcm_2 = stride; // LCM variables
  while (1){
    if(!temp_lcm_1){
      gcd = temp_lcm_2;
      break;
    }
    else{
      temp = temp_lcm_2;
      temp_lcm_2 = temp_lcm_1;
      temp_lcm_1 = temp % temp_lcm_1;
    }
  }
  // The LCM is the number of time steps to linearise into one vector. Non-overlaping kernels are the rows
  unsigned lcm = kernel_size * stride / gcd;
  total_in_cols = lcm * in_channels ;
  num_iter = lcm / stride;
  
  const ConvLayers_LR_Parallel_Params* tparams = (ConvLayers_LR_Parallel_Params*) params;
  // Perform the convolution. Zero-pad is from 0 to padding and in_time + padding to in_time + 2 * padding
  // There are typically 5 cases
  // 1) Filter not yet inside the input
  // 2) Filter partially inside the input
  // 3) Filter fully inside the input
  // 4) Filter partly outside the input
  // 5) Filter fully outside the input

  // Buffer to hold the output. For corner cases, this will be realtively big. 
  // But will be needed for the central condition (filter inside input).
  unsigned buffer_steps = in_time / lcm, rank = tparams->rank;
  // Buffer for W2 out
  float* temp_rank_out = (float*)malloc(buffer_steps * tparams->rank * sizeof(float));
  // Buffer for W1 out
  float* temp_out = (float*)malloc(out_channels * tparams->rank * sizeof(float));

  unsigned t_in_start, t_in_end, t_out; // Values are needed outside the loops. Hence declared here
  for (t_in_start = 0, t_in_end = kernel_size - 1, t_out = 0; 
        t_in_start < padding && t_out < out_time;
        t_out++, t_in_start += stride, t_in_end += stride) {
    if (t_in_end < padding) {
      // Filter outside the input region and in the padded region
      memset(output_signal + t_out * out_channels, 0, out_channels * sizeof(float));
    } 
    else { //(t_in_end >= padding)
      // Filter partially entered the input
      // In this case we using only a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tparams->W2 + (padding - t_in_start) * in_channels, 
                input_signal, rank, (t_in_end - padding + 1) * in_channels, 
                kernel_size * in_channels, 1, 0, temp_rank_out);
      // The row_stride and ncols are provided with the same value in the function call below and vec_stride = 1, depthwise = 0 
      // Hence, this call will be the same as a regular MatVec function call (without any scaling)
      offset_matVec_conv1d(tparams->W1, temp_rank_out, out_channels,
              rank, rank, 1, 0, temp_out); 
      memcpy(output_signal + t_out * out_channels, temp_out, out_channels * sizeof(float));
    }
  }
  // The main part, when the Filter is fully inside the input. We can think of the non-overlapping cases as parallel cases
  // For the overlapingcases, each of the iterations is used for the same. Hence we find the lcm for the best parallel factor
  // Using the above logic, we can convert the matVec opeartion into a matMul operation
  // Ideally both implementation would be the same. However for edge devices the matMul was found to be faster matVec (both tilied)
  t_in_start -= padding; // remove the padding offset temporarily
  for (unsigned iter = 0; iter < num_iter; iter++, t_in_start += stride, t_out++) {
    memset(temp_rank_out, 0, buffer_steps * tparams->rank * sizeof(float));
    memset(temp_out, 0, out_channels * tparams->rank * sizeof(float));
    unsigned in_rows = (in_time - t_in_start) / lcm;
    if (t_in_end < (t_in_start + ((in_rows - 1) * lcm))) {
      // t_in_end is used to find the furthest time step was used for the calculation
      // This value will be use for the final iteration
      t_in_end = ((in_rows - 1) * lcm) + t_in_start;
    }
    transposed_tiledMatMul(input_signal  + t_in_start * in_channels , tparams->W2,  
                            in_rows, ncols, rank,
                            total_in_cols, ncols,
                            temp_rank_out, tparams->block_size);
    transposed_tiledMatMul(temp_rank_out , tparams->W1,  
                            in_rows, rank, out_channels,
                            rank, rank,
                            temp_out, tparams->block_size);
    // Copy all the data into the output
    for (unsigned t_iter = 0; t_iter < in_rows; t_iter++) {
      memcpy(output_signal + (t_out + t_iter * num_iter) * out_channels,
              temp_out + t_iter * out_channels, out_channels * sizeof(float));
    }
  }
  // Initialize the time iterators outside the loop for readability
  t_in_start = t_in_end + padding + stride; // Add the padding and stride offsets again
  t_in_end = t_in_start + kernel_size - 1;
  t_out = t_in_start / stride;
  for (; t_out < out_time; t_out++, t_in_start += stride, t_in_end += stride) {
    if (t_in_start < (in_time + padding) && (t_in_end >= (in_time + padding))) {
      // Filter partially exited the input
      // In this case we using only a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tparams->W2, input_signal  + (t_in_start - padding) * in_channels, 
                rank, (in_time + padding - t_in_start) * in_channels, 
                kernel_size * in_channels, 1, 0, temp_rank_out);
      // The row_stride and ncols are provided with the same value in the function call below and vec_stride = 1, depthwise = 0 
      // Hence, this call will be the same as a regular MatVec function call (without any scaling)
      offset_matVec_conv1d(tparams->W1, temp_rank_out, out_channels,
              rank, rank, 1, 0, temp_out);
      memcpy(output_signal + t_out * out_channels, temp_out, out_channels * sizeof(float));
    }
    else if (t_in_start < (in_time + padding) && (t_in_end < (in_time + padding))) {
      // Filter fully in the input but very close to the edges. 
      // Due to the lcm divisibility constrinat in the parallel step, these computations might be skipped
      offset_matVec_conv1d(tparams->W2, input_signal + (t_in_start - padding) * in_channels,
              rank, kernel_size * in_channels,
              kernel_size * in_channels, 1, 0, temp_rank_out);
      // The row_stride and ncols are provided with the same value in the function call below and vec_stride = 1, depthwise = 0 
      // Hence, this call will be the same as a regular MatVec function call (without any scaling)
      offset_matVec_conv1d(tparams->W1, temp_rank_out, out_channels,
              rank, rank, 1, 0, temp_out);
      memcpy(output_signal + t_out * out_channels, temp_out, out_channels * sizeof(float));
    }
    else {
      // Filter completely outside the input and in the padding region
      memset(output_signal + t_out * out_channels, 0, out_channels * sizeof(float));
    }
  }
  // Bias and activation
  for (t_out = 0; t_out < out_time; t_out++) {
    unsigned t_index = t_out * out_channels;
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

int conv1d(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  const ConvLayers_Params* tparams= (ConvLayers_Params*) params;
  unsigned vec_stride = 1, cols_scale = in_channels;
  if (tparams->depthwise) {
    vec_stride = in_channels;
    out_channels = in_channels;
    cols_scale = 1;
  }

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
              out_channels, kernel_size * cols_scale,
              kernel_size * cols_scale, vec_stride, tparams->depthwise, temp_out);
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    } 
    else if ((t_in_start < padding) && (t_in_end >= padding)) {
      // Filter partially entered the input
      // In this case we using only a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tparams->W + (padding - t_in_start) * cols_scale, 
                input_signal, out_channels, (t_in_end - padding + 1) * cols_scale,
                kernel_size * cols_scale, vec_stride, tparams->depthwise, temp_out);
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    }
    else if (t_in_start < (in_time + padding) && (t_in_end >= (in_time + padding))) {
      // Filter partially exited the input
      // In this case we using only a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tparams->W, input_signal  + (t_in_start - padding) * in_channels, 
                out_channels, (in_time + padding - t_in_start) * cols_scale,
                kernel_size * cols_scale, vec_stride, tparams->depthwise, temp_out);
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    }
    else {
      // Filter completely in the padding region
      // The filter is either fully outside the input or has not yet entered the input
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

int conv1d_parallel(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {
  
  // Compute the LCM. Necessary  for the MatMul iterations
  unsigned total_in_cols, num_iter, ncols = kernel_size * in_channels; // MatMul variables
  unsigned gcd, temp, temp_lcm_1 = kernel_size, temp_lcm_2 = stride; // LCM variables
  while (1){
    if(!temp_lcm_1){
      gcd = temp_lcm_2;
      break;
    }
    else{
      temp = temp_lcm_2;
      temp_lcm_2 = temp_lcm_1;
      temp_lcm_1 = temp % temp_lcm_1;
    }
  }
  // The LCM is the number of time steps to linearise into one vector. Non-overlaping kernels are the rows
  unsigned lcm = kernel_size * stride / gcd;
  total_in_cols = lcm * in_channels ;
  num_iter = lcm / stride;
  
  const ConvLayers_Parallel_Params* tparams = (ConvLayers_Parallel_Params*) params;
  // Perform the Convolution. Pad is from 0 to padding and in_time + padding to in_time + 2 * padding
  // There are typically 5 cases
  // 1) Filter not yet inside the input
  // 2) Filter partially inside the input
  // 3) Filter fully inside the input
  // 4) Filter partly outside the input
  // 5) Filter fully outside the input

  // Buffer to hold the output. For corner cases, this will be realtively big. 
  // But will be needed for the central condition (filter inside input).
  unsigned buffer_steps = in_time / lcm;
  float* temp_out = (float*)malloc(buffer_steps * out_channels * sizeof(float));
  unsigned t_in_start, t_in_end, t_out; // Values are needed outside the loops. Hence declared here
  for (t_in_start = 0, t_in_end = kernel_size - 1, t_out = 0; 
        t_in_start < padding && t_out < out_time;
        t_out++, t_in_start += stride, t_in_end += stride) {
    if (t_in_end < padding) {
      // Filter outside the input region and in the padded region
      memset(output_signal + t_out * out_channels, 0, out_channels * sizeof(float));
    } 
    else { //(t_in_end >= padding)
      // Filter partially entered the input
      // In this case we using only a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tparams->W + (padding - t_in_start) * in_channels, 
                input_signal, out_channels, (t_in_end - padding + 1) * in_channels, 
                ncols, 1, 0, temp_out);
      memcpy(output_signal + t_out * out_channels, temp_out, out_channels * sizeof(float));
    }
  }
  // The main part, when the Filter is fully inside the input. We can think of the non-overlapping cases as parallel cases
  // For the overlapingcases, each of the iterations is used for the same. Hence we find the lcm for the best parallel factor
  // Using the above logic, we can convert the matVec opeartion into a matMul operation
  // Ideally both implementation would be the same. However for edge devices the matMul was found to be faster matVec (both tilied)
  t_in_start -= padding; // remove the padding offset temporarily
  for (unsigned iter = 0; iter < num_iter; iter++, t_in_start += stride, t_out++) {
    memset(temp_out, 0, buffer_steps * out_channels * sizeof(float));
    unsigned in_rows = (in_time - t_in_start) / lcm;
    if (t_in_end < (t_in_start + ((in_rows - 1) * lcm))) {
      // t_in_end is used to find the furthest time step was used for the calculation
      // This value will be use for the final iteration
      t_in_end = ((in_rows - 1) * lcm) + t_in_start;
    }
    transposed_tiledMatMul(input_signal  + t_in_start * in_channels , tparams->W,  
                            in_rows, ncols, out_channels,
                            total_in_cols, ncols,
                            temp_out, tparams->block_size);
    // Copy all the data into the output
    for (unsigned t_iter = 0; t_iter < in_rows; t_iter++) {
      memcpy(output_signal + (t_out + t_iter * num_iter) * out_channels,
              temp_out + t_iter * out_channels, out_channels * sizeof(float));
    }
  }
  // Initialize the time iterators outside the loop for readability
  t_in_start = t_in_end + padding + stride; // Add the padding and stride offsets again
  t_in_end = t_in_start + kernel_size - 1;
  t_out = t_in_start / stride;
  for (; t_out < out_time; t_out++, t_in_start += stride, t_in_end += stride) {
    if (t_in_start < (in_time + padding) && (t_in_end >= (in_time + padding))) {
      // Filter partially exited the input
      // In this case we using only a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps
      // Hence the number of columns needed reduces. But the whole matrix is a continuous piece of memory. So we need to discard/skip certain columns
      // Hence we provide a separate row_stride to hop from one row to another
      offset_matVec_conv1d(tparams->W, input_signal  + (t_in_start - padding) * in_channels, 
                out_channels, (in_time + padding - t_in_start) * in_channels, 
                ncols, 1, 0, temp_out);
      memcpy(output_signal + t_out * out_channels, temp_out, out_channels * sizeof(float));
    }
    else if (t_in_start < (in_time + padding) && (t_in_end < (in_time + padding))) {
      // Filter fully in the input but very close to the edges. 
      // Due to the lcm divisibility constrinat in the parallel step, these computations might be skipped
      offset_matVec_conv1d(tparams->W, input_signal + (t_in_start - padding) * in_channels,
              out_channels, kernel_size * in_channels,
              kernel_size * in_channels, 1, 0, temp_out);
      memcpy(output_signal + t_out * out_channels, temp_out, out_channels * sizeof(float));
    }
    else {
      // Filter completely outside the input and in the padding region
      memset(output_signal + t_out * out_channels, 0, out_channels * sizeof(float));
    }
  }
  // Bias and activation
  for (t_out = 0; t_out < out_time; t_out++) {
    unsigned t_index = t_out * out_channels;
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

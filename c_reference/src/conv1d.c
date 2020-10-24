// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdlib.h>
#include <math.h>
#include "conv1d.h"
#include "utils.h"

int conv1d_lr(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  const ConvLayers_LR_Params* tparams= (ConvLayers_LR_Params*) params;
  
  float* tempW = (float*)malloc(out_channels * in_channels * kernel_size * sizeof(float));
  matmul(tparams->W1, tparams->W2, out_channels, tparams->rank,
    in_channels * kernel_size, 0, 1.0, tempW);
  // Perform the Convolution
  for (unsigned t_in = 0, t_out = 0; t_out < out_time; t_out++, t_in += stride) {
    for (unsigned co = 0; co < out_channels; co++) {
      float sum = 0;
      for (unsigned tf = 0; tf < kernel_size; tf++) {
        for (unsigned ci = 0; ci < in_channels; ci++) {
          if (((t_in + tf) < padding) || ((t_in + tf) >= (in_time + padding))) {
            continue;
          }
          else {
            sum += (input_signal[((tf + t_in) - padding) * in_channels + ci] 
                    * tempW[co * in_channels * kernel_size + ci * kernel_size + tf]);
          }
        }
      }
      // Post-Conv activation. More activation functions can be added should the necessity arise
      if (activation == 1) {
        output_signal[t_out * out_channels + co] = sigmoid(sum + tparams->B[co]);
      }
      else if (activation == 2) {
        output_signal[t_out * out_channels + co] = tanh(sum + tparams->B[co]);
      }
      else if (activation == 3) {
        output_signal[t_out * out_channels + co] = relu(sum + tparams->B[co]);
      }
      else {
        output_signal[t_out * out_channels + co] = sum + tparams->B[co];
      }
    }
  }
  free(tempW);
  return 0;
}

int conv1d_depth_lr(float* output_signal, unsigned out_time, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  const ConvLayers_LR_Params* tparams= (ConvLayers_LR_Params*) params;

  float* tempW = (float*)malloc(in_channels * kernel_size * sizeof(float));
  matmul(tparams->W1, tparams->W2, in_channels,tparams->rank,
    kernel_size, 0, 1.0, tempW);
  // Perform the Convolution
  for (unsigned t_in = 0, t_out = 0; t_out < out_time; t_out++, t_in += stride) {
    for (unsigned ci = 0; ci < in_channels; ci++) {
      float sum = 0;
      for (unsigned tf = 0; tf < kernel_size; tf++) {
        if (((t_in + tf) < padding) || ((t_in + tf) >= (in_time + padding))) {
          continue;
        }
        else {
          sum += (input_signal[((tf + t_in) - padding) * in_channels + ci] 
                  * tempW[ci * kernel_size + tf]);
        }
      }
      // Post-Conv activation. More activation functions can be added should the necessity arise
      if (activation == 1) {
        output_signal[t_out * in_channels + ci] = sigmoid(sum + tparams->B[ci]);
      }
      else if (activation == 2) {
        output_signal[t_out * in_channels + ci] = tanh(sum + tparams->B[ci]);
      }
      else if (activation == 3) {
        output_signal[t_out * in_channels + ci] = relu(sum + tparams->B[ci]);
      }
      else {
        output_signal[t_out * in_channels + ci] = sum + tparams->B[ci];
      }
    }
  }    
  free(tempW);
  return 0;
}

int conv1d(float* output_signal, unsigned out_time, unsigned out_channels, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  const ConvLayers_Params* tparams= (ConvLayers_Params*) params;

  // Perform the Convolution
  for (unsigned t_in = 0, t_out = 0; t_out < out_time; t_out++, t_in += stride) {
      for (unsigned co = 0; co < out_channels; co++) {
      float sum = 0;
      for (unsigned tf = 0; tf < kernel_size; tf++) {
        for (unsigned ci = 0; ci < in_channels; ci++) {
          if (((t_in + tf) < padding) || ((t_in + tf) >= (in_time + padding))) {
            continue;
          }
          else {
            sum += (input_signal[((tf + t_in) - padding) * in_channels + ci] 
                    * tparams->W[co * in_channels * kernel_size + ci * kernel_size + tf]);
          }
        }
      }
      // Post-Conv activation. More activation functions can be added should the necessity arise
      if (activation == 1) {
        output_signal[t_out * out_channels + co] = sigmoid(sum + tparams->B[co]);
      }
      else if (activation == 2) {
        output_signal[t_out * out_channels + co] = tanh(sum + tparams->B[co]);
      }
      else if (activation == 3) {
        output_signal[t_out * out_channels + co] = relu(sum + tparams->B[co]);
      }
      else {
        output_signal[t_out * out_channels + co] = sum + tparams->B[co];
      }
    }
  }    
  return 0;
}

int conv1d_depth(float* output_signal, unsigned out_time, const float* input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  const ConvLayers_Params* tparams= (ConvLayers_Params*) params;

  // Perform the Convolution
  for (unsigned t_in = 0, t_out = 0; t_out < out_time; t_out++, t_in += stride) {
    for (unsigned ci = 0; ci < in_channels; ci++) {
      float sum = 0;
      for (unsigned tf = 0; tf < kernel_size; tf++) {
        if (((t_in + tf) < padding) || ((t_in + tf) >= (in_time + padding))) {
          continue;
        }
        else {
          sum += (input_signal[((tf + t_in) - padding) * in_channels + ci] 
                  * tparams->W[ci * kernel_size + tf]);
        }
      }
      // Post-Conv activation. More activation functions can be added should the necessity arise
      if (activation == 1) {
        output_signal[t_out * in_channels + ci] = sigmoid(sum + tparams->B[ci]);
      }
      else if (activation == 2) {
        output_signal[t_out * in_channels + ci] = tanh(sum + tparams->B[ci]);
      }
      else if (activation == 3) {
        output_signal[t_out * in_channels + ci] = relu(sum + tparams->B[ci]);
      }
      else {
        output_signal[t_out * in_channels + ci] = sum + tparams->B[ci];
      }
    }
  }
  return 0;
}

int avgpool1d(float* output_signal, unsigned out_time, const float* input_signal,
  unsigned in_time, unsigned in_channels,
  unsigned padding, unsigned kernel_size, unsigned stride, unsigned activation) {

  // Iterate over the time steps and average them. Similar to Conv1D_Dept with a filter kernel of ones
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
        output_signal[t_out * in_channels + ci] = sigmoid(sum / (float)kernel_size);
      }
      else if (activation == 2) {
        output_signal[t_out * in_channels + ci] = tanh(sum / (float)kernel_size);
      }
      else if (activation == 3) {
        output_signal[t_out * in_channels + ci] = relu(sum / (float)kernel_size);
      }
      else {
        output_signal[t_out * in_channels + ci] = sum / (float)kernel_size;
      }
    }
  }
  return 0;
}

int batchnorm1d(float* output_signal, float* input_signal,
  unsigned in_time, unsigned in_channels,
  const float* const mean, const float* const var,
  unsigned affine, const float* const gamma , const float* const beta,
  unsigned in_place, float eps) {
  // Check if affine values was learnt
  if (affine) {
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

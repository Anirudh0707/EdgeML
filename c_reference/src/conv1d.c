// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include<stdlib.h>
#include<math.h>
#include "conv1d.h"
#include "utils.h"

int conv1d_lr(float *output_signal, unsigned out_time, unsigned out_channels, const float *input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, int activations) {

  const ConvLayers_LR_Params* tparams= (ConvLayers_LR_Params*) params;
  
  float* tempW = (float*)malloc(out_channels * in_channels * kernel_size * sizeof(float));
  matmul(tparams->W1, tparams->W2, out_channels, tparams->rank,  in_channels * kernel_size, 0, 1.0, tempW);
  // Perform the Convolution
  for (int t = 0; t < out_time; t++) {
    for (int co = 0; co < out_channels; co++) {
      float sum = 0;
      for (int tf = 0; tf < kernel_size; tf++) {
        for (int ci = 0; ci < in_channels; ci++) {
          if (((t + tf) < padding) || ((t + tf) >= (in_time + padding)))
            continue;
          else
            sum += (input_signal[((tf + t) - padding) * in_channels + ci] * tempW[co * in_channels * kernel_size + ci * kernel_size + tf]);
        }
      }
      // Post-Conv activations. More activations can be added should the necessity arise
      if (activations == 1)
        output_signal[t * out_channels + co] = sigmoid(sum + tparams->B[co]);
      else if (activations == 2)
        output_signal[t * out_channels + co] = tanh(sum + tparams->B[co]);
      else if (activations == 3)
        output_signal[t * out_channels + co] = relu(sum + tparams->B[co]);
      else
        output_signal[t * out_channels + co] = sum + tparams->B[co];
    }
  }
  free(tempW);
  return 0;
}

int conv1d_depth_lr(float *output_signal, unsigned out_time, const float *input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, int activations) {

  const ConvLayers_LR_Params* tparams= (ConvLayers_LR_Params*) params;

  float* tempW = (float*)malloc(in_channels * kernel_size * sizeof(float));
  matmul(tparams->W1, tparams->W2, in_channels, tparams->rank,  kernel_size, 0, 1.0, tempW);
  // Perform the Convolution
  for (int t = 0; t < out_time; t++) {
    for (int ci = 0; ci < in_channels; ci++) {
      float sum = 0;
      for (int tf = 0; tf < kernel_size; tf++) {
        if (((t + tf) < padding) || ((t + tf) >= (in_time + padding)))
          continue;
        else
          sum += (input_signal[((tf + t) - padding) * in_channels + ci] * tempW[ci * kernel_size + tf]);
      }
      // Post-Conv activations. More activations can be added should the necessity arise
      if (activations == 1)
        output_signal[t * in_channels + ci] = sigmoid(sum + tparams->B[ci]);
      else if (activations == 2)
        output_signal[t * in_channels + ci] = tanh(sum + tparams->B[ci]);
      else if (activations == 3)
        output_signal[t * in_channels + ci] = relu(sum + tparams->B[ci]);
      else
        output_signal[t * in_channels + ci] = sum + tparams->B[ci];
    }
  }    
  free(tempW);
  return 0;
}



int conv1d(float *output_signal, unsigned out_time, unsigned out_channels, const float *input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, int activations) {

  const ConvLayers_Params* tparams= (ConvLayers_Params*) params;

  // Perform the Convolution
  for (int t = 0; t < out_time; t++) {
      for (int co = 0; co < out_channels; co++) {
      float sum = 0;
      for (int tf = 0; tf < kernel_size; tf++) {
        for (int ci = 0; ci < in_channels; ci++) {
          if (((t + tf) < padding) || ((t + tf) >= (in_time + padding)))
            continue;
          else
            sum += (input_signal[((tf + t) - padding) * in_channels + ci] * tparams->W[co * in_channels * kernel_size + ci * kernel_size + tf]);
        }
      }
      // Post-Conv activations. More activations can be added should the necessity arise
      if (activations == 1)
        output_signal[t * out_channels + co] = sigmoid(sum + tparams->B[co]);
      else if (activations == 2)
        output_signal[t * out_channels + co] = tanh(sum + tparams->B[co]);
      else if (activations == 3)
        output_signal[t * out_channels + co] = relu(sum + tparams->B[co]);
      else
        output_signal[t * out_channels + co] = sum + tparams->B[co];
    }
  }    
  return 0;
}

int conv1d_depth(float *output_signal, unsigned out_time, const float *input_signal,
  unsigned in_time, unsigned in_channels, unsigned padding, unsigned kernel_size,
  const void* params, int activations) {

  const ConvLayers_Params* tparams= (ConvLayers_Params*) params;

  // Perform the Convolution
  for (int t = 0; t < out_time; t++) {
    for (int ci = 0; ci < in_channels; ci++) {
      float sum = 0;
      for (int tf = 0; tf < kernel_size; tf++) {
        if (((t + tf) < padding) || ((t + tf) >= (in_time + padding)))
          continue;
        else
          sum += (input_signal[((tf + t) - padding) * in_channels + ci] * tparams->W[ci * kernel_size + tf]);
      }
      // Post-Conv activations. More activations can be added should the necessity arise
      if (activations == 1)
        output_signal[t * in_channels + ci] = sigmoid(sum + tparams->B[ci]);
      else if (activations == 2)
        output_signal[t * in_channels + ci] = tanh(sum + tparams->B[ci]);
      else if (activations == 3)
        output_signal[t * in_channels + ci] = relu(sum + tparams->B[ci]);
      else
        output_signal[t * in_channels + ci] = sum + tparams->B[ci];
    }
  }
  return 0;
}

int avgpool1d(float *output_signal, unsigned out_time, const float *input_signal, unsigned in_time, unsigned in_channels,
  unsigned padding, unsigned kernel_size, int activations) {

  // Iterate over the time steps and average them. Similar to Conv1D_Dept with a filter kernel of ones
  for (int t = 0; t < out_time; t++) {
    for (int ci = 0; ci < in_channels; ci++) {
      float sum = 0;
      for (int tf = 0; tf < kernel_size; tf++) {
        if (((t + tf) < padding) || ((t + tf) >= (in_time + padding)))
          continue;
        else
          sum += (input_signal[((tf + t) - padding) * in_channels + ci]);
      }
      if (activations == 1)
        output_signal[t * in_channels + ci] = sigmoid(sum/(float)kernel_size);
      else if (activations == 2)
        output_signal[t * in_channels + ci] = tanh(sum/(float)kernel_size);
      else if (activations == 3)
        output_signal[t * in_channels + ci] = relu(sum/(float)kernel_size);
      else
        output_signal[t * in_channels + ci] = sum/(float)kernel_size;
    }
  }
  return 0;
}

int batchnorm1d(float* output_signal, float* input_signal, unsigned in_time, unsigned in_channels,
  float* mean, float* var, unsigned affine, float* gamma , float * beta, unsigned in_place, float eps) {
  // Check if affine values are learnt
  if (affine) {
    // Check for in-place computation
    if (in_place) {
      for (int t = 0; t < in_time; t++) {
        for (int d = 0; d < in_channels; d++) {
          input_signal[t * in_channels + d]  = gamma[d]*((input_signal[t * in_channels + d] - mean[d])/sqrt(var[d] + eps)) + beta[d];
        }
      }
    }
    else {
      for (int t = 0; t < in_time; t++) {
        for (int d = 0; d < in_channels; d++) {
          output_signal[t * in_channels + d] = gamma[d]*((input_signal[t * in_channels + d] - mean[d])/sqrt(var[d] + eps)) + beta[d];
        }
      }
    }
  }
  else {
      // Check for in-place computation
    if (in_place) {
      for (int t = 0; t < in_time; t++) {
        for (int d = 0; d < in_channels; d++) {
          input_signal[t * in_channels + d]  = ((input_signal[t * in_channels + d] - mean[d])/sqrt(var[d] + eps));
        }
      }
    }
    else {
      for (int t = 0; t < in_time; t++) {
        for (int d = 0; d < in_channels; d++) {
          output_signal[t * in_channels + d] = ((input_signal[t * in_channels + d] - mean[d])/sqrt(var[d] + eps));
        }
      }
    }
  }
  return 0;
}

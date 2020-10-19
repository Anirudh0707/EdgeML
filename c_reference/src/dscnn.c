// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "dscnn.h"
#include "conv1d.h"
#include "utils.h"

int phon_pred_lr_cnn(float* output_signal, float* input_signal, unsigned in_time, unsigned in_channels,
  float* mean, float* var, unsigned affine, float* gamma, float* beta, unsigned in_place,
  unsigned cnn_hidden, unsigned cnn_padding, unsigned cnn_kernel_size,
  const void* cnn_params, int cnn_activations) {
  
  unsigned out_time = in_time - cnn_kernel_size + 2*cnn_padding + 1;
  if (in_place) {
    // BatchNorm
    batchnorm1d(0, input_signal, in_time, in_channels, 
      mean, var, affine, gamma, beta, in_place, 0.00001);
    // CNN
    conv1d_lr(output_signal, out_time, cnn_hidden, input_signal, 
      in_time, in_channels, cnn_padding, cnn_kernel_size, 
      cnn_params, cnn_activations);
  }
  else {
    // BatchNorm
    float* norm_out = (float*)malloc(in_time * in_channels * sizeof(float));
    batchnorm1d(norm_out, input_signal, in_time, in_channels, 
      mean, var, affine, gamma, beta, in_place, 0.00001);
    // CNN
    conv1d_lr(output_signal, out_time, cnn_hidden, norm_out, 
      in_time, in_channels, cnn_padding, cnn_kernel_size, 
      cnn_params, cnn_activations);
    free(norm_out);
  }
  return 0;
}

int phon_pred_depth_point_lr_cnn(float* output_signal, float* input_signal, unsigned in_time, unsigned in_channels,
  float* mean, float* var, unsigned affine, float* gamma, float* beta, unsigned in_place,
  unsigned depth_cnn_hidden, unsigned depth_cnn_padding, unsigned depth_cnn_kernel_size,
  const void* depth_cnn_params, int depth_cnn_activations,
  unsigned point_cnn_hidden, unsigned point_cnn_padding, unsigned point_cnn_kernel_size,
  const void* point_cnn_params, int point_cnn_activations,
  unsigned pool_padding, unsigned pool_kernel_size, int pool_activation) {
  
  // Activation
  unsigned out_time;
  float* act_out= (float*)malloc(in_time * (in_channels>>1) * sizeof(float));
  semi_sigmoid_tanh(act_out, input_signal, in_time, in_channels);

  in_channels >>= 1;
  float* depth_out;
  if (in_place) {
    // Norm
    batchnorm1d(0, act_out, in_time, in_channels, 
      mean, var, affine, gamma, beta, in_place, 0.00001);
    // Depth CNN
    out_time = in_time - depth_cnn_kernel_size + 2*depth_cnn_padding + 1;
    depth_out = (float*)malloc(out_time * depth_cnn_hidden * sizeof(float));
    conv1d_depth(depth_out, out_time, act_out, 
      in_time, in_channels, depth_cnn_padding, depth_cnn_kernel_size, 
      depth_cnn_params, depth_cnn_activations);
    free(act_out);
  }
  else {
    // Norm
    float* norm_out = (float*)malloc(in_time * in_channels * sizeof(float));
    batchnorm1d(norm_out, act_out, in_time, in_channels, 
      mean, var, affine, gamma, beta, in_place, 0.00001);
    free(act_out);
    // Depth CNN
    out_time = in_time - depth_cnn_kernel_size + 2*depth_cnn_padding + 1;
    depth_out = (float*)malloc(out_time * depth_cnn_hidden * sizeof(float));
    conv1d_depth(depth_out, out_time, norm_out, 
      in_time, in_channels, depth_cnn_padding, depth_cnn_kernel_size, 
      depth_cnn_params, depth_cnn_activations);
    free(norm_out);
  }  

  // Point CNN
  in_time = out_time;
  out_time = in_time - point_cnn_kernel_size + 2*point_cnn_padding + 1;
  float* point_out = (float*)malloc(out_time * point_cnn_hidden * sizeof(float));
  conv1d_lr(point_out, out_time, point_cnn_hidden, depth_out, 
    in_time, depth_cnn_hidden, point_cnn_padding, point_cnn_kernel_size, 
    point_cnn_params, point_cnn_activations);
  free(depth_out);
  
  // Pool
  in_time = out_time;
  out_time = in_time - pool_kernel_size + 2*pool_padding + 1;
  avgpool1d(output_signal, out_time, point_out, in_time, point_cnn_hidden, 
    pool_padding, pool_kernel_size, pool_activation);
  free(point_out);
  return 0;
}

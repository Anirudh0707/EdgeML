// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include<stdio.h>
#include<stdlib.h>
#include"conv_param_depth.h"
#include"conv1d.h"
#include"conv_utils.h"

int main() {
  ConvLayers_Params conv_params = {
    .W = CONV_WEIGHT,
    .B = CONV_BIAS,
  };

  float pred[O_T * I_F] = {};

  conv1d_depth(pred, O_T, INPUT, I_T, I_F, PAD, FILT, &conv_params, ACT);
  
  float error = 0;
  float denom = 0;
  for (int t = 0; t < O_T; t++) {
    for (int d = 0; d < O_F; d++) {
      error += ((pred[t * O_F + d] - OUTPUT[t * O_F + d]) * (pred[t * O_F + d] - OUTPUT[t * O_F + d]));
      denom += OUTPUT[t * O_F + d] * OUTPUT[t * O_F + d];
    }
  }
  float avg_error = error/(O_T*I_F);
  printf("Testing Depthwise Convolution\n");
  printf("Squared Error : %f \t ; MSE : %f  \n", error, avg_error);
  printf("Relative Squared Error : %f \n", error/denom);
  return 0;
}

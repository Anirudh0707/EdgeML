// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include<stdio.h>
#include<stdlib.h>
#include"conv_param_lr.h"
#include"conv1d.h"
#include"conv_utils.h"

int main() {
  ConvLayers_LR_Params conv_params = {
    .W1 = CONV_W1,
    .W2 = CONV_W2,
    .B = CONV_BIAS,
    .rank = LR_RANK
  };

  float pred[O_T * O_F] = {};

  conv1d_lr(pred, O_T, O_F, INPUT, I_T, I_F, PAD, FILT, &conv_params, ACT);
  
  float error = 0;
  float denom = 0;
  for (int t = 0; t < O_T; t++) {
    for (int d = 0; d < O_F; d++) {
      error += ((pred[t * O_F + d] - OUTPUT[t * O_F + d]) * (pred[t * O_F + d] - OUTPUT[t * O_F + d]));
      denom += OUTPUT[t * O_F + d] * OUTPUT[t * O_F + d];
    }
  }
  float avg_error = error/(O_T*I_F);
  printf("Testing Low Rank Convolution\n");
  printf("Squared Error : %f \t ; MSE : %f  \n", error, avg_error);
  printf("Relative Squared Error : %f \n", error/denom);
  return 0;
}

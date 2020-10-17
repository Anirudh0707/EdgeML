// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include<stdio.h>
#include<stdlib.h>
#include "dscnn_param_lr_depth_point.h"
#include "conv1d.h"
#include "dscnn.h"

int main() {
    
  ConvLayers_Params depth_param = {
    .W = CNN2_DEPTH_W,
    .B = CNN2_DEPTH_BIAS,
  };

  ConvLayers_LR_Params point_param = {
    .W1 = CNN2_POINT_W1,
    .W2 = CNN2_POINT_W2,
    .B = CNN2_POINT_BIAS,
    .rank = LOW_RANK
  };

  float pred[O_T * O_F] = {};


  phon_pred_depth_point_lr_cnn(pred, INPUT, I_T, I_F,
    BNORM_CNN2_MEAN, BNORM_CNN2_VAR, 0, 0, 0, 1, 
    O_F>>1, 2, DEPTH_FILT, 
    &depth_param, 0, 
    O_F, 0, POINT_FILT,
    &point_param, 0,
    0, POOL_FILT, 0);

  // Calculate Error(Aggregate Squared, Mean Squared and Relative Squared)
  float error = 0;
  float denom = 0;
  for (int t = 0; t < O_T; t++) {
    for (int d = 0; d < O_F; d++) {
      error += ((pred[t * O_F + d] - OUTPUT[t * O_F + d]) * (pred[t * O_F + d] - OUTPUT[t * O_F + d]));
      denom += (OUTPUT[t * O_F + d] * OUTPUT[t * O_F + d]);
    }
  }
  float avg_error = error/(O_T*O_F);
  printf("DSCNN Better Block\n");
  printf("Aggregate Squared Error : %f   ;   Mean Sqaured Error : %f  \n", error, avg_error);
  printf("RMS : %f \n", error/denom);

  return 0 ;
}

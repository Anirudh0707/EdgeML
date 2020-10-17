// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include<stdio.h>
#include<stdlib.h>
#include"rnn_bricked.h"
#include"fastgrnn.h"
#include"utils.h"

#include"rnn_params.h"
#include"rnn_bricked_io.h"

int main() {

  FastGRNN_LR_Params bwd_RNN_params = {
    .mean   = 0,
    .stdDev = 0,
    .W1     = B_W1,
    .W2     = B_W2,
    .wRank  = RNN_LOW_RANK,
    .U1     = B_U1,
    .U2     = B_U2,
    .uRank  = RNN_LOW_RANK,
    .Bg     = B_BIAS_GATE,
    .Bh     = B_BIAS_UPDATE,
    .sigmoid_zeta = sigmoid(B_ZETA),
    .sigmoid_nu   = sigmoid(B_NU)
  };

  FastGRNN_LR_Params fwd_RNN_params = {
    .mean   = 0,
    .stdDev = 0,
    .W1     = F_W1,
    .W2     = F_W2,
    .wRank  = RNN_LOW_RANK,
    .U1     = F_U1,
    .U2     = F_U2,
    .uRank  = RNN_LOW_RANK,
    .Bg     = F_BIAS_GATE,
    .Bh     = F_BIAS_UPDATE,
    .sigmoid_zeta = sigmoid(F_ZETA),
    .sigmoid_nu   = sigmoid(F_NU)
  };

  float preComp[RNN_I_F] = { 0.0 };
  float tempLRW[RNN_LOW_RANK] = { 0.0 };
  float tempLRU[RNN_LOW_RANK] = { 0.0 };
  float normFeatures[RNN_I_F] = { 0.0 };
  FastGRNN_LR_Buffers buffers = {
    .preComp = preComp,
    .tempLRW = tempLRW,
    .tempLRU = tempLRU,
    .normFeatures = normFeatures
  };

  float pred[O_T * O_F] = {};

  forward_bricked_rnn(pred, RNN_O_F>>1, INPUT,
    I_T, RNN_I_F, FWD_WINDOW, HOP,
    fastgrnn_lr, &fwd_RNN_params, &buffers,
    1, 1, 0);

  backward_bricked_rnn(pred + (RNN_O_F>>1), RNN_O_F>>1, INPUT,
    I_T, RNN_I_F, BWD_WINDOW, HOP,
    fastgrnn_lr, &bwd_RNN_params, &buffers,
    1, 1, 0);
  
  float error = 0;
  float denom = 0;
  for (int t = 0 ; t < O_T ; t++) {
    for (int d = 0 ; d < O_F ; d++) {
      error += ((pred[t * O_F + d] - OUTPUT[t * O_F + d]) * (pred[t * O_F + d] - OUTPUT[t * O_F + d]));
      denom += OUTPUT[t * O_F + d] * OUTPUT[t * O_F + d] ;
    }
  }
  float avg_error = error/(O_T*O_F);
  printf("Testing Bricked RNNs Bi-Directional\n");
  printf("Squared Error : %f \t ; MSE : %f  \n", error, avg_error);
  printf("Relative Squared Error : %f \n", error/denom);
  return 0 ;
}

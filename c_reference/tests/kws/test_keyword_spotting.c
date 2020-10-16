// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include"conv1d.h"
#include"dscnn.h"
#include"fastgrnn.h"
#include"utils.h"

#include"keyword_spotting_io_2.h"
#include"precnn_params.h"
#include"rnn_params.h"
#include"postcnn_params.h"

// Check out time steps with label time steps
int checkTime(unsigned out_T) {
  if (out_T != O_T) {
    printf("Error, Estimated Output and Actual ouput time axis mis-match");
    return 1;
  }
  return 0;
}
// Error Check
void checkError(float* pred, float* label) {
  float error = 0, denom = 0;
  for (int t = 0; t < O_T; t++) {
    for (int d = 0; d < POST_CNN_O_F; d++) {
      error += ((pred[t * POST_CNN_O_F + d]-label[t * POST_CNN_O_F + d])*(pred[t * POST_CNN_O_F + d]-label[t * POST_CNN_O_F + d]));
      denom += label[t * POST_CNN_O_F + d] * label[t * POST_CNN_O_F + d];
    }
  }
  float avg_error = error/(O_T*POST_CNN_O_F);
  printf("Full Network\n");
  printf("Aggregate Squared Error : %f   ;   Mean Sqaured Error : %f  \n", error, avg_error);
  printf("Denominator for RMSE : %f \n", denom);
  printf("RMSE : %f \n", error/denom);
}
void key_word_spotting(float* mem_buf) {
  ConvLayers_LR_Params conv_params = {
    .W1 = CNN1_W1,
    .W2 = CNN1_W2,
    .B = CNN1_BIAS,
    .rank = PRE_CNN_LOW_RANK
  };

  ConvLayers_Params depth_param_2 = {
    .W = CNN2_DEPTH_W,
    .B = CNN2_DEPTH_BIAS,
  };

  ConvLayers_LR_Params point_param_2 = {
    .W1 = CNN2_POINT_W1,
    .W2 = CNN2_POINT_W2,
    .B = CNN2_POINT_BIAS,
    .rank = POST_CNN_LOW_RANK
  };

  ConvLayers_Params depth_param_3 = {
    .W = CNN3_DEPTH_W,
    .B = CNN3_DEPTH_BIAS,
  };

  ConvLayers_LR_Params point_param_3 = {
    .W1 = CNN3_POINT_W1,
    .W2 = CNN3_POINT_W2,
    .B = CNN3_POINT_BIAS,
    .rank = POST_CNN_LOW_RANK
  };

  ConvLayers_Params depth_param_4 = {
    .W = CNN4_DEPTH_W,
    .B = CNN4_DEPTH_BIAS,
  };

  ConvLayers_LR_Params point_param_4 = {
    .W1 = CNN4_POINT_W1,
    .W2 = CNN4_POINT_W2,
    .B = CNN4_POINT_BIAS,
    .rank = POST_CNN_LOW_RANK
  };

  ConvLayers_Params depth_param_5 = {
    .W = CNN5_DEPTH_W,
    .B = CNN5_DEPTH_BIAS,
  };

  ConvLayers_LR_Params point_param_5 = {
    .W1 = CNN5_POINT_W1,
    .W2 = CNN5_POINT_W2,
    .B = CNN5_POINT_BIAS,
    .rank = POST_CNN_LOW_RANK
  };

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
  
  unsigned in_T, out_T;

  /* Pre-CNN */
  in_T = I_T;
  out_T = in_T - PRE_CNN_FILT + (PRE_CNN_FILT_PAD<<1) + 1; // Depth pad = 2 for kernel size =5. SAME PAD
  float* cnn1_out = (float*)malloc(out_T * PRE_CNN_O_F * sizeof(float));
  // Since bnorm is the first layer and inplce will alter input. Use only if input can be discarded/altered. Else avoid inplace
  dscnn_lr(cnn1_out, mem_buf, in_T, PRE_CNN_I_F,
    BNORM_CNN1_MEAN, BNORM_CNN1_VAR, 0, 0, 0, PRE_CNN_BNORM_INPLACE,
    PRE_CNN_O_F, PRE_CNN_FILT_PAD, PRE_CNN_FILT,
    &conv_params, PRE_CNN_FILT_ACT); // regular tanh activation

  batchnorm1d(0, cnn1_out, in_T, RNN_I_F, 
    BNORM_RNN_MEAN, BNORM_RNN_VAR, 0, 0, 0, 1, 0.00001); // Currently In-place only

  /* Bricked Bi-FastGRNN Block */
  int rnn_hidden = RNN_O_F>>1, out_index = 0;

  out_T = in_T/HOP + 1;
  float* temp_hiddenstate = (float*)calloc(rnn_hidden, sizeof(float));
  float* rnn_out = (float*)malloc(out_T * RNN_O_F * sizeof(float));
  // Forward Pass
  for (int t = 0; t < FWD_WINDOW; t++) {
    fastgrnn_lr(temp_hiddenstate, rnn_hidden,
      cnn1_out + (t * RNN_I_F) , RNN_I_F, 1,
      &fwd_RNN_params, &buffers, 0, 0);
    if (t % HOP==0)
      memcpy(rnn_out + ((out_index++)*RNN_O_F), temp_hiddenstate, rnn_hidden*sizeof(float));
  }
  memcpy(rnn_out + ((out_index++)*RNN_O_F), temp_hiddenstate, rnn_hidden*sizeof(float));
  for (int t = HOP; t <= in_T - FWD_WINDOW; t += HOP ) {
    memset(temp_hiddenstate, 0, rnn_hidden*sizeof(float));
    fastgrnn_lr(temp_hiddenstate, rnn_hidden,
      cnn1_out + (t * RNN_I_F) , RNN_I_F, FWD_WINDOW,
      &fwd_RNN_params, &buffers, 0, 0);
    memcpy(rnn_out + ((out_index++)*RNN_O_F), temp_hiddenstate, rnn_hidden*sizeof(float));    
  }

  // Backward Pass
  out_index = 0;
  for (int t = 0; t < in_T - BWD_WINDOW; t += HOP ) {
    memset(temp_hiddenstate, 0, rnn_hidden*sizeof(float));
    fastgrnn_lr(temp_hiddenstate, rnn_hidden,
      cnn1_out + (t * RNN_I_F) , RNN_I_F, BWD_WINDOW,
      &bwd_RNN_params, &buffers, 1, 0);
    memcpy(rnn_out + ((out_index++)*RNN_O_F + rnn_hidden), temp_hiddenstate, rnn_hidden*sizeof(float));
  }
  out_index += BWD_WINDOW/HOP;
  memset(temp_hiddenstate, 0, rnn_hidden*sizeof(float));
  for (int t = in_T - 1; t >= in_T - BWD_WINDOW; t--) {
    fastgrnn_lr(temp_hiddenstate, rnn_hidden,
      cnn1_out + (t * RNN_I_F) , RNN_I_F, 1,
      &bwd_RNN_params, &buffers, 0, 0);
    if ((in_T - 1 - t) % HOP == 0)
      memcpy(rnn_out + ((out_index--)*RNN_O_F + rnn_hidden), temp_hiddenstate, rnn_hidden*sizeof(float));
  }
  memcpy(rnn_out + ((out_index)*RNN_O_F + rnn_hidden), temp_hiddenstate, rnn_hidden*sizeof(float));
  free(temp_hiddenstate);

  free(cnn1_out);

  /* Post-CNN */
  // Since all inputs to the subsequent layers are temporary, in-place bnorm can be used without any input/output data alteration
  // CNN2
  in_T = out_T;
  out_T = in_T - POST_CNN_DEPTH_FILT + (POST_CNN_DEPTH_PAD<<1) + 1;
  out_T = out_T - POST_CNN_POOL + (POST_CNN_POOL_PAD<<1) + 1;
  float* cnn2_out = (float*)malloc(out_T * POST_CNN_INTER_F * sizeof(float));
  dscnn_depth_point_lr(cnn2_out, rnn_out, in_T, POST_CNN_INTER_F,
    CNN2_BNORM_MEAN, CNN2_BNORM_VAR, 0, 0, 0, POST_CNN_BNORM_INPLACE,
    POST_CNN_INTER_F>>1, POST_CNN_DEPTH_PAD, POST_CNN_DEPTH_FILT,
    &depth_param_2, POST_CNN_DEPTH_ACT,
    POST_CNN_INTER_F, POST_CNN_POINT_PAD, POST_CNN_POINT_FILT,
    &point_param_2, POST_CNN_POINT_ACT,
    POST_CNN_POOL_PAD, POST_CNN_POOL, POST_CNN_POOL_ACT);
  free(rnn_out);

  // CNN3
  in_T = out_T;
  out_T = in_T - POST_CNN_DEPTH_FILT + (POST_CNN_DEPTH_PAD<<1) + 1;
  out_T = out_T - POST_CNN_POOL + (POST_CNN_POOL_PAD<<1) + 1;
  float* cnn3_out = (float*)malloc(out_T * POST_CNN_INTER_F * sizeof(float));
  dscnn_depth_point_lr(cnn3_out, cnn2_out, in_T, POST_CNN_INTER_F,
    CNN3_BNORM_MEAN, CNN3_BNORM_VAR, 0, 0, 0, POST_CNN_BNORM_INPLACE,
    POST_CNN_INTER_F>>1, POST_CNN_DEPTH_PAD, POST_CNN_DEPTH_FILT,
    &depth_param_3, POST_CNN_DEPTH_ACT,
    POST_CNN_INTER_F, POST_CNN_POINT_PAD, POST_CNN_POINT_FILT,
    &point_param_3, POST_CNN_POINT_ACT,
    POST_CNN_POOL_PAD, POST_CNN_POOL, POST_CNN_POOL_ACT);
  free(cnn2_out);

  // CNN4
  in_T = out_T;
  out_T = in_T - POST_CNN_DEPTH_FILT + (POST_CNN_DEPTH_PAD<<1) + 1;
  out_T = out_T - POST_CNN_POOL + (POST_CNN_POOL_PAD<<1) + 1;
  float* cnn4_out = (float*)malloc(out_T * POST_CNN_INTER_F * sizeof(float));
  dscnn_depth_point_lr(cnn4_out, cnn3_out, in_T, POST_CNN_INTER_F,
    CNN4_BNORM_MEAN, CNN4_BNORM_VAR, 0, 0, 0, POST_CNN_BNORM_INPLACE,
    POST_CNN_INTER_F>>1, POST_CNN_DEPTH_PAD, POST_CNN_DEPTH_FILT,
    &depth_param_4, POST_CNN_DEPTH_ACT,
    POST_CNN_INTER_F, POST_CNN_POINT_PAD, POST_CNN_POINT_FILT,
    &point_param_4, POST_CNN_POINT_ACT,
    POST_CNN_POOL_PAD, POST_CNN_POOL, POST_CNN_POOL_ACT);
  free(cnn3_out);

  // CNN5
  in_T = out_T;
  out_T = in_T - POST_CNN_DEPTH_FILT + (POST_CNN_DEPTH_PAD<<1) + 1;
  out_T = out_T - POST_CNN_POOL + (POST_CNN_POOL_PAD<<1) + 1;
  float* pred = (float*)malloc(out_T * POST_CNN_O_F * sizeof(float));
  dscnn_depth_point_lr(pred, cnn4_out, in_T, POST_CNN_INTER_F,
    CNN5_BNORM_MEAN, CNN5_BNORM_VAR, 0, 0, 0, POST_CNN_BNORM_INPLACE,
    POST_CNN_INTER_F>>1, POST_CNN_DEPTH_PAD, POST_CNN_DEPTH_FILT,
    &depth_param_5, POST_CNN_DEPTH_ACT,
    POST_CNN_O_F, POST_CNN_POINT_PAD, POST_CNN_POINT_FILT,
    &point_param_5, POST_CNN_POINT_ACT,
    POST_CNN_POOL_PAD, POST_CNN_POOL, POST_CNN_POOL_ACT);
  free(cnn4_out);

  /* Output Time and Prediction Check. Created for Deugging */
  if (checkTime(out_T))
    return;
  else
    checkError(pred, OUTPUT);
  free(pred);
}

int main() {
  clock_t begin = clock();
  key_word_spotting(INPUT);
  clock_t end = clock();
  double time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
  printf("Time elapsed is %f seconds\n", time_spent);
  return 0;
}

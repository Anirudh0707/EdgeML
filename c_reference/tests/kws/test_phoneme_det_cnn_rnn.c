// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include "conv1d.h"
#include "dscnn.h"
#include "fastgrnn.h"
#include "utils.h"
#include "rnn_bricked.h"

#include "keyword_spotting_io_2.h"
#include "precnn_params.h"
#include "rnn_params.h"
#include "postcnn_params.h"

// Check out time steps with label time steps
int checkTime(unsigned out_time) {
  if (out_time != O_T) {
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
  printf("RMSE : %f \n", error/denom);
}

/* CNN-RNN based Phoneme Detection Model
 
  The phoneme detection model being used consists of 5 blocks.
  1st block is a CNN, where krenel size is 5.
  2nd block is an RNN, which has a specified forward and a backward context running at a stride/hop of 3.
  Hence it reduces the sequence length by a factor of 3.
  Rest of the blocks are a combination of CNNs, a depth cnn with a kernel size of 5 and a point cnn with a kernel size of 1

  Input to the architecture is of the form (seq_len, feature_dim) where feature dim refers to n_mels .
  Output is of the form (seq_len/3, 41) where 41 is the number of phonemes over which classification is done. 
  Phonemes are predicted for every 3rd time frame, assuming they dont vary faster then that.

  NOTE: Before deployment for real-time streaming applications, we would need to make minor modification
  These changes are subject to the input specs i.e fixing buffer time steps, number of features from the featurizer, method of reading into a buffer
*/
void phoneme_prediction(float* mem_buf) {
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
  
  unsigned in_time, out_time;

  /* Pre-CNN */
  in_time = I_T;
  out_time = in_time - PRE_CNN_FILT + (PRE_CNN_FILT_PAD<<1) + 1; // Depth pad = 2 for kernel size =5. SAME PAD
  float* cnn1_out = (float*)malloc(out_time * PRE_CNN_O_F * sizeof(float));
  // Since bnorm is the first layer and in-place will alter input. Use only if input can be discarded/altered. Else avoid inplace
  phon_pred_lr_cnn(cnn1_out, mem_buf, in_time, PRE_CNN_I_F,
    BNORM_CNN1_MEAN, BNORM_CNN1_VAR, 0, 0, 0, PRE_CNN_BNORM_INPLACE,
    PRE_CNN_O_F, PRE_CNN_FILT_PAD, PRE_CNN_FILT,
    &conv_params, PRE_CNN_FILT_ACT); // regular tanh activation

  batchnorm1d(0, cnn1_out, in_time, RNN_I_F, 
    BNORM_RNN_MEAN, BNORM_RNN_VAR, 0, 0, 0, 1, 0.00001); // Currently In-place only

  /* Bricked Bi-FastGRNN Block */

  out_time = in_time/HOP + 1;
  float* rnn_out = (float*)malloc(out_time * RNN_O_F * sizeof(float));
  forward_bricked_rnn(rnn_out, RNN_O_F>>1, cnn1_out,
    in_time, RNN_I_F, FWD_WINDOW, HOP,
    fastgrnn_lr, &fwd_RNN_params, &buffers,
    BI_DIR, SAMPLE_FIRST_BRICK, 0);

  backward_bricked_rnn(rnn_out + (RNN_O_F>>1), RNN_O_F>>1, cnn1_out,
    in_time, RNN_I_F, BWD_WINDOW, HOP,
    fastgrnn_lr, &bwd_RNN_params, &buffers,
    BI_DIR, SAMPLE_LAST_BRICK, 0);
  free(cnn1_out);

  /* Post-CNN */
  // Since all inputs to the subsequent layers are temporary, in-place bnorm can be used without any input/output data alteration
  // CNN2
  in_time = out_time;
  out_time = in_time - POST_CNN_DEPTH_FILT + (POST_CNN_DEPTH_PAD<<1) + 1;
  out_time = out_time - POST_CNN_POOL + (POST_CNN_POOL_PAD<<1) + 1;
  float* cnn2_out = (float*)malloc(out_time * POST_CNN_INTER_F * sizeof(float));
  phon_pred_depth_point_lr_cnn(cnn2_out, rnn_out, in_time, POST_CNN_INTER_F,
    CNN2_BNORM_MEAN, CNN2_BNORM_VAR, 0, 0, 0, POST_CNN_BNORM_INPLACE,
    POST_CNN_INTER_F>>1, POST_CNN_DEPTH_PAD, POST_CNN_DEPTH_FILT,
    &depth_param_2, POST_CNN_DEPTH_ACT,
    POST_CNN_INTER_F, POST_CNN_POINT_PAD, POST_CNN_POINT_FILT,
    &point_param_2, POST_CNN_POINT_ACT,
    POST_CNN_POOL_PAD, POST_CNN_POOL, POST_CNN_POOL_ACT);
  free(rnn_out);

  // CNN3
  in_time = out_time;
  out_time = in_time - POST_CNN_DEPTH_FILT + (POST_CNN_DEPTH_PAD<<1) + 1;
  out_time = out_time - POST_CNN_POOL + (POST_CNN_POOL_PAD<<1) + 1;
  float* cnn3_out = (float*)malloc(out_time * POST_CNN_INTER_F * sizeof(float));
  phon_pred_depth_point_lr_cnn(cnn3_out, cnn2_out, in_time, POST_CNN_INTER_F,
    CNN3_BNORM_MEAN, CNN3_BNORM_VAR, 0, 0, 0, POST_CNN_BNORM_INPLACE,
    POST_CNN_INTER_F>>1, POST_CNN_DEPTH_PAD, POST_CNN_DEPTH_FILT,
    &depth_param_3, POST_CNN_DEPTH_ACT,
    POST_CNN_INTER_F, POST_CNN_POINT_PAD, POST_CNN_POINT_FILT,
    &point_param_3, POST_CNN_POINT_ACT,
    POST_CNN_POOL_PAD, POST_CNN_POOL, POST_CNN_POOL_ACT);
  free(cnn2_out);

  // CNN4
  in_time = out_time;
  out_time = in_time - POST_CNN_DEPTH_FILT + (POST_CNN_DEPTH_PAD<<1) + 1;
  out_time = out_time - POST_CNN_POOL + (POST_CNN_POOL_PAD<<1) + 1;
  float* cnn4_out = (float*)malloc(out_time * POST_CNN_INTER_F * sizeof(float));
  phon_pred_depth_point_lr_cnn(cnn4_out, cnn3_out, in_time, POST_CNN_INTER_F,
    CNN4_BNORM_MEAN, CNN4_BNORM_VAR, 0, 0, 0, POST_CNN_BNORM_INPLACE,
    POST_CNN_INTER_F>>1, POST_CNN_DEPTH_PAD, POST_CNN_DEPTH_FILT,
    &depth_param_4, POST_CNN_DEPTH_ACT,
    POST_CNN_INTER_F, POST_CNN_POINT_PAD, POST_CNN_POINT_FILT,
    &point_param_4, POST_CNN_POINT_ACT,
    POST_CNN_POOL_PAD, POST_CNN_POOL, POST_CNN_POOL_ACT);
  free(cnn3_out);

  // CNN5
  in_time = out_time;
  out_time = in_time - POST_CNN_DEPTH_FILT + (POST_CNN_DEPTH_PAD<<1) + 1;
  out_time = out_time - POST_CNN_POOL + (POST_CNN_POOL_PAD<<1) + 1;
  float* pred = (float*)malloc(out_time * POST_CNN_O_F * sizeof(float));
  phon_pred_depth_point_lr_cnn(pred, cnn4_out, in_time, POST_CNN_INTER_F,
    CNN5_BNORM_MEAN, CNN5_BNORM_VAR, 0, 0, 0, POST_CNN_BNORM_INPLACE,
    POST_CNN_INTER_F>>1, POST_CNN_DEPTH_PAD, POST_CNN_DEPTH_FILT,
    &depth_param_5, POST_CNN_DEPTH_ACT,
    POST_CNN_O_F, POST_CNN_POINT_PAD, POST_CNN_POINT_FILT,
    &point_param_5, POST_CNN_POINT_ACT,
    POST_CNN_POOL_PAD, POST_CNN_POOL, POST_CNN_POOL_ACT);
  free(cnn4_out);

  /* Output Time and Prediction Check. Created for Deugging */
  if (checkTime(out_time))
    return;
  else
    checkError(pred, OUTPUT);
  free(pred);
}

int main() {
  clock_t begin = clock();
  phoneme_prediction(INPUT);
  clock_t end = clock();
  double time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
  printf("Time elapsed is %f seconds\n", time_spent);
  return 0;
}

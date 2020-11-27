// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rnn_bricked.h"
#include "fastgrnn.h"
#include "utils.h"

// Forward Pass
int forward_bricked_fastgrnn_lr(float* output_signal, unsigned rnn_hidden, 
  float* input_signal, unsigned in_time, unsigned in_dims, 
  unsigned window, unsigned hop, const void* params,
  unsigned bi_direction, unsigned sample_first_brick) {
  
  // Buffers and params
  const FastGRNN_LR_Params* tparams = (const FastGRNN_LR_Params*)params;

  unsigned rnn_assign_offset = rnn_hidden, out_index = 0;
  unsigned num_bricks = (in_time - window) / hop + 1;
  // If bi-directional is True(non-zero) then the actual output hidden state(allocated space) is twice rnn_hidden
  // This function only processes the forward context
  if (bi_direction) {
    rnn_assign_offset <<= 1;
  }
  
  // Compute W1 * W2 * X
  float* inputMulW = (float*)calloc(in_time * rnn_hidden, sizeof(float));
  float* tempLR = (float*)calloc(in_time * tparams->wRank, sizeof(float));
  float* hiddenState = (float*)calloc(num_bricks * rnn_hidden, sizeof(float));
  float* preComp = (float*)calloc(num_bricks * rnn_hidden, sizeof(float));
  transposed_tiledMatMul(input_signal, tparams->W1, in_time, in_dims,
    tparams->wRank, in_dims, in_dims, tempLR, 100);
  transposed_tiledMatMul(tempLR, tparams->W2, in_time, tparams->wRank,
    rnn_hidden, tparams->wRank, tparams->wRank, inputMulW, 100);
  free(tempLR);
  // We can reuse the low-rank buffer from Wx to Uh, since Wx is computed at one stretch
  tempLR = (float*)calloc(num_bricks * tparams->uRank, sizeof(float));
  for (unsigned t = 0; t < window; t++) {
    // From higher dims to lower dims
    memset(tempLR, 0, num_bricks * tparams->uRank * sizeof(float));
    transposed_tiledMatMul(hiddenState, tparams->U1, num_bricks, rnn_hidden,
      tparams->uRank, rnn_hidden, rnn_hidden, tempLR, 100);
    // From lower dims to higher dims
    memset(preComp, 0, num_bricks * rnn_hidden * sizeof(float));
    transposed_tiledMatMul(tempLR, tparams->U2, num_bricks, tparams->uRank,
      rnn_hidden, tparams->uRank, tparams->uRank, preComp, 100);
    // Add
    for (unsigned n = 0; n < num_bricks; n++) {
      for (unsigned d = 0; d < rnn_hidden; d++) {
        preComp[n * rnn_hidden + d] += inputMulW[n * hop * rnn_hidden + t * rnn_hidden + d];
      }
    }
    
    // Apply the gating
    float* hiddenState_offset = hiddenState;
    for (unsigned n = 0; n < num_bricks; n++) {
      float* preComp_offset = preComp + n * rnn_hidden;
      float* gateBias = (float*)tparams->Bg;
      float* hiddenBias = (float*)tparams->Bh;
      unsigned hidden = rnn_hidden;
      
      #ifdef LOOP_UNROLL
        unsigned len_unroll = hidden >> 2;
        hidden = rnn_hidden % 4;
        float gate, update;
        while (len_unroll--) {
        gate = sigmoid((*preComp_offset) + (*gateBias++));
        update = tanh((*preComp_offset++) + (*hiddenBias++));
        *hiddenState_offset = gate * (*hiddenState_offset) + 
                              (tparams->sigmoid_zeta * (1.0 - gate) + 
                              tparams->sigmoid_nu) * update;
        hiddenState_offset++;
        gate = sigmoid((*preComp_offset) + (*gateBias++));
        update = tanh((*preComp_offset++) + (*hiddenBias++));
        *hiddenState_offset = gate * (*hiddenState_offset) + 
                              (tparams->sigmoid_zeta * (1.0 - gate) + 
                              tparams->sigmoid_nu) * update;
        hiddenState_offset++;
        gate = sigmoid((*preComp_offset) + (*gateBias++));
        update = tanh((*preComp_offset++) + (*hiddenBias++));
        *hiddenState_offset = gate * (*hiddenState_offset) + 
                              (tparams->sigmoid_zeta * (1.0 - gate) + 
                              tparams->sigmoid_nu) * update;
        hiddenState_offset++;
        gate = sigmoid((*preComp_offset) + (*gateBias++));
        update = tanh((*preComp_offset++) + (*hiddenBias++));
        *hiddenState_offset = gate * (*hiddenState_offset) + 
                              (tparams->sigmoid_zeta * (1.0 - gate) + 
                              tparams->sigmoid_nu) * update;
        hiddenState_offset++;
        }
      #endif

      while (hidden--) {
        float gate = sigmoid((*preComp_offset) + (*gateBias++));
        float update = tanh((*preComp_offset++) + (*hiddenBias++));
        *hiddenState_offset = gate * (*hiddenState_offset) + 
                              (tparams->sigmoid_zeta * (1.0 - gate) + 
                              tparams->sigmoid_nu) * update;
        hiddenState_offset++;
      }
    }
    // Sample first block if necessary
    if (sample_first_brick) {
      if (t % hop == 0) {
        memcpy(output_signal + (out_index++) * rnn_assign_offset,
          hiddenState, rnn_hidden * sizeof(float));
      }
    }
  }
  if (bi_direction) {
    // If bi-directional then a gap would need to be left for the backward outputs
    float* hiddenState_offset = hiddenState;
    for (unsigned n = 0; n < num_bricks; n++) {
      memcpy(output_signal + (out_index++) * rnn_assign_offset,
        hiddenState_offset, rnn_hidden * sizeof(float));
      hiddenState_offset += rnn_hidden;
    }
  }
  else {
    // If only forward is needed, the the whole block of memory can be copied without the loop
    memcpy(output_signal + out_index * rnn_assign_offset,
      hiddenState, num_bricks * rnn_hidden * sizeof(float));
  }
  free(hiddenState);
  free(inputMulW);
  free(preComp);
  free(tempLR);
  return 0;
}

// Backward Pass
int backward_bricked_fastgrnn_lr(float* output_signal, unsigned rnn_hidden, 
  float* input_signal, unsigned in_time, unsigned in_dims, 
  unsigned window, unsigned hop, const void* params,
  unsigned bi_direction, unsigned sample_last_brick) {
  
  // Buffers and params
  const FastGRNN_LR_Params* tparams = (const FastGRNN_LR_Params*)params;

  unsigned rnn_assign_offset = rnn_hidden;
  unsigned num_bricks = (in_time - window) / hop + 1;
  unsigned out_index = in_time / hop; // = out_time - 1;
  // If bi-directional is True(non-zero) then the actual output hidden state(allocated space) is twice rnn_hidden
  // This function only processes the forward context
  if (bi_direction) {
    rnn_assign_offset <<= 1;
  }
  
  // Compute W1 * W2 * X
  float* inputMulW = (float*)calloc(in_time * rnn_hidden, sizeof(float));
  float* tempLR = (float*)calloc(in_time * tparams->wRank, sizeof(float));
  float* hiddenState = (float*)calloc(num_bricks * rnn_hidden, sizeof(float));
  float* preComp = (float*)calloc(num_bricks * rnn_hidden, sizeof(float));
  transposed_tiledMatMul(input_signal, tparams->W1, in_time, in_dims,
    tparams->wRank, in_dims, in_dims, tempLR, 100);
  transposed_tiledMatMul(tempLR, tparams->W2, in_time, tparams->wRank,
    rnn_hidden, tparams->wRank, tparams->wRank, inputMulW, 100);
  free(tempLR);
  // We can reuse the low-rank buffer from Wx to Uh, since Wx is computed at one stretch
  tempLR = (float*)calloc(num_bricks * tparams->uRank, sizeof(float));
  for (int t = window - 1; t >= 0; t--) {
    // From higher dims to lower dims
    memset(tempLR, 0, num_bricks * tparams->uRank * sizeof(float));
    transposed_tiledMatMul(hiddenState, tparams->U1, num_bricks, rnn_hidden,
      tparams->uRank, rnn_hidden, rnn_hidden, tempLR, 100);
    // From lower dims to higher dims
    memset(preComp, 0, num_bricks * rnn_hidden * sizeof(float));
    transposed_tiledMatMul(tempLR, tparams->U2, num_bricks, tparams->uRank,
      rnn_hidden, tparams->uRank, tparams->uRank, preComp, 100);
    // Add
    for (unsigned n = 0; n < num_bricks; n++) {
      for (unsigned d = 0; d < rnn_hidden; d++) {
        preComp[n * rnn_hidden + d] += inputMulW[n * hop * rnn_hidden + t * rnn_hidden + d];
      }
    }
    
    // Apply the gating
    float* hiddenState_offset = hiddenState;
    for (unsigned n = 0; n < num_bricks; n++) {
      float* preComp_offset = preComp + n * rnn_hidden;
      float* gateBias = (float*)tparams->Bg;
      float* hiddenBias = (float*)tparams->Bh;
      unsigned hidden = rnn_hidden;
      
      #ifdef LOOP_UNROLL
        unsigned len_unroll = hidden >> 2;
        hidden = rnn_hidden % 4;
        float gate, update;
        while (len_unroll--) {
          gate = sigmoid((*preComp_offset) + (*gateBias++));
          update = tanh((*preComp_offset++) + (*hiddenBias++));
          *hiddenState_offset = gate * (*hiddenState_offset) + 
                                (tparams->sigmoid_zeta * (1.0 - gate) + 
                                tparams->sigmoid_nu) * update;
          hiddenState_offset++;
          gate = sigmoid((*preComp_offset) + (*gateBias++));
          update = tanh((*preComp_offset++) + (*hiddenBias++));
          *hiddenState_offset = gate * (*hiddenState_offset) + 
                                (tparams->sigmoid_zeta * (1.0 - gate) + 
                                tparams->sigmoid_nu) * update;
          hiddenState_offset++;
          gate = sigmoid((*preComp_offset) + (*gateBias++));
          update = tanh((*preComp_offset++) + (*hiddenBias++));
          *hiddenState_offset = gate * (*hiddenState_offset) + 
                                (tparams->sigmoid_zeta * (1.0 - gate) + 
                                tparams->sigmoid_nu) * update;
          hiddenState_offset++;
          gate = sigmoid((*preComp_offset) + (*gateBias++));
          update = tanh((*preComp_offset++) + (*hiddenBias++));
          *hiddenState_offset = gate * (*hiddenState_offset) + 
                                (tparams->sigmoid_zeta * (1.0 - gate) + 
                                tparams->sigmoid_nu) * update;
          hiddenState_offset++;
        }
      #endif

      while (hidden--) {
        float gate = sigmoid((*preComp_offset) + (*gateBias++));
        float update = tanh((*preComp_offset++) + (*hiddenBias++));
        *hiddenState_offset = gate * (*hiddenState_offset) + 
                              (tparams->sigmoid_zeta * (1.0 - gate) + 
                              tparams->sigmoid_nu) * update;
        hiddenState_offset++;
      }
    }
    // Sample first block if necessary
    if (sample_last_brick) {
      if ((window - 1 - t) % hop == 0) {
        // Iterate over the output in reverse
        memcpy(output_signal + (out_index--) * rnn_assign_offset,
          hiddenState + (num_bricks - 1) * rnn_hidden, rnn_hidden * sizeof(float));
      }
    }
  }
  // Since the all first (final in reverse) hiddenstates are calculated, we assign the whole block
  out_index = 0;
  if (bi_direction) {
    // If bi-directional then a gap would need to be left for the backward outputs
    float* hiddenState_offset = hiddenState;
    for (unsigned n = 0; n < num_bricks; n++) {
      memcpy(output_signal + (out_index++) * rnn_assign_offset,
        hiddenState_offset, rnn_hidden * sizeof(float));
      hiddenState_offset += rnn_hidden;
    }
  }
  else {
    // If only forward is needed, the the whole block of memory can be copied without the loop
    memcpy(output_signal + out_index * rnn_assign_offset,
      hiddenState, num_bricks * rnn_hidden * sizeof(float));
  }
  free(hiddenState);
  free(inputMulW);
  free(preComp);
  free(tempLR);
  return 0;
}

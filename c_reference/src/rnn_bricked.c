// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rnn_bricked.h"
#include "fastgrnn.h"
#include "utils.h"

// Forward Pass
int forward_bricked_rnn(float* output_signal, unsigned rnn_hidden, float* input_signal,
  unsigned in_time, unsigned in_dims, unsigned window, unsigned hop,
  rnn_layer rnn, const void* params, void* buffers,
  unsigned bi_direction, unsigned sample_first_brick, int normalize) {
  unsigned out_index = 0, t; // t is an index, but we want to remember the value after the loop. Hence we define it outside

  unsigned rnn_assign_offset = rnn_hidden;
  float* temp_hiddenstate = (float*)calloc(rnn_hidden, sizeof(float));
  // If bi-directional is True(non-zero) then the actual output hidden state(allocated space) is twice rnn_hidden
  // This function only processes the forward context
  if (bi_direction) {
    rnn_assign_offset <<= 1;
  }
  // for the first window, sample every hop index only if sample_first_block = 1. else only the final hidden state is calculated
  for (t = 0; t < window; t++) {
    rnn(temp_hiddenstate, rnn_hidden,
      input_signal + (t * in_dims) , in_dims, 1,
      params, buffers, 0, normalize);
    if (sample_first_brick) {
      if (t % hop==0) {
        memcpy(output_signal + ((out_index++) * rnn_assign_offset),
          temp_hiddenstate, rnn_hidden * sizeof(float));
      }
    }
  }
  // sample the last hidden state of the first brick
  memcpy(output_signal + ((out_index++) * rnn_assign_offset),
    temp_hiddenstate, rnn_hidden * sizeof(float));
  // sample the last hidden state of all subsequent bricks, except the last
  for (t = hop; t < in_time - window; t += hop ) {
    memset(temp_hiddenstate, 0, rnn_hidden * sizeof(float));
    rnn(temp_hiddenstate, rnn_hidden,
      input_signal + (t * in_dims) , in_dims, window,
      params, buffers, 0, normalize);
    memcpy(output_signal + ((out_index++) * rnn_assign_offset),
      temp_hiddenstate, rnn_hidden * sizeof(float));    
  }
  // Calculated seperately since, the time steps left need not be equal to "window"
  // Hence if the last brick has less than "window" time steps
  // We only pass those values by reducing the forward-window length
  memset(temp_hiddenstate, 0, rnn_hidden * sizeof(float));
  rnn(temp_hiddenstate, rnn_hidden,
    input_signal + (t * in_dims) , in_dims, in_time - t,
    params, buffers, 0, normalize);
  memcpy(output_signal + out_index * rnn_assign_offset,
    temp_hiddenstate, rnn_hidden * sizeof(float));
  free(temp_hiddenstate);
  return 0;
}

int forward_bricked_rnn_parallel(float* output_signal, unsigned rnn_hidden, 
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

int backward_bricked_rnn_parallel(float* output_signal, unsigned rnn_hidden, 
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

// Backward Pass
int backward_bricked_rnn(float* output_signal, unsigned rnn_hidden, float* input_signal,
  unsigned in_time, unsigned in_dims, unsigned window, unsigned hop,
  rnn_layer rnn, const void* params, void* buffers,
  unsigned bi_direction, unsigned sample_last_brick, int normalize) {
  unsigned out_index = 0, t;

  // When bi-direction = 1, an offset of "rnn_hidden" will need to be provided during the function call(to the output_signal). 
  // This is to allocate the results of the backward pass correctly(each of size rnn_hidden, for each time step)
  unsigned rnn_assign_offset = rnn_hidden;
  float* temp_hiddenstate = (float*)calloc(rnn_hidden, sizeof(float));
  // If bi-directional is True(non-zero) then the actual output hidden state(allocated space) is twice rnn_hidden
  // This function only processes the backward context. for this context ouput size = rnn_hidden
  if (bi_direction) {
    rnn_assign_offset <<= 1;
  }
  // sample the last hidden state(in reverse) of all bricks, except the last one
  for (t = 0; t < in_time - window; t += hop ) {
    memset(temp_hiddenstate, 0, rnn_hidden * sizeof(float));
    rnn(temp_hiddenstate, rnn_hidden,
      input_signal + (t * in_dims) , in_dims, window,
      params, buffers, 1, normalize);
    memcpy(output_signal + ((out_index++) * rnn_assign_offset),
      temp_hiddenstate, rnn_hidden * sizeof(float));
  }
  // Necessary offset for output allocation, for the sample_last_brick = 1 case
  if (sample_last_brick)
    out_index += window / hop;
  // If sample_last_block = 1, sample every hop index only for the last window 
  // Else the final hidden state(in reverse) is calculated
  unsigned stop_time = t;
  memset(temp_hiddenstate, 0, rnn_hidden * sizeof(float));
  for (t = in_time - 1; t >= stop_time; t--) {
    rnn(temp_hiddenstate, rnn_hidden,
      input_signal + (t * in_dims) , in_dims, 1,
      params, buffers, 0, normalize); // Since only one time step is passed at a time, the backward flag can be set either way
    if (sample_last_brick) {
      if ((in_time - 1 - t) % hop == 0) {
        memcpy(output_signal + ((out_index--) * rnn_assign_offset),
          temp_hiddenstate, rnn_hidden * sizeof(float));
      }
    }
  }
  // sample the last hidden state(in reverse) of the last brick
  memcpy(output_signal + out_index * rnn_assign_offset,
    temp_hiddenstate, rnn_hidden * sizeof(float));
  free(temp_hiddenstate);
  return 0;
}

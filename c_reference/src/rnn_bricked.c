// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rnn_bricked.h"
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
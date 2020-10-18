// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __RNNBRICKED_H__
#define __RNNBRICKED_H__

// Function Pointer for the RNN to be pssed as a parameter
typedef int (*rnn_t)(float* const, unsigned, const float* const, unsigned, unsigned, const void*, void*, int, int);

// NOTES for bi-direction
// If bi_direction = 1, then actual rnn_output_dims is twice the rnn_hidden. 
// Each function will only process the its given context(forward/backward).
// The other context will need to be called seperately with an appropriate offset.
// E.g : 1st step -> foward(output, ..., input, ..., bi-direction=1, ...)
//       2nd step -> backward(output + rnn_hidden, ..., input, ..., bi-direction=1, ...)
//
// The actual ouput dims will be twice the rnn_hidden(rnn_hidden is output dims for each cell). 
// Hence, each cell will only calculate half the hidden state i.e. rnn_hidden slots of memory from the start of the output pointer
// Hence rnn_hidden is used as an offset for the backward pass. Offset for the forward pass is 0
// This use of offset is a way to exploit the nature of bi-rdirection to bypass concatenation step associated with bi-direction
//
// Constraints
// For Bi-Directional use, there are 2 constarints
// 1) (in_time - window) % hop == 0
// 2) both the window % hop == 0
// 3) sample_first_brick and sample_last_brick = 1
//
// Violation of these constraints can lead to one of the following issues
// 1) segmenation faults 
// 2) forward out_time != backward out_time
// 3) mismatch between forward index and backward index during sampling i.e forward index 8 would correspond to backward index 6. This index error continues for all consecutive bricks
// Hence, padding of the input and appropriate window choice is necessary
//
// The constraints can be ignored for uni-directional passes. However, it is favorable to follow constraints 1 and 2


/** Foward Bricking and application of the forward rnn for an input signal
 * @param[out]       output_signal        pointer to output signal. size = out_time * rnn_hidden
 * @param[in]        rnn_hidden           output dimension for the current cell
 * @param[in]        input_signal         pointer to input signal. size = in_time * in_dims
 * @param[in]        in_time                 number of input time steps.
 * @param[in]        in_dims              input dimensions
 * @param[in]        window               window length for each brick. For the final brick, the left over time steps are used(need not be window in length for the last brick) 
 * @param[in]        hop                  hop distance for between bricks
 * @param[in]        rnn                  function pointer to the rnn
 * @param[in]        params               pointer to the parameters for the rnn
 * @param[in,out]    buffers              pointer to buffer for the rnn
 * @param[in]        bi_direction         determine if the ouput if for a bi-directional rnn. 
 * @param[in]        sample_first_brick   determine if the 1st brick should also be sampled
 *                   -> if = 0, only the last hidden state of each brick is sampled. out_time = (in_time-window)/hop + 1
 *                   -> if = 1, for the 1st brick, we sample every hop index(similar to ::hop). For all the bricks(inlcuding the 1st) we sample the final hiddens state. out_time = in_time/hop + 1
 */
int forward_bricked_rnn(float* output_signal, unsigned rnn_hidden, float* input_signal,
  unsigned in_time, unsigned in_dims, unsigned window, unsigned hop,
  rnn_t rnn, const void* params, void* buffers,
  int bi_direction, int sample_first_brick, int normalize);

/** Backward Bricking and application of the backward rnn for an input signal
 * @param[out]       output_signal        pointer to output signal. size = out_time * rnn_hidden
 * @param[in]        rnn_hidden           output dimension for the current cell
 * @param[in]        input_signal         pointer to input signal. size = in_time * in_dims
 * @param[in]        in_time                 number of input time steps.
 * @param[in]        in_dims              input dimensions
 * @param[in]        window               window length for each brick. For the final brick, the left over time steps are used(need not be window in length for the last brick)
 * @param[in]        hop                  hop distance for between bricks
 * @param[in]        rnn                  function pointer to the rnn
 * @param[in]        params               pointer to the parameters for the rnn
 * @param[in,out]    buffers              pointer to buffer for the rnn
 * @param[in]        bi_direction         determine if the ouput if for a bi-directional rnn. 
 * @param[in]        sample_last_brick   determine if the last brick should also be sampled
 *                   -> if = 0, only the first(last in reverse) hidden state of each brick is sampled. out_time = (in_time-window)/hop + 1
 *                   -> if = 1, for the last brick, we sample every hop index in reverse(similar to ::hop in reverse). For all the bricks(inlcuding the last) we sample the first hiddens state(last in reverse). out_time = in_time/hop + 1
 */
int backward_bricked_rnn(float* output_signal, unsigned rnn_hidden, float* input_signal,
  unsigned in_time, unsigned in_dims, unsigned window, unsigned hop,
  rnn_t rnn, const void* params, void* buffers,
  int bi_direction, int sample_last_brick, int normalize);


#endif

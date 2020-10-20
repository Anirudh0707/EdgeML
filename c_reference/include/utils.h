// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __UTILS_H__
#define __UTILS_H__

#include <math.h>
#include <float.h>

float min(float a, float b);
float max(float a, float b);

float relu(float x);
float sigmoid(float x);
float tanhyperbolic(float x);
float quantTanh(float x);
float quantSigmoid(float x);

void v_relu(const float* const vec, unsigned len, float* const ret);
void v_sigmoid(const float* const vec, unsigned len, float* const ret);
void v_tanh(const float* const vec, unsigned len, float* const ret);
void v_quantSigmoid(const float* const vec, unsigned len, float* const ret);
void v_quantTanh(const float* const vec, unsigned len, float* const ret);

/* Scaled matrix-vector multiplication:  ret = alpha * ret + beta * mat * vec
   alpha and beta are scalars
   ret is of size nrows, vec is of size ncols
   mat is of size nrows * ncols, stored in row major */
void matVec(const float* const mat, const float* const vec,
  unsigned nrows, unsigned ncols,
  float alpha, float beta,
  float* const ret);

/* Scaled matrix-matrix multiplication: ret = alpha * ret + beta * matA * matB
   matA      first matrix; size = nrows * ncommon
   matB      second matrix; size = ncommon * ncols
   nrows     number of rows in the first matrix
   ncommon   number of columns in the first matrix/number of rows in the second matrix
   ncols     number of columns in the second matrix
   alpha     scaling factor for the previously-stored output matrix
   beta      scaling factor for the result of the multiplication (matA * matB)
   ret       matrix multiplication output
 */
void matmul(const float* const matA, const float* const matB,
  unsigned nrows, unsigned ncommon, unsigned ncols,
  float alpha, float beta,
  float* const ret);

// scaled vector addition: ret = scalar1 * vec1 + scalar2 * vector2
void v_add(float scalar1, const float* const vec1,
  float scalar2, const float* const vec2,
  unsigned len, float* const ret);

// point-wise vector multiplication ret = vec2 * vec1 
void v_mult(const float* const vec1, const float* const vec2,
  unsigned len, float* const ret);

// point-wise vector division ret = vec2 / vec1 
void v_div(const float* const vec1, const float* const vec2,
  unsigned len, float* const ret);

// Return squared Euclidean distance between vec1 and vec2
float l2squared(const float* const vec1,
  const float* const vec2, unsigned dim);

// Return index with max value, if tied, return first tied index.
unsigned argmax(const float* const vec, unsigned len);

// ret[i] = exp(input[i]) / \sum_i exp(input[i])
void softmax(const float* const input, unsigned len, float* const ret);

/* Custom non-linear layer for the phoneme detection model. It can be used for other time-series problems if necessary
   output_signal    pointer to the output signal, size = out_time * (in_channels / 2)
   input_signal     pointer to the input signal. size = in_time * in_channels
   in_time          number of input time steps
   in_channels      number of input channels. The output will have the half the number of input channels. 
                    Necessary for in_channels % 2 == 0
 */
int semi_sigmoid_tanh(float* output_signal, float* input_signal,
  unsigned in_time, unsigned in_channels);

#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include"rnn_io.h"
#include"rnn_params.h"
#include"fastgrnn.h"
#include"utils.h"

int main(){
    
    FastGRNN_LR_Params RNN_params = {
    .mean   = 0,
    .stdDev = 0,
    .W1     = F_W1,
    .W2     = F_W2,
    .wRank  = LOW_RANK,
    .U1     = F_U1,
    .U2     = F_U2,
    .uRank  = LOW_RANK,
    .Bg     = F_BIAS_GATE,
    .Bh     = F_BIAS_UPDATE,
    .sigmoid_zeta = sigmoid(F_ZETA),
    .sigmoid_nu   = sigmoid(F_NU)
    };

    float preComp[O_F] = { 0.0 };
    float tempLRW[LOW_RANK] = { 0.0 };
    float tempLRU[LOW_RANK] = { 0.0 };
    float normFeatures[I_F] = { 0.0 };
    FastGRNN_LR_Buffers buffers = {
        .preComp = preComp,
        .tempLRW = tempLRW,
        .tempLRU = tempLRU,
        .normFeatures = normFeatures
    };

    float pred[O_T * O_F] = {0.0};
    // float pred[O_F] = {0.0};
    // int fastgrnn_lr(float* const hiddenState, unsigned hiddenDims,
    // const float* const input, unsigned inputDims, unsigned steps,
    // const void* params, void* buffers, int backward, int normalize);

    float* temp_hiddenstate = (float*)malloc(O_F*sizeof(float));
    for(int t = 0 ; t < I_T ; t++){
        fastgrnn_lr(temp_hiddenstate, O_F,
            INPUT + (t * I_F) , I_F, 1,
            &RNN_params, &buffers, 0, 0);
        memcpy(pred + (t * O_F), temp_hiddenstate, O_F*sizeof(float));
    }
    free(temp_hiddenstate);
    float error = 0, denom = 0;
    for(int t = 0 ; t < O_T ; t++){
        for(int d = 0 ; d < O_F ; d++){
            error += ((pred[t * O_F + d] - OUTPUT[t * O_F + d]) * (pred[t * O_F + d] - OUTPUT[t * O_F + d]));
            denom += OUTPUT[t * O_F + d] * OUTPUT[t * O_F + d];
        }
    }
    float avg_error = error/(O_T*O_F);
    printf("RNN Block\n");
    printf("Aggregate Squared Error : %f   ;   Mean Sqaured Error : %f  \n", error, avg_error);
    printf("RMS : %f \n", error/denom);

    return 0 ;
}
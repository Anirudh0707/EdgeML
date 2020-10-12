// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include"bricked_rnn_io.h"
#include"rnn_params.h"
#include"fastgrnn.h"
#include"utils.h"

int main(){
    
    FastGRNN_LR_Params bwd_RNN_params = {
    .mean   = 0,
    .stdDev = 0,
    .W1     = B_W1,
    .W2     = B_W2,
    .wRank  = LOW_RANK,
    .U1     = B_U1,
    .U2     = B_U2,
    .uRank  = LOW_RANK,
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

    
    int fwd_window = 60, hop = 3, bwd_window = 15, rnn_hidden = O_F>>1,out_index = 0; 
    unsigned out_T = I_T/hop + 1;
    float pred[O_T * O_F] = {0.0};
    // int fastgrnn_lr(float* const hiddenState, unsigned hiddenDims,
    // const float* const input, unsigned inputDims, unsigned steps,
    // const void* params, void* buffers, int backward, int normalize);
    float* temp_hiddenstate = (float*)calloc(rnn_hidden, sizeof(float));

    // Forward
    for(int t = 0 ; t < fwd_window ; t++){
        fastgrnn_lr(temp_hiddenstate, rnn_hidden,
            INPUT + (t * I_F) , I_F, 1,
            &fwd_RNN_params, &buffers, 0, 0);
        if(t % 3==0){
            memcpy(pred + ((out_index++)*2*rnn_hidden), temp_hiddenstate, rnn_hidden*sizeof(float));
        }
    }
    memcpy(pred + ((out_index++)*2*rnn_hidden), temp_hiddenstate, rnn_hidden*sizeof(float));
    for(int t = hop ; t <= I_T - fwd_window ; t += hop ){
        memset(temp_hiddenstate, 0, rnn_hidden*sizeof(float));
        fastgrnn_lr(temp_hiddenstate, rnn_hidden,
            INPUT + (t * I_F) , I_F, fwd_window,
            &fwd_RNN_params, &buffers, 0, 0);
        memcpy(pred + ((out_index++)*2*rnn_hidden), temp_hiddenstate, rnn_hidden*sizeof(float));    
    }

    // Backward
    out_index = 0;
    for(int t = 0 ; t < I_T - bwd_window  ; t += hop ){
        memset(temp_hiddenstate, 0, rnn_hidden*sizeof(float));
        fastgrnn_lr(temp_hiddenstate, rnn_hidden,
            INPUT + (t * I_F) , I_F, bwd_window,
            &bwd_RNN_params, &buffers, 1, 0);
        memcpy(pred + ((out_index++)*2*rnn_hidden + rnn_hidden), temp_hiddenstate, rnn_hidden*sizeof(float));
    }
    out_index += bwd_window/3;
    memset(temp_hiddenstate, 0, rnn_hidden*sizeof(float));
    for(int t = I_T - 1 ; t >= I_T - bwd_window; t--){
        fastgrnn_lr(temp_hiddenstate, rnn_hidden,
            INPUT + (t * I_F) , I_F, 1,
            &bwd_RNN_params, &buffers, 0, 0);
        if((t - I_T + 1) % 3 == 0){
            memcpy(pred + ((out_index--)*2*rnn_hidden + rnn_hidden), temp_hiddenstate, rnn_hidden*sizeof(float));
        }
    }
    memcpy(pred + ((out_index)*2*rnn_hidden + rnn_hidden), temp_hiddenstate, rnn_hidden*sizeof(float));
    free(temp_hiddenstate);
    
    float error = 0, denom = 0;
    for(int t = 0 ; t < out_T ; t++){
        for(int d = 0 ; d < O_F ; d++){
            error += ((pred[t * O_F + d] - OUTPUT[t * O_F + d]) * (pred[t * O_F + d] - OUTPUT[t * O_F + d]));
            denom += OUTPUT[t * O_F + d] * OUTPUT[t * O_F + d];
        }
    }
    float avg_error = error/(O_T*O_F);
    printf("RNN Fully-Bricked Block\n");
    printf("Aggregate Squared Error : %f   ;   Mean Sqaured Error : %f  \n", error, avg_error);
    printf("RMS : %f \n", error/denom);

    return 0 ;
}
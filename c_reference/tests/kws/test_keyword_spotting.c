// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include"keyword_spotting_io_1.h"
#include"precnn_params.h"
#include"rnn_params.h"
#include"postcnn_params.h"

#include"conv1d.h"
#include"dscnn.h"
#include"fastgrnn.h"
#include"utils.h"

ConvLayers_LR_Params conv_params = {
    .W1 = CNN1_W1,
    .W2 = CNN1_W2,
    .B = CNN1_BIAS,
    .rank = LOW_RANK
};

ConvLayers_Params depth_param_2 = {
    .W = CNN2_DEPTH_W,
    .B = CNN2_DEPTH_BIAS,
};

ConvLayers_LR_Params point_param_2 = {
    .W1 = CNN2_POINT_W1,
    .W2 = CNN2_POINT_W2,
    .B = CNN2_POINT_BIAS,
    .rank = LOW_RANK
};

ConvLayers_Params depth_param_3 = {
    .W = CNN3_DEPTH_W,
    .B = CNN3_DEPTH_BIAS,
};

ConvLayers_LR_Params point_param_3 = {
    .W1 = CNN3_POINT_W1,
    .W2 = CNN3_POINT_W2,
    .B = CNN3_POINT_BIAS,
    .rank = LOW_RANK
};

ConvLayers_Params depth_param_4 = {
    .W = CNN4_DEPTH_W,
    .B = CNN4_DEPTH_BIAS,
};

ConvLayers_LR_Params point_param_4 = {
    .W1 = CNN4_POINT_W1,
    .W2 = CNN4_POINT_W2,
    .B = CNN4_POINT_BIAS,
    .rank = LOW_RANK
};

ConvLayers_Params depth_param_5 = {
    .W = CNN5_DEPTH_W,
    .B = CNN5_DEPTH_BIAS,
};

ConvLayers_LR_Params point_param_5 = {
    .W1 = CNN5_POINT_W1,
    .W2 = CNN5_POINT_W2,
    .B = CNN5_POINT_BIAS,
    .rank = LOW_RANK
};

float preComp[RNN_I_F] = { 0.0 };
float tempLRW[LOW_RANK] = { 0.0 };
float tempLRU[LOW_RANK] = { 0.0 };
float normFeatures[RNN_I_F] = { 0.0 };
FastGRNN_LR_Buffers buffers = {
    .preComp = preComp,
    .tempLRW = tempLRW,
    .tempLRU = tempLRU,
    .normFeatures = normFeatures
};

// Under Dev
void key_word_spotting(){

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
    float pred[O_T * POST_CNN_O_F] = {0.0};
    unsigned in_T, out_T;

    /* Pre-CNN */
    in_T = I_T;
    out_T = in_T - FILT + 2*(FILT>>1) + 1; // Depth pad = 2 for kernel size =5. SAME PAD
    float* cnn1_out = (float*)malloc(out_T * PRE_CNN_O_F * sizeof(float));
    DSCNN_LR(cnn1_out, INPUT, in_T, I_F, BNORM_CNN1_MEAN, BNORM_CNN1_VAR,
        0, 0, 0, 0, PRE_CNN_O_F, 2, FILT,
        &conv_params, 2); // regular tanh activation

    BatchNorm1d(0, cnn1_out, in_T, PRE_CNN_O_F, 
        BNORM_RNN_MEAN, BNORM_RNN_VAR, 0, 0, 0, 1,  0.00001);

    /* RNN */
    int fwd_window = 60, hop = 3, bwd_window = 15, rnn_hidden = RNN_O_F>>1, out_index = 0;
 
    out_T = in_T/hop + 1;
    float* temp_hiddenstate = (float*)calloc(rnn_hidden, sizeof(float));
    float* rnn_out = (float*)malloc(out_T * (rnn_hidden<<1) * sizeof(float));
    // Forward
    for(int t = 0 ; t < fwd_window ; t++){
        fastgrnn_lr(temp_hiddenstate, rnn_hidden,
            cnn1_out + (t * RNN_I_F) , RNN_I_F, 1,
            &fwd_RNN_params, &buffers, 0, 0);
        if(t % 3==0){
            memcpy(rnn_out + ((out_index++)*(rnn_hidden<<1)), temp_hiddenstate, rnn_hidden*sizeof(float));
        }
    }
    memcpy(rnn_out + ((out_index++)*(rnn_hidden<<1)), temp_hiddenstate, rnn_hidden*sizeof(float));
    for(int t = hop ; t <= in_T - fwd_window ; t += hop ){
        memset(temp_hiddenstate, 0, rnn_hidden*sizeof(float));
        fastgrnn_lr(temp_hiddenstate, rnn_hidden,
            cnn1_out + (t * RNN_I_F) , RNN_I_F, fwd_window,
            &fwd_RNN_params, &buffers, 0, 0);
        memcpy(rnn_out + ((out_index++)*(rnn_hidden<<1)), temp_hiddenstate, rnn_hidden*sizeof(float));    
    }

    // Backward
    out_index = 0;
    for(int t = 0 ; t < in_T - bwd_window  ; t += hop ){
        memset(temp_hiddenstate, 0, rnn_hidden*sizeof(float));
        fastgrnn_lr(temp_hiddenstate, rnn_hidden,
            cnn1_out + (t * RNN_I_F) , RNN_I_F, bwd_window,
            &bwd_RNN_params, &buffers, 1, 0);
        memcpy(rnn_out + ((out_index++)*(rnn_hidden<<1) + rnn_hidden), temp_hiddenstate, rnn_hidden*sizeof(float));
    }
    out_index += bwd_window/hop;
    memset(temp_hiddenstate, 0, rnn_hidden*sizeof(float));
    for(int t = in_T - 1 ; t >= in_T - bwd_window; t--){
        fastgrnn_lr(temp_hiddenstate, rnn_hidden,
            cnn1_out + (t * RNN_I_F) , RNN_I_F, 1,
            &bwd_RNN_params, &buffers, 0, 0);
        if((in_T - 1 - t) % 3 == 0){
            memcpy(rnn_out + ((out_index--)*(rnn_hidden<<1) + rnn_hidden), temp_hiddenstate, rnn_hidden*sizeof(float));
        }
    }
    memcpy(rnn_out + ((out_index)*(rnn_hidden<<1) + rnn_hidden), temp_hiddenstate, rnn_hidden*sizeof(float));
    free(temp_hiddenstate);

    free(cnn1_out);

    /* Post-CNN */
    in_T = out_T;
    out_T = in_T - DEPTH_FILT + 2*(DEPTH_FILT>>1) + 1; // Depth pad = 2 for kernel size =5. SAME PAD
    out_T = out_T - POST_CNN_POOL + 2*(0) + 1; // Pool pad = none/0
    float* cnn2_out = (float*)malloc(out_T * POST_CNN_I_F * sizeof(float));
    DSCNN_LR_Point_Depth(cnn2_out, rnn_out, in_T, POST_CNN_I_F, CNN2_BNORM_MEAN, CNN2_BNORM_VAR,
        0, 0, 0, 1, POST_CNN_I_F>>1, DEPTH_FILT>>1, 
        DEPTH_FILT, &depth_param_2, 0, POST_CNN_I_F, 
        0, POINT_FILT, &point_param_2, 0, 
        0, POST_CNN_POOL, 0);
    free(rnn_out);
    
    in_T = out_T;
    out_T = in_T - DEPTH_FILT + 2*(DEPTH_FILT>>1) + 1; // Depth pad = 2 for kernel size =5. SAME PAD
    out_T = out_T - POST_CNN_POOL + 2*(0) + 1; // Pool pad = none/0
    float* cnn3_out = (float*)malloc(out_T * POST_CNN_I_F * sizeof(float));
    DSCNN_LR_Point_Depth(cnn3_out, cnn2_out, in_T, POST_CNN_I_F, CNN3_BNORM_MEAN, CNN3_BNORM_VAR,
        0, 0, 0, 1, POST_CNN_I_F>>1, DEPTH_FILT>>1, 
        DEPTH_FILT, &depth_param_3, 0, POST_CNN_I_F, 
        0, POINT_FILT, &point_param_3, 0, 
        0, POST_CNN_POOL, 0);
        free(cnn2_out);

    in_T = out_T;
    out_T = in_T - DEPTH_FILT + 2*(DEPTH_FILT>>1) + 1; // Depth pad = 2 for kernel size =5. SAME PAD
    out_T = out_T - POST_CNN_POOL + 2*(0) + 1; // Pool pad = none/0
    float* cnn4_out = (float*)malloc(out_T * POST_CNN_I_F * sizeof(float));
    DSCNN_LR_Point_Depth(cnn4_out, cnn3_out, in_T, POST_CNN_I_F, CNN4_BNORM_MEAN, CNN4_BNORM_VAR,
        0, 0, 0, 1, POST_CNN_I_F>>1, DEPTH_FILT>>1, 
        DEPTH_FILT, &depth_param_4, 0, POST_CNN_I_F, 
        0, POINT_FILT, &point_param_4, 0, 
        0, POST_CNN_POOL, 0);
    free(cnn3_out);

    in_T = out_T;
    out_T = in_T - DEPTH_FILT + 2*(DEPTH_FILT>>1) + 1; // Depth pad = 2 for kernel size =5. SAME PAD
    out_T = out_T - POST_CNN_POOL + 2*(0) + 1; // Pool pad = none/0
    DSCNN_LR_Point_Depth(pred, cnn4_out, in_T, POST_CNN_I_F, CNN5_BNORM_MEAN, CNN5_BNORM_VAR,
        0, 0, 0, 1, POST_CNN_I_F>>1, DEPTH_FILT>>1, 
        DEPTH_FILT, &depth_param_5, 0, POST_CNN_O_F, 
        0, POINT_FILT, &point_param_5, 0, 
        0, POST_CNN_POOL, 0);
    free(cnn4_out);

    if(out_T != O_T){
        printf("Error, Estimated Output and Actual ouput time axis mis-match");
        return;
    }

    // Error Check
    float error = 0, denom = 0;
    for(int t = 0 ; t < out_T ; t++){
        for(int d = 0 ; d < POST_CNN_O_F ; d++){
            error += ((pred[t * POST_CNN_O_F + d] - OUTPUT[t * POST_CNN_O_F + d]) * (pred[t * POST_CNN_O_F + d] - OUTPUT[t * POST_CNN_O_F + d]));
            denom += OUTPUT[t * POST_CNN_O_F + d] * OUTPUT[t * POST_CNN_O_F + d];
        }
    }
    
    float avg_error = error/(O_T*POST_CNN_O_F);
    printf("Full Network\n");
    printf("Aggregate Squared Error : %f   ;   Mean Sqaured Error : %f  \n", error, avg_error);
    printf("Denominator for RMSE : %f \n", denom);
    printf("RMSE : %f \n", error/denom);
}

int main(){
    
    key_word_spotting();

    return 0 ;
}
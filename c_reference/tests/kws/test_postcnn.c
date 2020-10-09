#include<stdio.h>
#include<stdlib.h>

#include"postcnn_params.h"
#include"postcnn_io.h"
#include"conv1d.h"
#include"dscnn.h"

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

int main(){
    unsigned out_T, in_T;

    float pred[O_T * O_F] = {};


    // DSCNN LR Point Depth(float* output_signal, float* input_signal, unsigned in_T, unsigned in_channels, float* mean, float* var,
    // unsigned affine, float* gamma, float* beta, unsigned in_place, unsigned depth_cnn_hidden, int depth_cnn_padding, 
    // unsigned depth_cnn_kernel_size, const void* depth_cnn_params, int depth_cnn_activations, unsigned point_cnn_hidden, 
    // int point_cnn_padding, unsigned point_cnn_kernel_size, const void* point_cnn_params, int point_cnn_activations, 
    // int pool_padding, unsigned pool_kernel_size, int pool_activation)
    in_T = I_T;
    out_T = in_T - DEPTH_FILT + 2*(DEPTH_FILT>>1) + 1; // Depth pad = 2 for kernel size =5. SAME PAD
    out_T = out_T - POOL_FILT + 2*(0) + 1; // Pool pad = none/0
    float* cnn2_out = (float*)malloc(out_T * I_F * sizeof(float));
    DSCNN_LR_Point_Depth(cnn2_out, INPUT, in_T, I_F, CNN2_BNORM_MEAN, CNN2_BNORM_VAR,
    0, 0, 0, 1, I_F>>1, DEPTH_FILT>>1, 
    DEPTH_FILT, &depth_param_2, 0, I_F, 
    0, POINT_FILT, &point_param_2, 0, 
    0, POOL_FILT, 0);
    // free(cnn1_out);
    
    in_T = out_T;
    out_T = in_T - DEPTH_FILT + 2*(DEPTH_FILT>>1) + 1; // Depth pad = 2 for kernel size =5. SAME PAD
    out_T = out_T - POOL_FILT + 2*(0) + 1; // Pool pad = none/0
    float* cnn3_out = (float*)malloc(out_T * I_F * sizeof(float));
    DSCNN_LR_Point_Depth(cnn3_out, cnn2_out, in_T, I_F, CNN3_BNORM_MEAN, CNN3_BNORM_VAR,
    0, 0, 0, 1, I_F>>1, DEPTH_FILT>>1, 
    DEPTH_FILT, &depth_param_3, 0, I_F, 
    0, POINT_FILT, &point_param_3, 0, 
    0, POOL_FILT, 0);
    free(cnn2_out);

    in_T = out_T;
    out_T = in_T - DEPTH_FILT + 2*(DEPTH_FILT>>1) + 1; // Depth pad = 2 for kernel size =5. SAME PAD
    out_T = out_T - POOL_FILT + 2*(0) + 1; // Pool pad = none/0
    float* cnn4_out = (float*)malloc(out_T * I_F * sizeof(float));
    DSCNN_LR_Point_Depth(cnn4_out, cnn3_out, in_T, I_F, CNN4_BNORM_MEAN, CNN4_BNORM_VAR,
    0, 0, 0, 1, I_F>>1, DEPTH_FILT>>1, 
    DEPTH_FILT, &depth_param_4, 0, I_F, 
    0, POINT_FILT, &point_param_4, 0, 
    0, POOL_FILT, 0);
    free(cnn3_out);

    in_T = out_T;
    out_T = in_T - DEPTH_FILT + 2*(DEPTH_FILT>>1) + 1; // Depth pad = 2 for kernel size =5. SAME PAD
    out_T = out_T - POOL_FILT + 2*(0) + 1; // Pool pad = none/0
    // float* cnn5_out = (float*)malloc(out_T * I_F * sizeof(float));
    DSCNN_LR_Point_Depth(pred, cnn4_out, in_T, I_F, CNN5_BNORM_MEAN, CNN5_BNORM_VAR,
    0, 0, 0, 1, I_F>>1, DEPTH_FILT>>1, 
    DEPTH_FILT, &depth_param_5, 0, O_F, 
    0, POINT_FILT, &point_param_5, 0, 
    0, POOL_FILT, 0);
    free(cnn4_out);

    if(out_T != O_T){
        printf("Error, Estimatd Output and Actual ouput teim axis mis-match");
        return 0;
    }
    
    // Calculate Error(Aggregate Squared and Mean Squared)
    float error = 0;
    float denom = 0;
    for(int t = 0 ; t < O_T ; t++){
        for(int d = 0 ; d < O_F ; d++){
            error += ((pred[t * O_F + d] - OUTPUT[t * O_F + d]) * (pred[t * O_F + d] - OUTPUT[t * O_F + d]));
            denom += (OUTPUT[t * O_F + d] * OUTPUT[t * O_F + d]);
        }
    }
    float avg_error = error/(O_T*O_F);
    printf("Post-RNN CNN Block\n");
    printf("Aggregate Squared Error : %f   ;   Mean Sqaured Error : %f  \n", error, avg_error);
    printf("RMS : %f \n", error/denom);

    return 0 ;
}
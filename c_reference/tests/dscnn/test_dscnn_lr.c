#include<stdio.h>
#include<stdlib.h>

#include"dscnn_param_lr.h"
#include"conv1d.h"
#include"dscnn.h"
#include"conv_utils.h"

int main(){
    
    ConvLayers_LR_Params conv_params = {
        .W1 = CNN1_W1,
        .W2 = CNN1_W2,
        .B = CNN1_BIAS,
        .rank = LOW_RANK
    };

    float pred[O_T * O_F] = {};
    // int DSCNN_LR(float* output_signal, float* input_signal, unsigned in_T, unsigned in_channels, float* mean, float* stddev,
    // unsigned affine, float* gamma, float* beta, unsigned in_place, unsigned cnn_hidden, int cnn_padding, unsigned cnn_kernel_size,
    // const void* cnn_params, int cnn_activations)
    DSCNN_LR(pred, INPUT, I_T, I_F, BNORM_CNN1_MEAN, BNORM_CNN1_VAR,
    0, 0, 0, 0, O_F, 2, FILT,
    &conv_params, 0);

    // Calculate Error(Aggregate Squared and Mean Squared)
    float error = 0, denom = 0;
    for(int t = 0 ; t < O_T ; t++){
        for(int d = 0 ; d < O_F ; d++){
            error += ((pred[t * O_F + d] - OUTPUT[t * O_F + d]) * (pred[t * O_F + d] - OUTPUT[t * O_F + d]));
            denom += OUTPUT[t * O_F + d] * OUTPUT[t * O_F + d];
        }
    }
    float avg_error = error/(O_T*O_F);
    printf("DSCNN K5 Block\n");
    printf("Aggregate Squared Error : %f   ;   Mean Sqaured Error : %f  \n", error, avg_error);
    printf("RMS : %f \n", error/denom);

    return 0 ;
}
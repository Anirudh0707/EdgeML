#include<stdio.h>
#include<stdlib.h>

#include"conv_param_lr_depth.h"
#include"conv1d.h"
#include"conv_utils.h"

int main(){
    
    ConvLayers_LR_Params conv_params = {
        .W1 = CONV_W1,
        .W2 = CONV_W2,
        .B = CONV_BIAS,
        .rank = LR_RANK
    };

    float pred[O_T * O_F] = {};
    Conv1D_Depth_LR(pred, O_T, INPUT, 1, I_T, I_F, PAD, FILT, &conv_params, ACT);
    float error = 0;
    for(int t = 0 ; t < O_T ; t++){
        for(int d = 0 ; d < I_F ; d++){
            error += ((pred[t * I_F + d] - OUTPUT[t * I_F + d]) * (pred[t * I_F + d] - OUTPUT[t * I_F + d]));
        }
    }
    float avg_error = error/(O_T*I_F);
    printf("Squared Error : %f \t ; MSE : %f  \n", error, avg_error);

    return 0 ;
}
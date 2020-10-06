#include<stdio.h>
#include<stdlib.h>

#include"conv_param_depth.h"
#include"conv1d.h"
#include"conv_utils.h"

int main(){
    
    ConvLayers_Params conv_params = {
        .W = CONV_WEIGHT,
        .B = CONV_BIAS,
    };

    float pred[O_T * I_F] = {};
    Conv1D_Depth(pred, O_T, INPUT, I_T, I_F, PAD, FILT, &conv_params, ACT);
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
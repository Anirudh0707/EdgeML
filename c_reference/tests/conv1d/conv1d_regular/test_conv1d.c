#include<stdio.h>
#include<stdlib.h>

#include"conv_param.h"
#include"conv1d.h"
#include"conv_utils.h"

int main(){
    
    ConvLayers_Params conv_params = {
        .W = CONV_WEIGHT,
        .B = CONV_BIAS,
    };

    float pred[O_T * O_F] = {};
    Conv1D(pred, O_T, O_F, INPUT, 1, I_T, I_F, PAD, FILT, &conv_params, ACT);
    float error = 0;
    for(int t = 0 ; t < O_T ; t++){
        for(int d = 0 ; d < O_F ; d++){
            error += ((pred[t * O_F + d] - OUTPUT[t * O_F + d]) * (pred[t * O_F + d] - OUTPUT[t * O_F + d]));
        }
    }
    float avg_error = error/(O_T*O_F);
    printf("Squared Error : %f \t ; MSE : %f  \n", error, avg_error);

    return 0 ;
}
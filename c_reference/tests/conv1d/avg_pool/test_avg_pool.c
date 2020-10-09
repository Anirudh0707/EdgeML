#include<stdio.h>
#include<stdlib.h>

#include"avg_io.h"
#include"conv1d.h"

int main(){

    float pred[O_T * O_F] = {};
    AvgPool1D(pred, O_T, INPUT, I_T, I_F, PAD, FILT, ACT);
    float error = 0;
    float denom = 0;
    for(int t = 0 ; t < O_T ; t++){
        for(int d = 0 ; d < O_F ; d++){
            error += ((pred[t * O_F + d] - OUTPUT[t * O_F + d]) * (pred[t * O_F + d] - OUTPUT[t * O_F + d]));
            denom += OUTPUT[t * O_F + d] * OUTPUT[t * O_F + d] ;
        }
    }
    float avg_error = error/(O_T*O_F);
    printf("Testing Average Pool\n");
    printf("Squared Error : %f \t ; MSE : %f  \n", error, avg_error);
    printf("Relative Squared Error : %f", error/denom)
    return 0 ;
}
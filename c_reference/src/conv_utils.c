#include"conv_utils.h"
#include <math.h>
#include <float.h>

int prepareLowRankConvMat(float* out, float* W1, float* W2, unsigned rank, unsigned I, unsigned J){
    for(i = 0 ; i < I, i++){
        for(j = 0 ; j < J, j++){
            float sum = 0;
            for(k = 0; k < rank ; k++){
                sum += (W1[i * rank + k] * W2[k * J + j]);
            }
            out[i * J + j] = sum;
        }
    }
}
#include"conv1d.h"
#include"conv_utils.h"
#include<stdlib.h>
#include<math.h>
#include<stdio.h>

int Conv1D_LR(float *output_signal, unsigned out_T, unsigned out_channels, const float *input_signal, 
    unsigned in_T, unsigned in_channels, int padding, unsigned kernel_size, 
    const void* params, int activations){

    const ConvLayers_LR_Params* tparams= (ConvLayers_LR_Params*) params;

    if(padding == -1){
        padding = kernel_size >> 1;
    }
    
    float* tempW = (float*)malloc(out_channels * in_channels * kernel_size * sizeof(float)) ;
    prepareLowRankConvMat(tempW, tparams->W1, tparams->W2, tparams->rank, out_channels, in_channels * kernel_size);
    // Perform the Convolution
    // input.shape  = [in_T,  in_channels] 
    // output.shape = [out_T, out_channels]
    // filter.shape = [out_channels, in_channels, kernel_size]
    for(int t = 0; t < out_T; t++){
        for(int co = 0; co < out_channels ; co++){
            float sum = 0;
            for(int tf = 0 ; tf < kernel_size ; tf++ ){
                for(int ci = 0 ; ci < in_channels ; ci++){
                    if(((t + tf) < padding) || ((t + tf) >= (in_T + padding)))
                        continue;
                    else
                        sum += (input_signal[((tf + t) - padding) * in_channels + ci] * tempW[co * in_channels * kernel_size + ci * kernel_size + tf]);
                }
            }
            if(activations == 1){
                output_signal[t * out_channels + co] = sigmoid(sum + tparams->B[co]);     
            }
            else if(activations == 2){
                output_signal[t * out_channels + co] = tanh(sum + tparams->B[co]);
            }
            else if(activations == 3){
                output_signal[t * out_channels + co] = relu(sum + tparams->B[co]);
            }
            else{
                output_signal[t * out_channels + co] = sum + tparams->B[co];
            }
        }
    }
    free(tempW);
    return 0;
}

int Conv1D_Depth_LR(float *output_signal, unsigned out_T, const float *input_signal, 
    unsigned in_T, unsigned in_channels, int padding, unsigned kernel_size, 
    const void* params, int activations){

    const ConvLayers_LR_Params* tparams= (ConvLayers_LR_Params*) params;

    if(padding == -1){
        padding = kernel_size >> 1;
    }

    float* tempW = (float*)malloc(in_channels * kernel_size * sizeof(float)) ;
    prepareLowRankConvMat(tempW, tparams->W1, tparams->W2, tparams->rank, in_channels, kernel_size);
    // Perform the Convolution
    // input.shape  = [N,  in_T,  in_channels] 
    // output.shape = [N, out_T, in_channels]
    // filter.shape = [(out)in_channels, in_channels, kernel_size]
    for(int t = 0; t < out_T; t++){
        for(int ci = 0; ci < in_channels ; ci++){
            float sum = 0;
            for(int tf = 0 ; tf < kernel_size ; tf++ ){
                if(((t + tf) < padding) || ((t + tf) >= (in_T + padding)))
                    continue;
                else
                    sum += (input_signal[((tf + t) - padding) * in_channels + ci] * tempW[ci * kernel_size + tf]);
            }
            if(activations == 1){
                output_signal[t * in_channels + ci] = sigmoid(sum + tparams->B[ci]);     
            }
            else if(activations == 2){
                output_signal[t * in_channels + ci] = tanh(sum + tparams->B[ci]);
            }
            else if(activations == 3){
                output_signal[t * in_channels + ci] = relu(sum + tparams->B[ci]);
            }
            else{
                output_signal[t * in_channels + ci] = sum + tparams->B[ci];
            }
        }
    }    
    free(tempW);
    return 0;
}



int Conv1D(float *output_signal, unsigned out_T, unsigned out_channels, const float *input_signal, 
    unsigned in_T, unsigned in_channels, int padding, unsigned kernel_size, 
    const void* params, int activations){

    const ConvLayers_Params* tparams= (ConvLayers_Params*) params;

    if(padding == -1){
        padding = kernel_size >> 1;
    }
    float sum;
    // Perform the Convolution
    // input.shape  = [N,  in_T,  in_channels] 
    // output.shape = [N, out_T, out_channels]
    // filter.shape = [out_channels, in_channels, kernel_size]
    for(int t = 0; t < out_T; t++){
        for(int co = 0; co < out_channels ; co++){
            sum = 0;
            for(int tf = 0 ; tf < kernel_size ; tf++ ){
                for(int ci = 0 ; ci < in_channels ; ci++){
                    if(((t + tf) < padding) || ((t + tf) >= (in_T + padding)))
                        continue;
                    else
                        sum += (input_signal[((tf + t) - padding) * in_channels + ci] * tparams->W[co * in_channels * kernel_size + ci * kernel_size + tf]);
                }
            }
            if(activations == 1){
                output_signal[t * out_channels + co] = sigmoid(sum + tparams->B[co]);     
            }
            else if(activations == 2){
                output_signal[t * out_channels + co] = tanh(sum + tparams->B[co]);
            }
            else if(activations == 3){
                output_signal[t * out_channels + co] = relu(sum + tparams->B[co]);
            }
            else{
                output_signal[t * out_channels + co] = sum + tparams->B[co];
            }
        }
    }    
    return 0;
}

int Conv1D_Depth(float *output_signal, unsigned out_T, const float *input_signal, 
    unsigned in_T, unsigned in_channels, int padding, unsigned kernel_size, 
    const void* params, int activations){

    const ConvLayers_Params* tparams= (ConvLayers_Params*) params;

    if(padding == -1){
        padding = kernel_size >> 1;
    }

    // Perform the Convolution
    // input.shape  = [N,  in_T,  in_channels] 
    // output.shape = [N, out_T, in_channels]
    // filter.shape = [(out)in_channels, in_channels, kernel_size]
    for(int t = 0; t < out_T; t++){
        for(int ci = 0; ci < in_channels ; ci++){
            float sum = 0;
            for(int tf = 0 ; tf < kernel_size ; tf++ ){
                if(((t + tf) < padding) || ((t + tf) >= (in_T + padding)))
                    continue;
                else
                    sum += (input_signal[((tf + t) - padding) * in_channels + ci] * tparams->W[ci * kernel_size + tf]);
            }
            if(activations == 1){
                output_signal[t * in_channels + ci] = sigmoid(sum + tparams->B[ci]);     
            }
            else if(activations == 2){
                output_signal[t * in_channels + ci] = tanh(sum + tparams->B[ci]);
            }
            else if(activations == 3){
                output_signal[t * in_channels + ci] = relu(sum + tparams->B[ci]);
            }
            else{
                output_signal[t * in_channels + ci] = sum + tparams->B[ci];
            }
        }
    }
    return 0;
}

int AvgPool1D(float *output_signal, unsigned out_T, const float *input_signal, unsigned in_T, unsigned in_channels, 
    int padding, unsigned kernel_size, int activations){
    
    if(padding == -1){
        padding = kernel_size >> 1;
    }

    for(int t = 0; t < out_T; t++){
        for(int ci = 0 ; ci < in_channels; ci++){
            float sum = 0;
            for(int tf = 0; tf < kernel_size ; tf++){
                if(((t + tf) < padding) || ((t + tf) >= (in_T + padding)))
                    continue;
                else
                    sum += (input_signal[((tf + t) - padding) * in_channels + ci]);
            }
            if(activations == 1){
                output_signal[t * in_channels + ci] = sigmoid(sum/(float)kernel_size);     
            }
            else if(activations == 2){
                output_signal[t * in_channels + ci] = tanh(sum/(float)kernel_size);
            }
            else if(activations == 3){
                output_signal[t * in_channels + ci] = relu(sum/(float)kernel_size);
            }
            else{
                output_signal[t * in_channels + ci] = sum/(float)kernel_size;
            }
            
        }
    }
    return 0;
}

int BatchNorm1d(float* output_signal, float* input_signal, unsigned in_T, unsigned in_channels, 
    float* mean, float* stddev, unsigned in_place){
    if(in_place){
        for(int t = 0; t < in_T ; t++){
            for(int d = 0 ; d < in_channels ; d++){
                input_signal[t * in_channels + d]  = (input_signal[t * in_channels + d] - mean[d])/stddev[d];
            }
        }
    }
    else{
        for(int t = 0; t < in_T ; t++){
            for(int d = 0 ; d < in_channels ; d++){
                output_signal[t * in_channels + d] = (input_signal[t * in_channels + d] - mean[d])/stddev[d];
            }
        }
    }
    return 0;
}
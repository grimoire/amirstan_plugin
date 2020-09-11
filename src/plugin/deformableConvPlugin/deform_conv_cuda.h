#pragma once

#include <cublas_v2.h>

typedef struct _DCN_PARAMS
{
    cublasHandle_t cublas_handle;
    int batchSize = 1;
    int inputChannel = 1;
    int inputW = 256;
    int inputH = 256;
    int outputChannel = 1;
    int kernelW = 3;
    int kernelH = 3;
    int strideW = 1;
    int strideH = 1;
    int padW = 0;
    int padH = 0;
    int dilationW = 1;
    int dilationH = 1;
    int group = 1;
    int deformable_group = 1;
    int im2col_step = 64;
} DCN_PARAMS;

int deform_conv_forward_cuda(float *input, float *weight, float *bias, float *offset,
                             float *output, void* workspace,
                             const DCN_PARAMS &dcn_params,
                             cudaStream_t stream = 0);

                             
void modulated_deform_conv_cuda_forward(
    float* input, float* weight, float* bias,
    float* offset, float* mask, float* output, 
    void *workspace, const DCN_PARAMS &dcn_params, cudaStream_t stream=0);
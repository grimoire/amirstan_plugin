#include "deform_conv_cuda.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cublas_v2.h>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <iostream>

// #include "amir_cuda_util/cublas_util.h"

void deformable_im2col(
    float *data_input, float *data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, float *data_col, cudaStream_t stream);

void tensorPermute(float *dst, float *src, int *src_size, int *permute, int src_dim, cudaStream_t stream);

int deform_conv_forward_cuda(float *input, float *weight, float *offset,
                             float *output, void *workspace,
                             const DCN_PARAMS &dcn_params, cudaStream_t stream)
{
    int sizeof_dtype = sizeof(float);
    cublasHandle_t cublas_handle = dcn_params.cublas_handle;
    int kW = dcn_params.kernelW;
    int kH = dcn_params.kernelH;
    int dW = dcn_params.strideW;
    int dH = dcn_params.strideH;
    int padW = dcn_params.padW;
    int padH = dcn_params.padH;
    int dilationW = dcn_params.dilationW;
    int dilationH = dcn_params.dilationH;
    int group = dcn_params.group;
    int deformable_group = dcn_params.deformable_group;
    int im2col_step = dcn_params.im2col_step;

    long batchSize = dcn_params.batchSize;
    long nInputPlane = dcn_params.inputChannel;
    long inputHeight = dcn_params.inputH;
    long inputWidth = dcn_params.inputW;

    im2col_step = std::min(int(batchSize), im2col_step);
    assert(batchSize % im2col_step == 0);

    long nOutputPlane = dcn_params.outputChannel;

    long outputWidth =
        (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    long outputHeight =
        (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    long long columns_size = nInputPlane * kW * kH * im2col_step * outputHeight * outputWidth;
    float *columns = (float *)workspace;

    float *output_buffer;
    long long output_buffer_size = 0;
    if (im2col_step == 1)
    {
        output_buffer = output;
    }
    else
    {
        output_buffer = columns + columns_size;
        output_buffer_size = batchSize * nOutputPlane * outputWidth * outputHeight;
    }

    long long input_elt_step = im2col_step * nInputPlane * inputHeight * inputWidth;
    long long offset_elt_step = im2col_step * deformable_group * 2 * kH * kW * outputHeight * outputWidth;
    long long out_buffer_step = nOutputPlane * im2col_step * outputHeight * outputWidth;
    long long col_g_step = nInputPlane * kW * kH / group * im2col_step * outputHeight * outputWidth;
    long long weight_g_step = nOutputPlane / group * nInputPlane / group * kH * kW;
    long long out_buffer_g_step = nOutputPlane / group * im2col_step * outputHeight * outputWidth;
    int m = nOutputPlane / group;
    int n = im2col_step * outputHeight * outputWidth;
    int k = nInputPlane / group * kH * kW;
    float alpha = 1.;
    float beta = 0.;

    for (int elt = 0; elt < batchSize / im2col_step; elt++)
    {
        float *input_start = input + elt * input_elt_step;
        float *offset_start = offset + elt * offset_elt_step;

        deformable_im2col(input_start, offset_start, nInputPlane, inputHeight,
                          inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                          dilationW, im2col_step, deformable_group, columns, stream);

        for (int g = 0; g < group; ++g)
        {
            float *weight_start = weight + g * weight_g_step;
            float *col_start = columns + g * col_g_step;
            float *out_buffer_start = output_buffer + elt * out_buffer_step + g * out_buffer_g_step;

            cublasSgemm(cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, m, k,
                        &alpha,
                        col_start, n,
                        weight_start, k,
                        &beta,
                        output_buffer, n);
        }
    }

    if (im2col_step != 1)
    {
        int output_buffer_shape[5] = {batchSize / im2col_step, nOutputPlane,
                                     im2col_step, outputHeight, outputWidth};
        int output_buffer_permute[5] = {0, 2, 1, 3, 4};
        tensorPermute(output, output_buffer, &output_buffer_shape[0], &output_buffer_permute[0], 5, stream);
    }

    return 0;
}
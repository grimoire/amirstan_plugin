#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>

#include "amir_cuda_util/common_util.h"
#include "amir_cuda_util/cuda_util.h"
#include "group_norm.h"

namespace amirstan {
namespace plugin {
using namespace amirstan::cuda;
template<typename T>
__global__ void group_norm_kernel(T*           output,
                                  const T*     input,
                                  size_t       input_size,
                                  int          batch_size,
                                  int          num_groups,
                                  int          num_channels,
                                  int          WH,
                                  T            eps,
                                  T*           mean,
                                  T*           var,
                                  const float* weight,
                                  const float* bias)
{
    CUDA_KERNEL_LOOP(i, input_size)
    {
        const int mean_var_index = i / (num_channels * WH / num_groups);
        const int axpy_index     = (i % (num_channels * WH)) / WH;
        T         ret            = (input[i] - mean[mean_var_index]) / sqrt(var[mean_var_index] + eps);
        ret                      = ret * T(weight[axpy_index]) + T(bias[axpy_index]);
        output[i]                = ret;
    }
}

template<typename T>
void compute_group_norm(T*           output,
                        const T*     input,
                        int          batch_size,
                        int          num_groups,
                        int          num_channels,
                        int          WH,
                        T            eps,
                        const float* weight,
                        const float* bias,
                        cudaStream_t stream,
                        void*        workspace)
{
    size_t word_size = sizeof(T);
    T*     mean      = reinterpret_cast<T*>(workspace);
    workspace =
        reinterpret_cast<char*>(workspace) + amirstan::common::getAlignedSize(batch_size * num_groups * word_size);
    T* var = reinterpret_cast<T*>(workspace);
    workspace =
        reinterpret_cast<char*>(workspace) + amirstan::common::getAlignedSize(batch_size * num_groups * word_size);
    int  mean_var_shape[2]       = {batch_size * num_groups, num_channels * WH / num_groups};
    bool mean_var_reduce_dims[2] = {false, true};

    amirstan::cuda::tensorMeanVar<T>(
        mean, var, input, &mean_var_shape[0], &mean_var_reduce_dims[0], 2, stream, workspace);

    size_t input_size = batch_size * num_channels * WH;

    group_norm_kernel<T><<<GET_BLOCKS(input_size), CUDA_NUM_THREADS, 0, stream>>>(
        output, input, input_size, batch_size, num_groups, num_channels, WH, eps, mean, var, weight, bias);
}

template void compute_group_norm<float>(float*       output,
                                        const float* input,
                                        int          batch_size,
                                        int          num_groups,
                                        int          num_channels,
                                        int          WH,
                                        float        eps,
                                        const float* weight,
                                        const float* bias,
                                        cudaStream_t stream,
                                        void*        workspace);

}  // namespace plugin
}  // namespace amirstan

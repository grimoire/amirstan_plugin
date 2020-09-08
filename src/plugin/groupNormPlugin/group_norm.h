#pragma once
#include <cuda_runtime.h>



namespace amirstan
{
namespace plugin
{
    template<typename T>
    void compute_group_norm(T* output, const T* input, 
    int batch_size, int num_groups, int num_channels, int WH,
    T eps,
     const float* weight,const float* bias,  cudaStream_t stream, void* workspace);
}
}
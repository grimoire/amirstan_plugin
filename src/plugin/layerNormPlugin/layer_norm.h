#pragma once
#include <cuda_runtime.h>



namespace amirstan
{
namespace plugin
{
    template<typename T>
    void compute_layer_norm(T* output, const T* input, 
    int norm_size, int layer_size,
    T eps,
     const T* weight,const T* bias,  cudaStream_t stream, void* workspace);
}
}
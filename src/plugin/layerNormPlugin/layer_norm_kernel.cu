#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <cuda_fp16.h>
#include "layer_norm.h"
#include "amir_cuda_util/cuda_util.h"

namespace amirstan
{
namespace plugin
{


    using namespace amirstan::cuda;
    template<typename T>
    __global__ void layer_norm_kernel(T* output,const T* input, size_t input_size,
        int norm_size, int layer_size,
        T eps, 
        T * mean,  T * var, const T* weight,const T* bias){
        CUDA_KERNEL_LOOP(i, input_size) {
            const int mean_var_index = i/layer_size;
            const int axpy_index = i%layer_size;
            T ret = (input[i]- mean[mean_var_index])/sqrt(var[mean_var_index]+eps);
            ret = ret*weight[axpy_index] + bias[axpy_index];
            output[i] = ret;
        }
    }

    template<typename T>
    void compute_layer_norm(T* output, const T* input, 
        int norm_size, int layer_size,
         T eps, 
        const T* weight,const T* bias,  cudaStream_t stream, void* workspace){
        T* mean = (T*)workspace;
        T* var = mean + norm_size;

        int mean_var_shape[2] = {norm_size, layer_size};
        bool mean_var_reduce_dims[2] = {false,true};

        amirstan::cuda::tensorMeanVar<T>(mean, var, input,
            &mean_var_shape[0], &mean_var_reduce_dims[0] , 2,
                stream, (void*)(var+norm_size));
        
        size_t input_size = norm_size * layer_size;

        layer_norm_kernel<T><<<GET_BLOCKS(input_size), CUDA_NUM_THREADS,0,stream>>>(output, input, input_size,
            norm_size, layer_size,
             eps,
             mean, var, weight, bias);
        
    }

    template void compute_layer_norm<float>(float* output, const float* input,
        int norm_size, int layer_size,
        float eps,
         const float* weight,const float* bias,  cudaStream_t stream, void* workspace);


    // template void compute_layer_norm<half>(half* output, const half* input,
    //         int norm_size, int layer_size,
    //         half eps,
    //          const half* weight,const half* bias,  cudaStream_t stream, void* workspace);
}
}
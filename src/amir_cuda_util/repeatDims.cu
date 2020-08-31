#include <algorithm>
#include <stdio.h>
#include <cuda_fp16.h>

#include "amir_cuda_util/cuda_util.h"

namespace amirstan
{
namespace cuda
{

    template<typename T>
    __global__ void repeat_dims_kernel(T* dst, const T* src, size_t dst_elems,
        TensorSize input_size, TensorStride input_stride, TensorStride output_stride, 
        int dims){

            size_t* src_stride = &(input_stride.size[0]);
            size_t* dst_stride = &(output_stride.size[0]);
            int* src_size = &(input_size.size[0]);
            CUDA_KERNEL_LOOP(index, dst_elems){
                size_t dst_index = index;
                size_t src_index = 0;
                
                for (int i = 0; i < dims; ++i)
                {
                    int dim_index = dst_index / dst_stride[i];
                    dst_index = dst_index % dst_stride[i];
                    src_index += (dim_index % src_size[i]) * src_stride[i];
                }
                dst[index] = src[src_index];
            }

    }

    
    template<typename T>
    void repeat_dims(T* dst, const T* src,const int *input_size, const int *repeatDims, int dims, cudaStream_t stream){
        TensorSize ts_input_size;
        TensorStride input_stride;
        TensorStride output_stride;

        memcpy(&ts_input_size.size[0], input_size, sizeof(int)*dims);
        input_stride.size[dims-1] = 1;
        output_stride.size[dims-1] = 1;
        
        for(int i=dims-2; i>=0; --i){
            input_stride.size[i] = input_stride.size[i+1] * ts_input_size.size[i+1];
            output_stride.size[i] = output_stride.size[i+1] * ts_input_size.size[i+1] * repeatDims[i+1];
        }

        size_t dst_elems = output_stride.size[0] * ts_input_size.size[0] * repeatDims[0];

        repeat_dims_kernel<T><<<GET_BLOCKS(dst_elems), CUDA_NUM_THREADS, 0, stream>>>
        (dst, src, dst_elems, ts_input_size, input_stride, output_stride, dims);
    }

    template void repeat_dims<float>(float* dst, const float* src,const int *input_dims, const int *repeatDims, int dims, cudaStream_t stream);
    template void repeat_dims<int>(int* dst, const int* src,const int *input_dims, const int *repeatDims, int dims, cudaStream_t stream);
    template void repeat_dims<half>(half* dst, const half* src,const int *input_dims, const int *repeatDims, int dims, cudaStream_t stream);
    template void repeat_dims<char>(char* dst, const char* src,const int *input_dims, const int *repeatDims, int dims, cudaStream_t stream);

}
}
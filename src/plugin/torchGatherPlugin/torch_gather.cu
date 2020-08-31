#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <cuda_fp16.h>

#include "torch_gather.h"
#include "amir_cuda_util/cuda_util.h"

namespace amirstan
{
namespace plugin
{

    using namespace amirstan::cuda;

    using amirstan::cuda::TensorSize;
    using amirstan::cuda::TensorStride;

    template <typename T>
    __global__ void torch_gather_kernel(T* dst, const T* src, const int* gather_table,
                                        int dim, TensorStride input_stride, TensorStride output_stride, int nb_dims,
                                    int num_output){
        size_t* src_stride = &(input_stride.size[0]);
        size_t* dst_stride = &(output_stride.size[0]);
        
        CUDA_KERNEL_LOOP(index, num_output){
            size_t dst_index = index;
            size_t src_index = 0;
            const int gather_value = gather_table[dst_index];
            for (int i = 0; i < nb_dims; ++i)
            {
                int dim_index = dst_index / dst_stride[i];
                dst_index = dst_index % dst_stride[i];
                if(i!=dim){
                    src_index+=dim_index*src_stride[i];
                }else{
                    src_index+=gather_value*src_stride[i];
                }

            }
            dst[index] = src[src_index];
        }
    }


    template <typename T>
    void torch_gather(T *output, const T* input, const int* index, 
                            int dim, int* input_dims, int *index_dims, int nb_dims,
                            cudaStream_t stream){
        
        TensorSize ts_input_size;
        TensorStride input_stride;

        memcpy(&ts_input_size.size[0], input_dims, sizeof(int)*nb_dims);
        input_stride.size[nb_dims-1] = 1;
        for(int i=nb_dims-2; i>=0; --i){
            input_stride.size[i] = input_stride.size[i+1] * ts_input_size.size[i+1];
        }

        TensorSize ts_output_size;
        TensorStride output_stride;
        memcpy(&ts_output_size.size[0], index_dims, sizeof(int)*nb_dims);
        output_stride.size[nb_dims-1] = 1;
        for(int i=nb_dims-2; i>=0; --i){
            output_stride.size[i] = output_stride.size[i+1] * ts_output_size.size[i+1];
        }
        
        size_t num_output = output_stride.size[0]*ts_output_size.size[0];

        torch_gather_kernel<T><<<GET_BLOCKS(num_output), CUDA_NUM_THREADS, 0, stream>>>(
            output, input, index, dim,
            input_stride, output_stride, nb_dims,
            num_output
        );
    }

    template void torch_gather<float>(float *output, const float* input, const int* index, 
        int dim, int* input_dims, int *index_dims, int nb_dims,
        cudaStream_t stream);

    template void torch_gather<half>(half *output, const half* input, const int* index, 
        int dim, int* input_dims, int *index_dims, int nb_dims,
        cudaStream_t stream);

        template void torch_gather<int>(int *output, const int* input, const int* index, 
            int dim, int* input_dims, int *index_dims, int nb_dims,
            cudaStream_t stream);

}
}
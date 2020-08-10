#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <cuda_fp16.h>

#include "torch_flip.h"
#include "amir_cuda_util/cuda_util.h"


namespace amirstan
{
namespace plugin
{
  using namespace amirstan::cuda;


  template <typename T>
  __global__ void torch_flip_kernel(T* output, const T* input,
    const TensorSize input_size, const TensorStride input_stride,
    const TensorSize flip_mask, int nb_dims, int count){
      
      const int* in_size = &(input_size.size[0]);
      const size_t* stride = &(input_stride.size[0]);
      const int* mask = &(flip_mask.size[0]);
      
      CUDA_KERNEL_LOOP(index, count){

        size_t dst_index = index;
        size_t src_index = 0;
        for (int i = 0; i < nb_dims; ++i)
        {
            int dim_index = dst_index / stride[i];
            dst_index = dst_index % stride[i];
            
            if(mask[i]>0){
              dim_index = in_size[i]-1-dim_index;
            }

            src_index += dim_index * stride[i];
        }

        output[index] = input[src_index];
      }

  }


  static void create_size_stride(const int* dims, int nb_dims, TensorSize &size, TensorStride& stride){
        memcpy(&size.size[0], dims, sizeof(int)*nb_dims);
        stride.size[nb_dims-1] = 1;
        for(int i=nb_dims-2; i>=0; --i){
            stride.size[i] = stride.size[i+1] * size.size[i+1];
        }
  }



  template <typename T>
  void torch_flip(T *output, const T* input,
                  int* input_dims, int nb_dims,
                  int* flip_dims, int nb_flip_dims,
                  cudaStream_t stream){
        
        TensorSize ts_input_size;
        TensorStride input_stride;
        create_size_stride(input_dims, nb_dims, ts_input_size, input_stride);

        int count = ts_input_size.size[0];
        for(int i=1; i<nb_dims; ++i){
            count*=ts_input_size.size[i];
        }

        TensorSize flip_mask;
        for(int i=0;i<nb_dims;++i){
          flip_mask.size[i]=0;
        }
        for(int i=0;i<nb_flip_dims;++i){
          flip_mask.size[flip_dims[i]]=1;
        }

        torch_flip_kernel<T><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(output, input,
                                                                                ts_input_size, input_stride,
                                                                              flip_mask, nb_dims, count);
    }


  template void torch_flip<float>(float *output, const float* input,
                  int* input_dims, int nb_dims,
                  int* flip_dims, int nb_flip_dims,
                  cudaStream_t stream);
    
}   // namespace plugin
}   // namespace amirstan
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include "torch_cum.h"
#include "amir_cuda_util/cuda_util.h"


namespace amirstan
{
namespace plugin
{
  using namespace amirstan::cuda;

template<typename T>
struct CumScanProd
{
  __host__ __device__ __forceinline__ T operator()(const T& a, const T& b)
  {
    return a*b;
  }
};


template <typename T>
__global__ void torch_cum_warp_kernel(T* output, const T* input,
                                        size_t stride, int dim_size, size_t cum_size
                                      ,const int cum_type){
  
  // create block scan
  typedef cub::WarpScan<T> warpScan;
  __shared__ union {
    typename warpScan::TempStorage     scan[CUDA_NUM_WARP];
  } temp_storage;

  for (int index = (blockIdx.x*CUDA_NUM_WARP)+int(threadIdx.x/CUDA_WARP_SIZE); index < cum_size; index += gridDim.x*CUDA_NUM_WARP){
    // compute cum start
    const size_t pre_index = index/stride;
    const size_t post_index = index%stride;

    const size_t cum_start = pre_index*stride*dim_size + post_index;

    T aggregate_value=(T)0;
    if(cum_type==1){
      aggregate_value = (T)1;
    }

    for(int warp_offset = 0; warp_offset<dim_size; warp_offset += CUDA_WARP_SIZE){
      const size_t cum_position = warp_offset + threadIdx.x%CUDA_WARP_SIZE;
      T thread_data = cum_position<dim_size? input[cum_start+cum_position*stride]:0;
      if(cum_type==0){
        if(threadIdx.x%CUDA_WARP_SIZE==0){
          thread_data = thread_data + aggregate_value;
        }
        warpScan(temp_storage.scan[int(threadIdx.x/CUDA_WARP_SIZE)]).InclusiveSum(thread_data, thread_data, aggregate_value);
      }else{
        if(threadIdx.x%CUDA_WARP_SIZE==0){
          thread_data = thread_data * aggregate_value;
        }
        warpScan(temp_storage.scan[int(threadIdx.x/CUDA_WARP_SIZE)]).InclusiveScan(thread_data, thread_data, CumScanProd<T>(), aggregate_value);

      }

      // Store scanned items to output segment
      if(cum_position<dim_size){
        output[cum_start+cum_position*stride] = thread_data;
      }
    }

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
  void torch_cum(T *output, const T* input,
                  int* input_dims, int nb_dims,
                  int cum_dim, int cum_type,
                  cudaStream_t stream){

    TensorSize ts_input_size;
    TensorStride input_stride;
    create_size_stride(input_dims, nb_dims, ts_input_size, input_stride);

    size_t cum_size = 1;
    for(int i=0; i<nb_dims; ++i){
      if(i!=cum_dim){
        cum_size*=ts_input_size.size[i];
      }
    }

    size_t num_blocks = std::min<long>(kMaxGridNum,(cum_size+CUDA_NUM_WARP-1)/CUDA_NUM_WARP);
    torch_cum_warp_kernel<T><<<num_blocks, CUDA_NUM_THREADS, 0, stream>>>(output, input, 
                                                                          input_stride.size[cum_dim], ts_input_size.size[cum_dim], cum_size,
                                                                          cum_type);

  }


  template void torch_cum<float>(float *output, const float* input,
    int* input_dims, int nb_dims,
    int cum_dim, int cum_type,
    cudaStream_t stream);
    
}   // namespace plugin
}   // namespace amirstan
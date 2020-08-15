#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include "torch_cum_maxmin.h"
#include "amir_cuda_util/cuda_util.h"


namespace amirstan
{
namespace plugin
{
  using namespace amirstan::cuda;


template<typename T, bool is_max> struct BlockPrefixPairCumCallbackOp;

template<typename T>
struct BlockPrefixPairCumCallbackOp<T,true>
{
    // Running prefix
    cub::KeyValuePair<int, T> running_total;
    // Constructor
    __device__ BlockPrefixPairCumCallbackOp(cub::KeyValuePair<int, T> running_total) : running_total(running_total) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ cub::KeyValuePair<int, T> operator()(cub::KeyValuePair<int, T> block_aggregate)
    {
        cub::KeyValuePair<int, T> old_prefix = running_total;
        running_total = (block_aggregate.value > old_prefix.value) ? block_aggregate : old_prefix;
        return old_prefix;
    }
};

template<typename T>
struct BlockPrefixPairCumCallbackOp<T,false>
{
    // Running prefix
    cub::KeyValuePair<int, T> running_total;
    // Constructor
    __device__ BlockPrefixPairCumCallbackOp(cub::KeyValuePair<int, T> running_total) : running_total(running_total) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ cub::KeyValuePair<int, T> operator()(cub::KeyValuePair<int, T> block_aggregate)
    {
        cub::KeyValuePair<int, T> old_prefix = running_total;
        running_total = (block_aggregate.value < old_prefix.value) ? block_aggregate : old_prefix;
        return old_prefix;
    }
};

template <typename T>
__global__ void torch_cum_maxmin_warp_kernel(T* output, int* out_index, const T* input,
                                        size_t stride, int dim_size, size_t cum_size
                                      ,const int cum_type){
  
  // create block scan
  typedef cub::WarpScan<cub::KeyValuePair<int, T>> warpScan;
  __shared__ union {
    typename warpScan::TempStorage     scan[CUDA_NUM_WARP];
  } temp_storage;

  for (int index = (blockIdx.x*CUDA_NUM_WARP)+int(threadIdx.x/CUDA_WARP_SIZE); index < cum_size; index += gridDim.x*CUDA_NUM_WARP){
    // compute cum start
    const size_t pre_index = index/stride;
    const size_t post_index = index%stride;

    const size_t cum_start = pre_index*stride*dim_size + post_index;

    cub::KeyValuePair<int, T> aggregate_value{0, input[cum_start]};

    for(int warp_offset = 0; warp_offset<dim_size; warp_offset += CUDA_WARP_SIZE){
      const size_t cum_position = warp_offset + threadIdx.x%CUDA_WARP_SIZE;
      cub::KeyValuePair<int, T> thread_data = {cum_position, cum_position<dim_size? input[cum_start+cum_position*stride]:0};
      if(cum_type==0){
        thread_data = thread_data.value>aggregate_value.value? thread_data:aggregate_value;
        warpScan(temp_storage.scan[int(threadIdx.x/CUDA_WARP_SIZE)]).InclusiveScan(thread_data, thread_data, cub::ArgMax(), aggregate_value);
      }else{
        thread_data = thread_data.value<aggregate_value.value? thread_data:aggregate_value;
        warpScan(temp_storage.scan[int(threadIdx.x/CUDA_WARP_SIZE)]).InclusiveScan(thread_data, thread_data, cub::ArgMin(), aggregate_value);

      }

      // Store scanned items to output segment
      if(cum_position<dim_size){
        output[cum_start+cum_position*stride] = thread_data.value;
        out_index[cum_start+cum_position*stride] = thread_data.key;
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
  void torch_cum_maxmin(T *output, int *index, const T* input,
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
    torch_cum_maxmin_warp_kernel<T><<<num_blocks, CUDA_NUM_THREADS, 0, stream>>>(output, index, input, 
                                                                                  input_stride.size[cum_dim], ts_input_size.size[cum_dim], cum_size,
                                                                                  cum_type);

  }


  template void torch_cum_maxmin<float>(float *output, int *index, const float* input,
    int* input_dims, int nb_dims,
    int cum_dim, int cum_type,
    cudaStream_t stream);
    
}   // namespace plugin
}   // namespace amirstan
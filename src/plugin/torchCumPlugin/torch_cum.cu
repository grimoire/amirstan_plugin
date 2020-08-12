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

  static const int DATA_PER_THREAD=1;



template<typename T>
struct BlockPrefixCumSumCallbackOp
{
    // Running prefix
    T running_total;
    // Constructor
    __device__ BlockPrefixCumSumCallbackOp(T running_total) : running_total(running_total) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ T operator()(T block_aggregate)
    {
        T old_prefix = running_total;
        running_total = block_aggregate + old_prefix;
        return old_prefix;
    }
};

template<typename T>
struct BlockPrefixCumProdCallbackOp
{
    // Running prefix
    T running_total;
    // Constructor
    __device__ BlockPrefixCumProdCallbackOp(T running_total) : running_total(running_total) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ T operator()(T block_aggregate)
    {
        T old_prefix = running_total;
        running_total = block_aggregate * old_prefix;
        return old_prefix;
    }
};

template<typename T>
struct CumScanProd
{
  __host__ __device__ __forceinline__ T operator()(const T& a, const T& b)
  {
    return a*b;
  }
};


  template <typename T>
  __global__ void torch_cum_kernel(T* output, const T* input,
                                    size_t stride, int dim_size, size_t cum_size
                                  ,const int cum_type){
    
    // create block scan
    typedef cub::BlockScan<T, CUDA_NUM_THREADS> BlockScan;
    __shared__ union {
      typename BlockScan::TempStorage     scan;
    } temp_storage;

    for (int index = blockIdx.x; index < cum_size; index += gridDim.x){
      // compute cum start
      const size_t pre_index = index/stride;
      const size_t post_index = index%stride;

      const size_t cum_start = pre_index*stride*dim_size + post_index;
      
      BlockPrefixCumSumCallbackOp<T> sum_prefix_op((T)0);
      BlockPrefixCumProdCallbackOp<T> prod_prefix_op((T)1);


      for (int block_offset = 0; block_offset < cum_size; block_offset += CUDA_NUM_THREADS * DATA_PER_THREAD)
      {
          // Load a segment of consecutive items that are blocked across threads
          T thread_data[DATA_PER_THREAD];
          #pragma unroll
          for(int i=0;i<DATA_PER_THREAD;++i){
              const size_t cum_position = block_offset + threadIdx.x + i;
              thread_data[i] = input[cum_start+cum_position*stride];
          }
          // BlockLoad(temp_storage.load).Load(input + block_offset, thread_data);
          
          cub::CTA_SYNC();
          // Collectively compute the block-wide inclusive prefix max scan
          if(cum_type==0){
            // BlockScan(temp_storage.scan).InclusiveScan(
            //     thread_data, thread_data, cub::Sum(), sum_prefix_op);
            BlockScan(temp_storage.scan).InclusiveSum(thread_data, thread_data);
          }else{
            BlockScan(temp_storage.scan).InclusiveScan(
                thread_data, thread_data, CumScanProd<T>(), prod_prefix_op);
          }
          cub::CTA_SYNC();
          // Store scanned items to output segment
          #pragma unroll
          for(int i=0;i<DATA_PER_THREAD;++i){
              const size_t cum_position = block_offset + threadIdx.x + i;
              if(cum_position<dim_size){
                output[cum_start+cum_position*stride] = thread_data[i];
              }
          }
          cub::CTA_SYNC();
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

    torch_cum_kernel<T><<<GET_BLOCKS(cum_size), CUDA_NUM_THREADS, 0, stream>>>(output, input, 
                                                                                      input_stride.size[cum_dim], ts_input_size.size[cum_dim], cum_size,
                                                                                      cum_type);

  }


  template void torch_cum<float>(float *output, const float* input,
    int* input_dims, int nb_dims,
    int cum_dim, int cum_type,
    cudaStream_t stream);
    
}   // namespace plugin
}   // namespace amirstan
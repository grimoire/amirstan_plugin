#include <utility>
#include <stdio.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include "amir_cuda_util/cuda_util.h"
#include "reduceUtils.cuh"

namespace amirstan
{
namespace cuda
{

template <typename T>
struct unary_linear_index_to_reduce_index : public thrust::unary_function<T,T> {
  int nCols;
  
	__host__ __device__ unary_linear_index_to_reduce_index(int nCols) : nCols(nCols){}

	__host__ __device__ T operator()(T i) { 
    return i/nCols; 
  }
};

template <typename T>
struct unary_divide_by_scalar: public thrust::unary_function<T,T>
{
    const T a;

    unary_divide_by_scalar(T _a) : a(_a) {}

    __host__ __device__ T operator()(const T& x) const { 
      return x/a;
    }
};

template <typename T>
void tensorMean(T *dst, T *src, int* src_size, bool *reduce_dims, int dims, cudaStream_t stream, void* workspace){
  
  size_t src_length = 1;
  size_t dst_length = 1;
  size_t mean_length =1;
  int num_keep = 0;

  for (int i = 0;i< dims;++i)
  { 
    src_length*=src_size[i];

    if (!reduce_dims[i]){
      dst_length*=src_size[i];
      num_keep += 1;
    }else{
      mean_length*=src_size[i];
    }
  }

// compute permute dims
  bool need_permute = false;
  int keep_start = 0;
  int reduce_start = num_keep;

  int *permute = new int(dims);
  for(int i=0;i<dims;++i){
    if(!reduce_dims[i]){
        permute[keep_start++] = i;
        if(i>=num_keep){
            need_permute=true;
        }
    }else{
        permute[reduce_start++] = i;
    }
  }

  // create working memory
  T* permute_src;
  if(need_permute){
    if(workspace){
        permute_src = (T*)workspace;
    }else{
        cudaMalloc(&permute_src, src_length*sizeof(T));
    }
    memcpyPermute<T>(permute_src, src, src_size, permute, dims, stream);
  }else{
      permute_src = src;
  }

  reduce2DContigous<T>(dst, permute_src, (T)0., src_length, dst_length, 
    thrust::identity<T>{}, reduceSumOp<T>(), divScalarOp<T>(mean_length),
     stream);

  if(need_permute && workspace==nullptr){
      cudaFree(permute_src);
  }

  delete[] permute;
}


template void tensorMean<float>(float *dst, float *src, int* src_size, bool *reduce_dims, int dims, cudaStream_t stream, void* workspace);


// tensorMeanVar

template <typename T>
struct subMeanSquareOp {
    const T *mean;
    subMeanSquareOp(const T* mean):mean(mean){}
    inline __device__ T operator()(const T& x){
      const size_t slice_id = getLinearBlockId();
      const T sub_mean = x-mean[slice_id];
      return sub_mean*sub_mean;
}
};

// template <typename T>
// __global__ void subMeanSquareInplace(T *src, const T* mean, int src_size, int nCols){
//   CUDA_KERNEL_LOOP(i, src_size){
//     const int mean_id = i/nCols;
//     const T sub_mean = src[i] - mean[mean_id];
//     src[i] = sub_mean*sub_mean;
//   }
// }


template <typename T>
void tensorMeanVar(T *mean_dst, T* var_dst, const T *src, int* src_size, bool *reduce_dims, int dims, cudaStream_t stream, void* workspace){
    
  size_t src_length = 1;
  size_t dst_length = 1;
  size_t mean_length =1;
  int num_keep = 0;

  for (int i = 0;i< dims;++i)
  { 
    src_length*=src_size[i];

    if (!reduce_dims[i]){
      dst_length*=src_size[i];
      num_keep += 1;
    }else{
      mean_length*=src_size[i];
    }
  }

// compute permute dims
  bool need_permute = false;
  int keep_start = 0;
  int reduce_start = num_keep;

  int *permute = new int(dims);
  for(int i=0;i<dims;++i){
    if(!reduce_dims[i]){
        permute[keep_start++] = i;
        if(i>=num_keep){
            need_permute=true;
        }
    }else{
        permute[reduce_start++] = i;
    }
  }

  // create working memory
  T* permute_src;
  if(workspace){
      permute_src = (T*)workspace;
  }else{
      cudaMalloc(&permute_src, src_length*sizeof(T));
  }

  if(need_permute){
    memcpyPermute<T>(permute_src, src, src_size, permute, dims, stream);

    reduce2DContigous<T>(mean_dst, permute_src, (T)0., src_length, dst_length, 
      thrust::identity<T>{}, reduceSumOp<T>(), divScalarOp<T>(mean_length),
       stream);
  
    reduce2DContigous<T>(var_dst, permute_src, (T)0., src_length, dst_length, 
    subMeanSquareOp<T>(mean_dst), reduceSumOp<T>(), divScalarOp<T>(mean_length),
       stream);
  }else{
    
    reduce2DContigous<T>(mean_dst, src, (T)0., src_length, dst_length, 
      thrust::identity<T>{}, reduceSumOp<T>(), divScalarOp<T>(mean_length),
       stream);
  
    reduce2DContigous<T>(var_dst, src, (T)0., src_length, dst_length, 
    subMeanSquareOp<T>(mean_dst), reduceSumOp<T>(), divScalarOp<T>(mean_length),
       stream);
    // cudaMemcpyAsync(permute_src, src, src_length*sizeof(T), cudaMemcpyDeviceToDevice, stream);  
    // permute_src = src;
  }


  if(need_permute && workspace==nullptr){
      cudaFree(permute_src);
  }

  delete[] permute;
}

template void tensorMeanVar<float>(float *mean_dst, float *vat_dst, const float *src,
   int* src_size, bool *reduce_dims, int dims, cudaStream_t stream, void* workspace);

}
}

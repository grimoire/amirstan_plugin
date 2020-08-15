#pragma once
#include <cuda_runtime.h>

namespace amirstan
{
namespace cuda
{

#define CUDA_KERNEL_LOOP(i, n)                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)

#define cudaCheckError() { \
  cudaError_t e=cudaGetLastError(); \
  if(e!=cudaSuccess) { \
  printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  exit(0); \
  } \
  }
  
const int CUDA_NUM_THREADS = 512;
const int CUDA_WARP_SIZE=32;
const int CUDA_NUM_WARP=CUDA_NUM_THREADS/float(CUDA_WARP_SIZE);
const int kMaxGridNum = 65535;
inline int GET_BLOCKS(const int N)
{
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

struct TensorSize{
    int size[8];
    int dims;
};

struct TensorStride{
    size_t size[8];
    int dims;
};

template <class value_type>
void memcpyPermute(value_type *dst,const value_type *src, int *src_size, int *permute, int src_dim, cudaStream_t stream=0);

template <typename T>
void tensorMean(T *dst, T *src, int* src_size, bool *reduce_dims, int dims, cudaStream_t stream=0, void* workspace=nullptr);

template <typename T>
void tensorMeanVar(T *mean_dst, T* var_dst,const T *src, int* src_size, bool *reduce_dims, int dims, cudaStream_t stream=0, void* workspace=nullptr);

template<typename T>
void repeat_dims(T* dst, const T* src,const int *input_size, const int *repeatDims, int dims, cudaStream_t stream=0);
} // namespace cuda

} // namespace amirstan
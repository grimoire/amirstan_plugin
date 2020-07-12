#include <algorithm>
#include <stdio.h>
#include <iostream>

#include "amir_cuda_util/cuda_util.h"


namespace amirstan
{
namespace cuda
{


template <class value_type>
__global__ void copy_permute_kernel(value_type *dst, const value_type *src, int n, 
  TensorSize ts_src_stride, TensorSize ts_dst_stride, TensorSize ts_permute, int src_dim)
{
  int* src_stride = &(ts_src_stride.size[0]);
  int* dst_stride = &(ts_dst_stride.size[0]);
  int* permute = &(ts_permute.size[0]);
  CUDA_KERNEL_LOOP(index, n)
  { 
      size_t dst_index = index;
      size_t src_index = 0;
      for (int i = 0; i < src_dim; ++i)
      {
          int dim_index = dst_index / dst_stride[i];
          dst_index = dst_index % dst_stride[i];
          src_index += dim_index * src_stride[permute[i]];
      }
      dst[index] = src[src_index];
  }
}

template <class value_type>
void memcpyPermute(value_type *dst,const value_type *src, int *src_size, int *permute, int src_dim, cudaStream_t stream)
{
  size_t copy_size = 1;
  TensorSize ts_permute;
  memcpy(&(ts_permute.size[0]), permute, src_dim *sizeof(int));

  TensorSize ts_src_stride;
  TensorSize ts_dst_stride;
  TensorSize ts_dst_size;
  int *src_stride = &(ts_src_stride.size[0]);
  int *dst_stride = &(ts_dst_stride.size[0]);
  int *dst_size = &(ts_dst_size.size[0]);
  src_stride[src_dim - 1] = 1;
  dst_stride[src_dim - 1] = 1;
  
  for (int i = src_dim - 1; i >= 0; --i)
  {
      dst_size[i] = src_size[permute[i]];
      if (i < src_dim - 1)
      {
        src_stride[i] = src_stride[i + 1] * src_size[i + 1];
      }
  }
  
  for (int i = src_dim - 1; i >= 0; --i)
  {
      copy_size *= dst_size[i];
      if (i < src_dim - 1)
      {
        dst_stride[i] = dst_stride[i + 1] * dst_size[i + 1];
      }
  }

  copy_permute_kernel<value_type><<<GET_BLOCKS(copy_size), CUDA_NUM_THREADS, 0, stream>>> 
  (dst, src, copy_size, 
    ts_src_stride, ts_dst_stride, ts_permute, src_dim);
  
}

template void memcpyPermute<float>(float *dst,const float *src, int *src_size, int *permute, int src_dim, cudaStream_t stream);

} // namespace cuda

} // namespace amirstan
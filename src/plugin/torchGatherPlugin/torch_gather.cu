#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>

#include "amir_cuda_util/cuda_util.h"
#include "torch_gather.h"

namespace amirstan {
namespace plugin {

using namespace amirstan::cuda;

using amirstan::cuda::TensorSize;
using amirstan::cuda::TensorStride;

template <typename T, int nb_dims>
__global__ void torch_gather_kernel(T *__restrict__ dst,
                                    const T *__restrict__ src,
                                    const int *__restrict__ gather_table,
                                    int dim, TensorStride input_stride,
                                    TensorStride output_stride,
                                    int num_output) {
  size_t *__restrict__ src_stride = &(input_stride.size[0]);
  size_t *__restrict__ dst_stride = &(output_stride.size[0]);

  CUDA_KERNEL_LOOP(index, num_output) {
    int dst_index = index;
    int src_index = 0;
    const int gather_value = gather_table[dst_index];
#pragma unroll
    for (int i = 0; i < nb_dims; ++i) {
      const int dst_stride_i = dst_stride[i];
      const int src_stride_i = src_stride[i];
      int dim_index = dst_index / dst_stride_i;
      dst_index = dst_index % dst_stride_i;
      src_index += ((i != dim) ? dim_index : gather_value) * src_stride_i;
    }
    dst[index] = src[src_index];
  }
}

template <typename T>
void torch_gather(T *output, const T *input, const int *index, int dim,
                  int *input_dims, int *index_dims, int nb_dims,
                  cudaStream_t stream) {
  TensorSize ts_input_size;
  TensorStride input_stride;

  memcpy(&ts_input_size.size[0], input_dims, sizeof(int) * nb_dims);
  input_stride.size[nb_dims - 1] = 1;
  for (int i = nb_dims - 2; i >= 0; --i) {
    input_stride.size[i] = input_stride.size[i + 1] * ts_input_size.size[i + 1];
  }

  TensorSize ts_output_size;
  TensorStride output_stride;
  memcpy(&ts_output_size.size[0], index_dims, sizeof(int) * nb_dims);
  output_stride.size[nb_dims - 1] = 1;
  for (int i = nb_dims - 2; i >= 0; --i) {
    output_stride.size[i] =
        output_stride.size[i + 1] * ts_output_size.size[i + 1];
  }

  size_t num_output = output_stride.size[0] * ts_output_size.size[0];

  switch (nb_dims) {
    case 1:
      torch_gather_kernel<T, 1>
          <<<GET_BLOCKS(num_output), CUDA_NUM_THREADS, 0, stream>>>(
              output, input, index, dim, input_stride, output_stride,
              num_output);
      break;
    case 2:
      torch_gather_kernel<T, 2>
          <<<GET_BLOCKS(num_output), CUDA_NUM_THREADS, 0, stream>>>(
              output, input, index, dim, input_stride, output_stride,
              num_output);
      break;
    case 3:
      torch_gather_kernel<T, 3>
          <<<GET_BLOCKS(num_output), CUDA_NUM_THREADS, 0, stream>>>(
              output, input, index, dim, input_stride, output_stride,
              num_output);
      break;
    case 4:
      torch_gather_kernel<T, 4>
          <<<GET_BLOCKS(num_output), CUDA_NUM_THREADS, 0, stream>>>(
              output, input, index, dim, input_stride, output_stride,
              num_output);
      break;
    case 5:
      torch_gather_kernel<T, 5>
          <<<GET_BLOCKS(num_output), CUDA_NUM_THREADS, 0, stream>>>(
              output, input, index, dim, input_stride, output_stride,
              num_output);
      break;
    case 6:
      torch_gather_kernel<T, 6>
          <<<GET_BLOCKS(num_output), CUDA_NUM_THREADS, 0, stream>>>(
              output, input, index, dim, input_stride, output_stride,
              num_output);
      break;
    case 7:
      torch_gather_kernel<T, 7>
          <<<GET_BLOCKS(num_output), CUDA_NUM_THREADS, 0, stream>>>(
              output, input, index, dim, input_stride, output_stride,
              num_output);
      break;
    case 8:
      torch_gather_kernel<T, 8>
          <<<GET_BLOCKS(num_output), CUDA_NUM_THREADS, 0, stream>>>(
              output, input, index, dim, input_stride, output_stride,
              num_output);
      break;
    case 9:
      torch_gather_kernel<T, 9>
          <<<GET_BLOCKS(num_output), CUDA_NUM_THREADS, 0, stream>>>(
              output, input, index, dim, input_stride, output_stride,
              num_output);
      break;
    case 10:
      torch_gather_kernel<T, 10>
          <<<GET_BLOCKS(num_output), CUDA_NUM_THREADS, 0, stream>>>(
              output, input, index, dim, input_stride, output_stride,
              num_output);
      break;
    default:
      break;
  }
}

template void torch_gather<float>(float *output, const float *input,
                                  const int *index, int dim, int *input_dims,
                                  int *index_dims, int nb_dims,
                                  cudaStream_t stream);

template void torch_gather<half>(half *output, const half *input,
                                 const int *index, int dim, int *input_dims,
                                 int *index_dims, int nb_dims,
                                 cudaStream_t stream);

template void torch_gather<int>(int *output, const int *input, const int *index,
                                int dim, int *input_dims, int *index_dims,
                                int nb_dims, cudaStream_t stream);

}  // namespace plugin
}  // namespace amirstan

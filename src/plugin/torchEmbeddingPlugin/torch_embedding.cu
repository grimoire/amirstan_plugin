#include <stdio.h>

#include <algorithm>
#include <cmath>

#include "amir_cuda_util/cuda_util.h"
#include "torch_embedding.h"

namespace amirstan {
namespace plugin {
using namespace amirstan::cuda;

template <typename T>
__global__ void torch_embedding_kernel(T* output, const int* index,
                                       const T* weight, int num_indices,
                                       int embedding_dim) {
  CUDA_KERNEL_LOOP(i, num_indices) {
    const int pos = index[i];
    T* output_offset = output + i * embedding_dim;
    const T* weight_offset = weight + pos * embedding_dim;
    memcpy(output_offset, weight_offset, embedding_dim * sizeof(T));
  }
}

template <typename T>
void torch_embedding(T* output, const int* index, const T* weight,
                     int num_indices, int embedding_dim, cudaStream_t stream) {
  torch_embedding_kernel<<<GET_BLOCKS(num_indices), CUDA_NUM_THREADS, 0,
                           stream>>>(output, index, weight, num_indices,
                                     embedding_dim);
}

template void torch_embedding<float>(float* output, const int* index,
                                     const float* weight, int num_indices,
                                     int embedding_dim, cudaStream_t stream);

}  // namespace plugin
}  // namespace amirstan
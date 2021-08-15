#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>

#include "amir_cuda_util/cuda_util.h"
#include "torch_unfold.h"

namespace amirstan {
namespace plugin {

using namespace amirstan::cuda;

using amirstan::cuda::TensorSize;
using amirstan::cuda::TensorStride;

template <typename T>
__global__ void im2col_kernel(
    const int64_t n, const T *data_im, const int64_t height,
    const int64_t width, const int64_t kernel_height,
    const int64_t kernel_width, const int64_t pad_height,
    const int64_t pad_width, const int64_t stride_height,
    const int64_t stride_width, const int64_t dilation_height,
    const int64_t dilation_width, const int64_t height_col,
    const int64_t width_col, T *data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int64_t w_out = index % width_col;

    int64_t idx = index / width_col;

    int64_t h_out = idx % height_col;
    int64_t channel_in = idx / height_col;
    int64_t channel_out = channel_in * kernel_height * kernel_width;
    int64_t h_in = h_out * stride_height - pad_height;
    int64_t w_in = w_out * stride_width - pad_width;

    T *col = data_col + (channel_out * height_col + h_out) * width_col + w_out;
    const T *im = data_im + (channel_in * height + h_in) * width + w_in;

    for (int64_t i = 0; i < kernel_height; ++i) {
      for (int64_t j = 0; j < kernel_width; ++j) {
        int64_t h = h_in + i * dilation_height;
        int64_t w = w_in + j * dilation_width;
        *col = (h >= 0 && w >= 0 && h < height && w < width)
                   ? im[i * dilation_height * width + j * dilation_width]
                   : T(0);
        col += height_col * width_col;
      }
    }
  }
}

template <typename T>
void torch_unfold(T *output, const T *input, const int64_t channels,
                  const int64_t height, const int64_t width,
                  const int64_t height_col, const int64_t width_col,
                  const int64_t kernel_height, const int64_t kernel_width,
                  const int64_t pad_height, const int64_t pad_width,
                  const int64_t stride_height, const int64_t stride_width,
                  const int64_t dilation_height, const int64_t dilation_width,
                  cudaStream_t stream) {
  int64_t num_kernels = channels * height_col * width_col;

  im2col_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, input, height, width, kernel_height, kernel_width,
      pad_height, pad_width, stride_height, stride_width, dilation_height,
      dilation_width, height_col, width_col, output);
}

template void torch_unfold<float>(
    float *output, const float *input, const int64_t channels,
    const int64_t height, const int64_t width, const int64_t height_col,
    const int64_t width_col, const int64_t kernel_height,
    const int64_t kernel_width, const int64_t pad_height,
    const int64_t pad_width, const int64_t stride_height,
    const int64_t stride_width, const int64_t dilation_height,
    const int64_t dilation_width, cudaStream_t stream);

template void torch_unfold<int32_t>(
    int32_t *output, const int32_t *input, const int64_t channels,
    const int64_t height, const int64_t width, const int64_t height_col,
    const int64_t width_col, const int64_t kernel_height,
    const int64_t kernel_width, const int64_t pad_height,
    const int64_t pad_width, const int64_t stride_height,
    const int64_t stride_width, const int64_t dilation_height,
    const int64_t dilation_width, cudaStream_t stream);

}  // namespace plugin
}  // namespace amirstan

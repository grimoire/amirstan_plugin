#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>

#include "amir_cuda_util/cuda_util.h"
#include "grid_anchor_dynamic.h"

namespace amirstan {
namespace plugin {
using namespace amirstan::cuda;
template <typename T>
__global__ void grid_anchor_dynamic_kernel(T *__restrict__ output,
                                           const T *__restrict__ base_anchor,
                                           int width, int height, int stride,
                                           int num_base_anchor) {
  CUDA_KERNEL_LOOP(i, width * height * num_base_anchor) {
    const int y = i / (width * num_base_anchor);
    const int x = (i % (width * num_base_anchor)) / num_base_anchor;
    const int base_id = i % num_base_anchor;

    output[i * 4 + 0] = base_anchor[base_id * 4 + 0] + x * stride;
    output[i * 4 + 1] = base_anchor[base_id * 4 + 1] + y * stride;
    output[i * 4 + 2] = base_anchor[base_id * 4 + 2] + x * stride;
    output[i * 4 + 3] = base_anchor[base_id * 4 + 3] + y * stride;
  }
}

template <>
__global__ void grid_anchor_dynamic_kernel<float>(
    float *__restrict__ output, const float *__restrict__ base_anchor,
    int width, int height, int stride, int num_base_anchor) {
  CUDA_KERNEL_LOOP(i, width * height * num_base_anchor) {
    const int y = i / (width * num_base_anchor);
    const int x = (i % (width * num_base_anchor)) / num_base_anchor;
    const int base_id = i % num_base_anchor;

    const float4 base_anchor_val =
        reinterpret_cast<const float4 *>(base_anchor)[base_id];

    const float stride_f = float(stride);
    const float x_f = float(x);
    const float y_f = float(y);
    float4 out_val;
    out_val.x = fma(x_f, stride_f, base_anchor_val.x);
    out_val.y = fma(y_f, stride_f, base_anchor_val.y);
    out_val.z = fma(x_f, stride_f, base_anchor_val.z);
    out_val.w = fma(y_f, stride_f, base_anchor_val.w);

    reinterpret_cast<float4 *>(output)[i] = out_val;
  }
}

template <typename T>
void grid_anchor_dynamic(T *output, const T *base_anchor, int width, int height,
                         int stride, int num_base_anchor, cudaStream_t stream) {
  size_t input_size = num_base_anchor * height * width;
  grid_anchor_dynamic_kernel<T>
      <<<GET_BLOCKS(input_size), CUDA_NUM_THREADS, 0, stream>>>(
          output, base_anchor, width, height, stride, num_base_anchor);
}

template void grid_anchor_dynamic<float>(float *output,
                                         const float *base_anchor, int width,
                                         int height, int stride,
                                         int num_base_anchor,
                                         cudaStream_t stream);

}  // namespace plugin
}  // namespace amirstan

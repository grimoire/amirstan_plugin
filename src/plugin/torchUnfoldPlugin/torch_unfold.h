#pragma once

namespace amirstan {
namespace plugin {

template <typename T>
void torch_unfold(T* output, const T* input, const int64_t channels,
                  const int64_t height, const int64_t width,
                  const int64_t height_col, const int64_t width_col,
                  const int64_t kernel_height, const int64_t kernel_width,
                  const int64_t pad_height, const int64_t pad_width,
                  const int64_t stride_height, const int64_t stride_width,
                  const int64_t dilation_height, const int64_t dilation_width,
                  cudaStream_t stream);

}
}  // namespace amirstan

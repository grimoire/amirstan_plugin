#pragma once
#include <cuda_runtime.h>

namespace amirstan {
namespace plugin {

template <typename T_cls, typename T_bbox>
void fill_zeros(T_cls* out_cls, T_bbox* out_bbox, size_t num_cls_element,
                size_t num_bbox_element, cudaStream_t stream);

template <typename T>
void delta2bbox(T* out_bbox, const T* in_bbox, const T* anchor,
                const int* clip_range, int batch_size, int num_outbbox,
                int num_classes, float* mean, float* std, cudaStream_t stream);

}  // namespace plugin
}  // namespace amirstan

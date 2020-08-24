#pragma once
#include <cuda_runtime.h>



namespace amirstan
{
namespace plugin
{

    template<typename T>
    void delta2bbox(T* out_cls, T* out_bbox,
                    const T* in_cls, const T* in_bbox, const T* anchor, const int* clip_range,
                    int batch_size, int num_bbox, int num_outbbox, int num_classes, int num_ratios,
                    bool use_segmoid_cls,
                    float* mean, float* std,
                    cudaStream_t stream);

}
}
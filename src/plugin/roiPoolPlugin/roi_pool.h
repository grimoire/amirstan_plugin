#pragma once
#include <cuda_runtime.h>



namespace amirstan
{
namespace plugin
{

    template<typename T>
    void roi_pool(T* output, 
                const T* rois, int num_rois,
                const void *const *feats, int num_feats,
                int n,
                int c,
                int *h,
                int *w,
                float *strides,
                int out_size,
                float roi_scale_factor,
                int finest_scale,
                cudaStream_t stream);

}
}
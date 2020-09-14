#pragma once
#include <cuda_runtime.h>



namespace amirstan
{
namespace plugin
{

    template<typename T>
    void roi_extractor(T* output, 
                        const T* rois, int num_rois,
                        const void *const *feats, int num_feats,
                        int n,
                        int c,
                        int *h,
                        int *w,
                        float *strides,
                        int out_size,
                        int sample_num,
                        float roi_scale_factor,
                        int finest_scale,
                        bool aligned,
                        cudaStream_t stream);

}
}
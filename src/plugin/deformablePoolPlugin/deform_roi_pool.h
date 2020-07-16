#pragma once

#include <cuda_runtime.h>



namespace amirstan
{
namespace plugin
{

void deform_roi_pool_forward(float* input, float* rois, float* offset,
                             float* output, int pooled_height, int pooled_width,
                            int output_size, int channels, int height, int width,
                             float spatial_scale, int sampling_ratio,
                             float gamma,
                            cudaStream_t stream);

}
}
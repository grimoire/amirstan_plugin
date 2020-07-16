#include "deform_roi_pool_cuda_kernel.cuh"
#include "deform_roi_pool.h"

#include "amir_cuda_util/cuda_util.h"



namespace amirstan
{
namespace plugin
{
    using namespace amirstan::cuda;
    template <typename scalar_t>
    void DeformRoIPoolForwardCUDAKernelLauncher(scalar_t* input, scalar_t* rois,
        scalar_t* offset, scalar_t* output,
        int pooled_height, int pooled_width,
        int output_size, int channels, int height, int width,
        float spatial_scale,
        int sampling_ratio, float gamma, cudaStream_t stream) {
    
        deform_roi_pool_forward_cuda_kernel<scalar_t>
        <<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
        output_size, input,
        rois, offset,
        output, pooled_height, pooled_width,
        static_cast<scalar_t>(spatial_scale), sampling_ratio,
        static_cast<scalar_t>(gamma), channels, height, width);
    
    }

    void deform_roi_pool_forward(float* input, float* rois, float* offset,
                                float* output, int pooled_height, int pooled_width,
                                int output_size, int channels, int height, int width,
                                float spatial_scale, int sampling_ratio,
                                float gamma,
                                cudaStream_t stream){
            DeformRoIPoolForwardCUDAKernelLauncher<float>(input, rois, offset, output,
                                                    pooled_height, pooled_width,
                                                    output_size, channels, height, width,
                                                    spatial_scale, sampling_ratio, gamma,
                                                    stream);
    }

}
}
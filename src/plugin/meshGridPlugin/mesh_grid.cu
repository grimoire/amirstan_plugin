#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <cuda_fp16.h>

#include "mesh_grid.h"
#include "amir_cuda_util/cuda_util.h"

namespace amirstan
{
namespace plugin
{
    using namespace amirstan::cuda;


    template <typename T>
    __global__ void arange_mesh_grid_kernel(T* output,
                                        size_t pre_stride, size_t post_stride,
                                        float start, float stride, size_t N){
        
        CUDA_KERNEL_LOOP(i, N){
            const size_t index = (i%pre_stride)/post_stride;

            const T value = start + index * (stride);
            output[i] = value;
        }
    }


    template <typename T>
    void arange_mesh_grid(T *output,
                            const int* output_dims, int nb_dims,
                            int slice_dim, float start, float stride,
                            cudaStream_t stream){

        size_t post_stride = 1;
        int i=nb_dims-1;
        for(i=nb_dims-1; i>slice_dim; --i){
            post_stride*=output_dims[i];
        }
        size_t pre_stride = post_stride*output_dims[slice_dim];

        size_t N = 1;
        for(i=0; i<nb_dims; ++i){
            N*=output_dims[i];
        }

        arange_mesh_grid_kernel<T><<<GET_BLOCKS(N), CUDA_NUM_THREADS,0,stream>>>(output, 
                                                                                pre_stride, post_stride,
                                                                                start, stride, N);

    }

    template void arange_mesh_grid<float>(float *output,
                                        const int* output_dims, int nb_dims,
                                        int slice_dim, float start, float stride,
                                        cudaStream_t stream);

    template void arange_mesh_grid<int>(int *output,
                                        const int* output_dims, int nb_dims,
                                        int slice_dim, float start, float stride,
                                        cudaStream_t stream);
                                    
    template void arange_mesh_grid<half>(half *output,
                                        const int* output_dims, int nb_dims,
                                        int slice_dim, float start, float stride,
                                        cudaStream_t stream);

}
}
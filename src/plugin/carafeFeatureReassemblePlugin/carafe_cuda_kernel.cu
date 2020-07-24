#include <cmath>
#include <algorithm>
#include <stdio.h>
#include "carafe_cuda.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024  // 32 * 32
#define WARP_SIZE 32
#define THREADS_PER_PIXEL 32
#define MAX_SHARED_MEMORY 49152
#define MAX_SHARED_SCALAR_T 6144  // 49152 / 8 = 6144
#define MAXIMIZE_KERNEL_SIZE true
#define kTileDim 32
#define kBlockRows 8
#define FULL_MASK 0xffffffff

namespace amirstan
{
namespace plugin
{
        
    inline int divideUP(const int x, const int y) { return (((x) + (y)-1) / (y)); }

    __device__ inline int Loc2Index(const int n, const int c, const int h,
                                    const int w, const int channel_num,
                                    const int height, const int width) {
    int index = w + (h + (c + n * channel_num) * height) * width;
    return index;
    }
    /* TODO: move this to a common place */
    template <typename scalar_t>
    __device__ inline scalar_t min(scalar_t a, scalar_t b) {
    return a < b ? a : b;
    }

    template <typename scalar_t>
    __device__ inline scalar_t max(scalar_t a, scalar_t b) {
    return a > b ? a : b;
    }

    // Splits the original matrix into submatrices with size 32 * 32.
    // Each block transposes one submatrix by loading it into shared memory.
    // Reference https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
    template <typename scalar_t>
    __global__ void BatchTranspose2DCUDAKernel(const int N, const int H,
                                            const int W, const int dh,
                                            const int dw,
                                            const scalar_t *__restrict__ X,
                                            scalar_t *__restrict__ Y) {
    __shared__ scalar_t tile[kTileDim][kTileDim + 1];
    const int n = blockIdx.x / (dh * dw);
    const int k = blockIdx.x % (dh * dw);
    const int r = k / dw;
    const int c = k % dw;
    const int offset = n * H * W;
    int x = c * kTileDim + threadIdx.x;
    int y = r * kTileDim + threadIdx.y;
    if (x < W) {
        for (int i = 0; threadIdx.y + i < kTileDim && y + i < H; i += kBlockRows) {
        tile[threadIdx.y + i][threadIdx.x] = X[offset + (y + i) * W + x];
        }
    }
    __syncthreads();
    x = r * kTileDim + threadIdx.x;
    y = c * kTileDim + threadIdx.y;
    if (x < H) {
        for (int i = 0; threadIdx.y + i < kTileDim && y + i < W; i += kBlockRows) {
        Y[offset + (y + i) * H + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
    }
    template <typename scalar_t>
    __global__ void CARAFEForward(
        const int num_kernels, const scalar_t *__restrict__ bottom_data,
        const scalar_t *__restrict__ bottom_masks, const int kernel_size,
        const int group_size, const int scale_factor, const int channels,
        const int down_height, const int down_width, const int height,
        const int width, const int mask_channels, scalar_t *__restrict__ top_data) {
    #if MAXIMIZE_KERNEL_SIZE
    __shared__ float shared_mask[MAX_SHARED_SCALAR_T * 2];
    #else
    __shared__ scalar_t shared_mask[MAX_SHARED_SCALAR_T];
    #endif

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index > num_kernels - 1) {
        return;
    }
    const int pixel_id = threadIdx.x / THREADS_PER_PIXEL;
    const int split_id = threadIdx.x % THREADS_PER_PIXEL;
    index = index / THREADS_PER_PIXEL;
    const int pw = index % width;
    const int ph = (index / width) % height;
    const int n = index / width / height;

    const int down_pw = pw / scale_factor;
    const int down_ph = ph / scale_factor;

    const int start_w = down_pw - (kernel_size - 1) / 2;
    const int end_w = down_pw + (kernel_size - 1) / 2 + 1;
    const int start_h = down_ph - (kernel_size - 1) / 2;
    const int end_h = down_ph + (kernel_size - 1) / 2 + 1;
    for (int c = split_id; c < mask_channels; c += THREADS_PER_PIXEL) {
        int mask_index = Loc2Index(n, ph, pw, c, height, width, mask_channels);
        shared_mask[c * WARP_SIZE + pixel_id] = bottom_masks[mask_index];
    }
    __syncthreads();

    const int channels_per_group = ceilf(channels / (float)group_size);
    #pragma unroll
    for (int c = split_id; c < channels; c += THREADS_PER_PIXEL) {
        int mask_group = c / channels_per_group;
        scalar_t output_val = 0;
    #pragma unroll
        for (int iy = start_h; iy < end_h; iy++) {
    #pragma unroll
        for (int ix = start_w; ix < end_w; ix++) {
            if (iy < 0 || iy > down_height - 1 || ix < 0 || ix > down_width - 1) {
            continue;
            }
            int mask_iy = iy - down_ph + (kernel_size - 1) / 2;
            int mask_ix = ix - down_pw + (kernel_size - 1) / 2;
            int mask_c =
                (mask_group * kernel_size + mask_iy) * kernel_size + mask_ix;
            int feat_index =
                Loc2Index(n, iy, ix, c, down_height, down_width, channels);

            output_val += bottom_data[feat_index] *
                        shared_mask[mask_c * WARP_SIZE + pixel_id];
        }
        }

        int top_index = Loc2Index(n, ph, pw, c, height, width, channels);
        top_data[top_index] = output_val;
    }
    }

    template <class T>
    int CARAFEForwardLaucher(const T* features, const T* masks,
                         const int kernel_size, const int group_size,
                         const int scale_factor, const int batch_size,
                         const int channels, const int input_height,
                         const int input_width, const int output_height,
                         const int output_width, const int mask_channels,
                         T* rfeatures, T* routput,
                         T* rmasks, T* output,
                         cudaStream_t stream){

        // fill rfeature
        int dh = divideUP(channels, kTileDim);
        int dw = divideUP(input_height * input_width, kTileDim);
        BatchTranspose2DCUDAKernel<T>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, channels, input_height * input_width, dh, dw,
                features, rfeatures);

        // fill rmask
        dh = divideUP(mask_channels, kTileDim);
        dw = divideUP(output_height * output_width, kTileDim);
        BatchTranspose2DCUDAKernel<T>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, mask_channels, output_height * output_width, dh, dw,
                masks, rmasks);
        
        // carafe
        const int num_kernels = batch_size * output_height * output_width * THREADS_PER_PIXEL;
        const int block_size = (num_kernels+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        CARAFEForward<T>
            <<<block_size,
               THREADS_PER_BLOCK, 0, stream>>>(
                num_kernels, rfeatures, rmasks, kernel_size, group_size,
                scale_factor, channels, input_height, input_width,
                output_height, output_width, mask_channels, routput);

        dh = divideUP(output_height * output_width, kTileDim);
        dw = divideUP(channels, kTileDim);
        BatchTranspose2DCUDAKernel<T>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, output_height * output_width, channels, dh, dw,
                routput, output);
        
        return 0;
    }

    template int CARAFEForwardLaucher<float>(const float* features, const float* masks,
        const int kernel_size, const int group_size,
        const int scale_factor, const int batch_size,
        const int channels, const int input_height,
        const int input_width, const int output_height,
        const int output_width, const int mask_channels,
        float* rfeatures, float* routput,
        float* rmasks, float* output,
        cudaStream_t stream);
}
}
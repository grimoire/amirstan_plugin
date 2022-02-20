#include <stdio.h>
#include <thrust/fill.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <cmath>

#include "amir_cuda_util/cuda_util.h"
#include "delta2bbox.h"

namespace amirstan {
namespace plugin {
using namespace amirstan::cuda;

struct SMeanStd {
  float mean[4];
  float std[4];
};

template <typename T>
__global__ void delta2bbox_kernel(T *__restrict__ out_bbox,
                                  const T *__restrict__ in_bbox,
                                  const T *__restrict__ anchor,
                                  const int *__restrict__ clip_range,
                                  int batch_size, int num_outbbox,
                                  int num_classes, SMeanStd mean_std) {
  extern __shared__ T anchor_cache[][4];
  const T max_ratio = abs(logf(16. / 1000.));

  const int class_id = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_id = blockIdx.z;
  if (class_id > num_classes) {
    return;
  }
  CUDA_KERNEL_LOOP(box_id, num_outbbox) {
    // load thread idx
    if (threadIdx.y == 0) {
      anchor_cache[threadIdx.x][0] = anchor[box_id * 4];
      anchor_cache[threadIdx.x][1] = anchor[box_id * 4 + 1];
      anchor_cache[threadIdx.x][2] = anchor[box_id * 4 + 2];
      anchor_cache[threadIdx.x][3] = anchor[box_id * 4 + 3];
    }
    __syncthreads();

    const int delta_offset = (batch_id * num_outbbox * num_classes +
                              box_id * num_classes + class_id) *
                             4;
    const T dx = in_bbox[delta_offset] * mean_std.std[0] + mean_std.mean[0];
    const T dy = in_bbox[delta_offset + 1] * mean_std.std[1] + mean_std.mean[1];
    const T dw = in_bbox[delta_offset + 2] * mean_std.std[2] + mean_std.mean[2];
    const T dh = in_bbox[delta_offset + 3] * mean_std.std[3] + mean_std.mean[3];

    const T clamp_dw = max(-max_ratio, min(max_ratio, dw));
    const T clamp_dh = max(-max_ratio, min(max_ratio, dh));

    const int anchor_offset = threadIdx.x;
    const T ax1 = anchor_cache[anchor_offset][0];
    const T ay1 = anchor_cache[anchor_offset][1];
    const T ax2 = anchor_cache[anchor_offset][2];
    const T ay2 = anchor_cache[anchor_offset][3];
    const T px = (ax1 + ax2) * 0.5;
    const T py = (ay1 + ay2) * 0.5;
    const T pw = ax2 - ax1;
    const T ph = ay2 - ay1;

    const T gw = pw * exp(dw);
    const T gh = ph * exp(dh);

    const T gx = px + pw * dx;
    const T gy = py + ph * dy;

    const T x1 = gx - gw * 0.5;
    const T y1 = gy - gh * 0.5;
    const T x2 = gx + gw * 0.5;
    const T y2 = gy + gh * 0.5;

    const int out_offset = delta_offset;
    if (clip_range != nullptr) {
      out_bbox[out_offset] = max(T(0.), min(x1, T(clip_range[1] - 1)));
      out_bbox[out_offset + 1] = max(T(0.), min(y1, T(clip_range[0] - 1)));
      out_bbox[out_offset + 2] = max(T(0.), min(x2, T(clip_range[1] - 1)));
      out_bbox[out_offset + 3] = max(T(0.), min(y2, T(clip_range[0] - 1)));
    } else {
      out_bbox[out_offset] = x1;
      out_bbox[out_offset + 1] = y1;
      out_bbox[out_offset + 2] = x2;
      out_bbox[out_offset + 3] = y2;
    }
  }
}

template <>
__global__ void delta2bbox_kernel<float>(float *__restrict__ out_bbox,
                                         const float *__restrict__ in_bbox,
                                         const float *__restrict__ anchor,
                                         const int *__restrict__ clip_range,
                                         int batch_size, int num_outbbox,
                                         int num_classes, SMeanStd mean_std) {
  extern __shared__ float4 anchor_cache[];
  const float max_ratio = abs(logf(16. / 1000.));

  const int class_id = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_id = blockIdx.z;
  if (class_id > num_classes) {
    return;
  }
  CUDA_KERNEL_LOOP(box_id, num_outbbox) {
    // load thread idx
    if (threadIdx.y == 0) {
      anchor_cache[threadIdx.x] =
          reinterpret_cast<const float4 *>(anchor)[box_id];
    }
    __syncthreads();

    const int delta_offset = (batch_id * num_outbbox * num_classes +
                              box_id * num_classes + class_id);
    const float4 delta =
        reinterpret_cast<const float4 *>(in_bbox)[delta_offset];

    const float &dx = delta.x * mean_std.std[0] + mean_std.mean[0];
    const float &dy = delta.y * mean_std.std[1] + mean_std.mean[1];
    const float &dw = delta.z * mean_std.std[2] + mean_std.mean[2];
    const float &dh = delta.w * mean_std.std[3] + mean_std.mean[3];

    const float clamp_dw = max(-max_ratio, min(max_ratio, dw));
    const float clamp_dh = max(-max_ratio, min(max_ratio, dh));

    const int anchor_offset = threadIdx.x;
    const float &ax1 = anchor_cache[anchor_offset].x;
    const float &ay1 = anchor_cache[anchor_offset].y;
    const float &ax2 = anchor_cache[anchor_offset].z;
    const float &ay2 = anchor_cache[anchor_offset].w;
    const float px = (ax1 + ax2) * 0.5;
    const float py = (ay1 + ay2) * 0.5;
    const float pw = ax2 - ax1;
    const float ph = ay2 - ay1;

    const float gw = pw * exp(dw);
    const float gh = ph * exp(dh);

    const float gx = fmaf(pw, dx, px);
    const float gy = fmaf(ph, dy, py);

    const float x1 = fmaf(gw, -0.5f, gx);
    const float y1 = fmaf(gh, -0.5f, gy);
    const float x2 = fmaf(gw, 0.5f, gx);
    const float y2 = fmaf(gh, 0.5f, gy);

    const int out_offset = delta_offset;
    float4 out_bbox_val;

    if (clip_range != nullptr) {
      out_bbox_val.x = max(0.0f, min(x1, float(clip_range[1] - 1)));
      out_bbox_val.y = max(0.0f, min(y1, float(clip_range[0] - 1)));
      out_bbox_val.z = max(0.0f, min(x2, float(clip_range[1] - 1)));
      out_bbox_val.w = max(0.0f, min(y2, float(clip_range[0] - 1)));
    } else {
      out_bbox_val.x = x1;
      out_bbox_val.y = y1;
      out_bbox_val.z = x2;
      out_bbox_val.w = y2;
    }

    reinterpret_cast<float4 *>(out_bbox)[out_offset] = out_bbox_val;
  }
}

template <typename T_cls, typename T_bbox>
void fill_zeros(T_cls *out_cls, T_bbox *out_bbox, size_t num_cls_element,
                size_t num_bbox_element, cudaStream_t stream) {
  thrust::fill_n(thrust::cuda::par.on(stream), (float *)(out_cls),
                 num_cls_element, 0.0f);
  thrust::fill_n(thrust::cuda::par.on(stream), (float *)(out_bbox),
                 num_bbox_element, 0.0f);
}

template void fill_zeros<float, float>(float *out_cls, float *out_bbox,
                                       size_t num_cls_element,
                                       size_t num_bbox_element,
                                       cudaStream_t stream);

template <typename T>
void delta2bbox(T *out_bbox, const T *in_bbox, const T *anchor,
                const int *clip_range, int batch_size, int num_outbbox,
                int num_classes, float *mean, float *std, cudaStream_t stream) {
  SMeanStd mean_std;
  memcpy(&mean_std.mean[0], mean, sizeof(float) * 4);
  memcpy(&mean_std.std[0], std, sizeof(float) * 4);
  const int classes_per_block = min(CUDA_NUM_THREADS, num_classes);
  const int threads_per_classes = int(CUDA_NUM_THREADS / classes_per_block);
  const int blocks_for_classes = DIVUP(num_classes, classes_per_block);
  const int blocks_for_bboxes = DIVUP(num_outbbox, threads_per_classes);
  const dim3 block_size(blocks_for_bboxes, blocks_for_classes, batch_size);
  const dim3 thread_size(threads_per_classes, classes_per_block, 1);

  const int cache_size = threads_per_classes * 4;
  delta2bbox_kernel<T>
      <<<block_size, thread_size, cache_size * sizeof(T), stream>>>(
          out_bbox, in_bbox, anchor, clip_range, batch_size, num_outbbox,
          num_classes, mean_std);
}

template void delta2bbox<float>(float *out_bbox, const float *in_bbox,
                                const float *anchor, const int *clip_range,
                                int batch_size, int num_outbbox,
                                int num_classes, float *mean, float *std,
                                cudaStream_t stream);
}  // namespace plugin
}  // namespace amirstan

#include <stdio.h>

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

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t softmax_custom(const scalar_t *data_start,
                                                   int step, int class_id,
                                                   int num_classes) {
  const scalar_t up_val = exp(*(data_start + step * class_id));
  scalar_t down_val = 0;
#pragma unroll
  for (int i = 0; i < num_classes; ++i) {
    down_val += exp(*(data_start + step * i));
  }

  return up_val / down_val;
}

template <typename T, bool use_sigmoid_cls>
__global__ void delta2bbox_kernel(
    T *__restrict__ out_cls, T *__restrict__ out_bbox,
    const T *__restrict__ in_cls, const T *__restrict__ in_bbox,
    const T *__restrict__ anchor, const int *__restrict__ clip_range,
    int batch_size, int num_bbox, int num_outbbox, int num_classes,
    int num_ratios, SMeanStd mean_std) {
  const T max_ratio = abs(logf(16. / 1000.));
  const int out_batch_stride = num_outbbox * num_classes;
  const int out_bbox_stride = num_classes * num_ratios;
  const int in_batch_stride = num_bbox * num_classes * num_ratios;
  const int in_ratio_stride = num_bbox * num_classes;
  const int in_class_stride = num_bbox;
  CUDA_KERNEL_LOOP(i, batch_size * num_outbbox * num_classes) {
    int tmp_i = i;
    const int batch_id = tmp_i / out_batch_stride;
    tmp_i %= out_batch_stride;
    const int bbox_id = tmp_i / out_bbox_stride;
    if (bbox_id >= num_bbox) {
      continue;
    }
    tmp_i %= out_bbox_stride;
    const int ratio_id = tmp_i / num_classes;
    if (ratio_id >= num_ratios) {
      continue;
    }
    const int class_id = tmp_i % num_classes;

    // cls
    const int out_cls_id = batch_id * out_batch_stride +
                           bbox_id * out_bbox_stride + ratio_id * num_classes +
                           class_id;
    const int in_cls_id = batch_id * in_batch_stride +
                          ratio_id * in_ratio_stride +
                          class_id * in_class_stride;
    if (use_sigmoid_cls) {
      out_cls[out_cls_id] = sigmoid<T>(in_cls[in_cls_id + bbox_id]);
    } else {
      out_cls[out_cls_id] =
          softmax_custom<T>(in_cls + batch_id * in_batch_stride +
                                ratio_id * in_ratio_stride + bbox_id,
                            num_bbox, class_id, num_classes);
    }

    // bbox
    if (class_id != 0) {
      continue;
    }
    // const int out_bbox_id = out_cls_id * 4 /num_classes;
    const int out_bbox_id =
        (batch_id * num_outbbox + bbox_id * num_ratios + ratio_id) * 4;
    const int in_delta_id =
        batch_id * num_bbox * num_ratios + ratio_id * num_bbox;
    const T dx =
        in_bbox[in_delta_id * 4 + bbox_id] * mean_std.std[0] + mean_std.mean[0];
    const T dy =
        in_bbox[in_delta_id * 4 + num_bbox + bbox_id] * mean_std.std[1] +
        mean_std.mean[1];
    const T dw =
        in_bbox[in_delta_id * 4 + num_bbox * 2 + bbox_id] * mean_std.std[2] +
        mean_std.mean[2];
    const T dh =
        in_bbox[in_delta_id * 4 + num_bbox * 3 + bbox_id] * mean_std.std[3] +
        mean_std.mean[3];

    const T clamp_dw = max(-max_ratio, min(max_ratio, dw));
    const T clamp_dh = max(-max_ratio, min(max_ratio, dh));

    const int anchor_start = (bbox_id * num_ratios + ratio_id) * 4;
    const T px = (anchor[anchor_start] + anchor[anchor_start + 2]) * 0.5;
    const T py = (anchor[anchor_start + 1] + anchor[anchor_start + 3]) * 0.5;
    const T pw = anchor[anchor_start + 2] - anchor[anchor_start];
    const T ph = anchor[anchor_start + 3] - anchor[anchor_start + 1];

    const T gw = pw * exp(dw);
    const T gh = ph * exp(dh);

    const T gx = px + pw * dx;
    const T gy = py + ph * dy;

    const T x1 = gx - gw * 0.5;
    const T y1 = gy - gh * 0.5;
    const T x2 = gx + gw * 0.5;
    const T y2 = gy + gh * 0.5;

    if (clip_range != nullptr) {
      out_bbox[out_bbox_id] = max(T(0.), min(x1, T(clip_range[1] - 1)));
      out_bbox[out_bbox_id + 1] = max(T(0.), min(y1, T(clip_range[0] - 1)));
      out_bbox[out_bbox_id + 2] = max(T(0.), min(x2, T(clip_range[1] - 1)));
      out_bbox[out_bbox_id + 3] = max(T(0.), min(y2, T(clip_range[0] - 1)));
    } else {
      out_bbox[out_bbox_id] = x1;
      out_bbox[out_bbox_id + 1] = y1;
      out_bbox[out_bbox_id + 2] = x2;
      out_bbox[out_bbox_id + 3] = y2;
    }
  }
}

template <bool use_sigmoid_cls>
__global__ void delta2bbox_kernel<float>(
    float *__restrict__ out_cls, float *__restrict__ out_bbox,
    const float *__restrict__ in_cls, const float *__restrict__ in_bbox,
    const float *__restrict__ anchor, const int *__restrict__ clip_range,
    int batch_size, int num_bbox, int num_outbbox, int num_classes,
    int num_ratios, SMeanStd mean_std) {
  const float max_ratio = abs(logf(16. / 1000.));
  const int out_batch_stride = num_outbbox * num_classes;
  const int out_bbox_stride = num_classes * num_ratios;
  const int in_batch_stride = num_bbox * num_classes * num_ratios;
  const int in_ratio_stride = num_bbox * num_classes;
  const int in_class_stride = num_bbox;
  CUDA_KERNEL_LOOP(i, batch_size * num_outbbox * num_classes) {
    int tmp_i = i;
    const int batch_id = tmp_i / out_batch_stride;
    tmp_i %= out_batch_stride;
    const int bbox_id = tmp_i / out_bbox_stride;
    if (bbox_id >= num_bbox) {
      continue;
    }
    tmp_i %= out_bbox_stride;
    const int ratio_id = tmp_i / num_classes;
    if (ratio_id >= num_ratios) {
      continue;
    }
    const int class_id = tmp_i % num_classes;

    // cls
    const int out_cls_id = batch_id * out_batch_stride +
                           bbox_id * out_bbox_stride + ratio_id * num_classes +
                           class_id;
    const int in_cls_id = batch_id * in_batch_stride +
                          ratio_id * in_ratio_stride +
                          class_id * in_class_stride;
    if (use_sigmoid_cls) {
      out_cls[out_cls_id] = sigmoid<float>(in_cls[in_cls_id + bbox_id]);
    } else {
      out_cls[out_cls_id] =
          softmax_custom<float>(in_cls + batch_id * in_batch_stride +
                                    ratio_id * in_ratio_stride + bbox_id,
                                num_bbox, class_id, num_classes);
    }

    // bbox
    if (class_id != 0) {
      continue;
    }
    const int out_bbox_id =
        batch_id * num_outbbox + bbox_id * num_ratios + ratio_id;
    const int in_delta_id =
        batch_id * num_bbox * num_ratios + ratio_id * num_bbox;

    const int in_bbox_base = in_delta_id << 2 + bbox_id;
    const float in_x = in_bbox[in_bbox_base];
    const float in_y = in_bbox[in_bbox_base + num_bbox];
    const float in_w = in_bbox[in_bbox_base + num_bbox * 2];
    const float in_h = in_bbox[in_bbox_base + num_bbox * 3];
    const float dx = fma(in_x, mean_std.std[0], mean_std.mean[0]);
    const float dy = fma(in_y, mean_std.std[1], mean_std.mean[1]);
    const float dw = fma(in_w, mean_std.std[2], mean_std.mean[2]);
    const float dh = fma(in_h, mean_std.std[3], mean_std.mean[3]);

    const float clamp_dw = max(-max_ratio, min(max_ratio, dw));
    const float clamp_dh = max(-max_ratio, min(max_ratio, dh));

    const int anchor_start = bbox_id * num_ratios + ratio_id;
    const float4 anchor_val =
        reinterpret_cast<const float4 *>(anchor)[anchor_start];
    const float &anchor_x0 = anchor_val.x;
    const float &anchor_y0 = anchor_val.y;
    const float &anchor_x1 = anchor_val.z;
    const float &anchor_y1 = anchor_val.w;

    const float px = (anchor_x0 + anchor_x1) * 0.5;
    const float py = (anchor_y0 + anchor_y1) * 0.5;
    const float pw = anchor_x1 - anchor_x0;
    const float ph = anchor_y1 - anchor_y0;

    const float gw = pw * exp(dw);
    const float gh = ph * exp(dh);

    const float gx = fma(pw, dx, px);
    const float gy = fma(ph, dy, py);

    const float x1 = fma(gw, -0.5, gx);
    const float y1 = fma(gh, -0.5, gy);
    const float x2 = fma(gw, 0.5, gx);
    const float y2 = fma(gh, 0.5, gy);

    float4 out_val;
    const bool need_clip_range = clip_range != nullptr;
    out_val.x =
        need_clip_range ? max(0.f, min(x1, float(clip_range[1] - 1))) : x1;
    out_val.y =
        need_clip_range ? max(0.f, min(y1, float(clip_range[0] - 1))) : y1;
    out_val.z =
        need_clip_range ? max(0.f, min(x2, float(clip_range[1] - 1))) : x2;
    out_val.w =
        need_clip_range ? max(0.f, min(y2, float(clip_range[0] - 1))) : y2;

    reinterpret_cast<float4 *>(out_bbox)[out_bbox_id] = out_val;
  }
}

template <typename T>
void delta2bbox(T *out_cls, T *out_bbox, const T *in_cls, const T *in_bbox,
                const T *anchor, const int *clip_range, int batch_size,
                int num_bbox, int num_outbbox, int num_classes, int num_ratios,
                bool use_sigmoid_cls, float *mean, float *std,
                cudaStream_t stream) {
  SMeanStd mean_std;
  memcpy(&mean_std.mean[0], mean, sizeof(float) * 4);
  memcpy(&mean_std.std[0], std, sizeof(float) * 4);
  const size_t input_size = batch_size * num_outbbox * num_classes;

  if (use_sigmoid_cls) {
    delta2bbox_kernel<T, true>
        <<<GET_BLOCKS(input_size), CUDA_NUM_THREADS, 0, stream>>>(
            out_cls, out_bbox, in_cls, in_bbox, anchor, clip_range, batch_size,
            num_bbox, num_outbbox, num_classes, num_ratios, mean_std);
  } else {
    delta2bbox_kernel<T, false>
        <<<GET_BLOCKS(input_size), CUDA_NUM_THREADS, 0, stream>>>(
            out_cls, out_bbox, in_cls, in_bbox, anchor, clip_range, batch_size,
            num_bbox, num_outbbox, num_classes, num_ratios, mean_std);
  }
}

template void delta2bbox<float>(float *out_cls, float *out_bbox,
                                const float *in_cls, const float *in_bbox,
                                const float *anchor, const int *clip_range,
                                int batch_size, int num_bbox, int num_outbbox,
                                int num_classes, int num_ratios,
                                bool use_sigmoid_cls, float *mean, float *std,
                                cudaStream_t stream);
}  // namespace plugin
}  // namespace amirstan

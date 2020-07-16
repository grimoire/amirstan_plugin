#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cmath>
#include <algorithm>

#include "amir_cuda_util/cuda_util.h"



template <typename T>
__device__ T bilinear_interpolate(const T* input, const int height,
                                  const int width, T y, T x,
                                  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = input[y_low * width + x_low];
  T v2 = input[y_low * width + x_high];
  T v3 = input[y_high * width + x_low];
  T v4 = input[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void deform_roi_pool_forward_cuda_kernel(
    const int nthreads, const T* input, const T* rois, const T* offset,
    T* output, const int pooled_height, const int pooled_width,
    const T spatial_scale, const int sampling_ratio, const T gamma,
    const int channels, const int height, const int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_rois[1] * spatial_scale - 0.5;
    T roi_start_h = offset_rois[2] * spatial_scale - 0.5;
    T roi_end_w = offset_rois[3] * spatial_scale - 0.5;
    T roi_end_h = offset_rois[4] * spatial_scale - 0.5;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h =
        (sampling_ratio > 0)
            ? sampling_ratio
            : static_cast<int>(ceil(roi_height / pooled_height));
    int roi_bin_grid_w = (sampling_ratio > 0)
                             ? sampling_ratio
                             : static_cast<int>(ceil(roi_width / pooled_width));

    // Compute roi offset
    if (offset != NULL) {
      const T* offset_cur_w = offset + n * pooled_width * pooled_height * 2 +
                              ph * pooled_width + pw;
      T offset_roi_w = gamma * roi_width * offset_cur_w[0];
      T offset_roi_h =
          gamma * roi_height * offset_cur_w[pooled_width * pooled_height];
      roi_start_w += offset_roi_w;
      roi_start_h += offset_roi_h;
    }

    // We do average pooling inside a bin
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = roi_start_h + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);
        T val = bilinear_interpolate(offset_input, height, width, y, x, index);
        output_val += val;
      }
    }
    output[index] = output_val / count;
  }
}
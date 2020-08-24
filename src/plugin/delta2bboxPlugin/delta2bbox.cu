#include <cmath>
#include <algorithm>
#include <stdio.h>
#include "delta2bbox.h"
#include "amir_cuda_util/cuda_util.h"


namespace amirstan
{
namespace plugin
{
  using namespace amirstan::cuda;

  struct SMeanStd{
    float mean[4];
    float std[4];
  };

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
    return 1.0 / (1.0 + exp(-z));
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t softmax_custom(const scalar_t *data_start, int step, int class_id, int num_classes) {
    const scalar_t up_val = exp(*(data_start+step*class_id));
    scalar_t down_val = 0;
    #pragma unroll
    for(int i=0;i<num_classes;++i){
      down_val += exp(*(data_start+step*i));
    }

    return up_val/down_val;
  }

  template<typename T>
  __global__ void delta2bbox_kernel(T* out_cls, T* out_bbox,
    const T* in_cls, const T* in_bbox, const T* anchor, const int* clip_range,
    int batch_size, int num_bbox, int num_outbbox, int num_classes, int num_ratios,
    bool use_segmoid_cls, 
    SMeanStd mean_std){
    
    const T max_ratio = abs(logf(16./1000.));
    const int out_batch_stride = num_outbbox*num_classes;
    const int out_bbox_stride = num_classes*num_ratios;
    const int in_batch_stride = num_bbox*num_classes*num_ratios;
    const int in_ratio_stride = num_bbox*num_classes;
    const int in_class_stride = num_bbox;
    CUDA_KERNEL_LOOP(i, batch_size*num_outbbox*num_classes){
      int tmp_i = i;
      const int batch_id = tmp_i/out_batch_stride;
      tmp_i %= out_batch_stride;
      const int bbox_id = tmp_i/out_bbox_stride;
      if(bbox_id>=num_bbox){
        continue;
      }
      tmp_i %= out_bbox_stride;
      const int ratio_id = tmp_i/num_classes;
      if(ratio_id>=num_ratios){
        continue;
      }
      const int class_id = tmp_i%num_classes;

      // cls
      const int out_cls_id = batch_id*out_batch_stride + bbox_id*out_bbox_stride + ratio_id*num_classes + class_id;
      const int in_cls_id = batch_id*in_batch_stride 
                            + ratio_id*in_ratio_stride 
                            + class_id*in_class_stride;
      if(use_segmoid_cls){
        out_cls[out_cls_id] = sigmoid<T>(in_cls[in_cls_id + bbox_id]);
      }else{
        out_cls[out_cls_id] = softmax_custom<T>(
          in_cls+batch_id*in_batch_stride + ratio_id*in_ratio_stride + bbox_id, num_bbox,
           class_id,
           num_classes);
      }

      // bbox
      if(class_id!=0){
        continue;
      }
      // const int out_bbox_id = out_cls_id * 4 /num_classes;
      const int out_bbox_id = (batch_id*num_outbbox + bbox_id*num_ratios+ratio_id)*4;
      const int in_delta_id = batch_id*num_bbox*num_ratios + ratio_id*num_bbox;
      const T dx = in_bbox[in_delta_id*4 + bbox_id]*mean_std.std[0] + mean_std.mean[0];
      const T dy = in_bbox[in_delta_id*4 + num_bbox + bbox_id]*mean_std.std[1] + mean_std.mean[1];
      const T dw = in_bbox[in_delta_id*4 + num_bbox*2 + bbox_id]*mean_std.std[2] + mean_std.mean[2];
      const T dh = in_bbox[in_delta_id*4 + num_bbox*3 + bbox_id]*mean_std.std[3] + mean_std.mean[3];

      const T clamp_dw = max(-max_ratio, min(max_ratio, dw));
      const T clamp_dh = max(-max_ratio, min(max_ratio, dh));

      const int anchor_start = (bbox_id*num_ratios+ratio_id)*4;
      const T px = (anchor[anchor_start]+anchor[anchor_start+2])*0.5;
      const T py = (anchor[anchor_start+1]+anchor[anchor_start+3])*0.5;
      const T pw = anchor[anchor_start+2]-anchor[anchor_start];
      const T ph = anchor[anchor_start+3]-anchor[anchor_start+1];

      const T gw = pw*exp(dw);
      const T gh = ph*exp(dh);

      const T gx = px + pw * dx;
      const T gy = py + ph * dy;

      const T x1 = gx - gw * 0.5;
      const T y1 = gy - gh * 0.5;
      const T x2 = gx + gw * 0.5;
      const T y2 = gy + gh * 0.5;

      if(clip_range!=nullptr){
        out_bbox[out_bbox_id] = max(T(0.), min(x1, T(clip_range[1])));
        out_bbox[out_bbox_id+1] = max(T(0.), min(y1, T(clip_range[0])));
        out_bbox[out_bbox_id+2] = max(T(0.), min(x2, T(clip_range[1])));
        out_bbox[out_bbox_id+3] = max(T(0.), min(y2, T(clip_range[0])));
      }else{
        out_bbox[out_bbox_id] = x1;
        out_bbox[out_bbox_id+1] = y1;
        out_bbox[out_bbox_id+2] = x2;
        out_bbox[out_bbox_id+3] = y2;
      }
    }
  }
  
  template<typename T>
  void delta2bbox(T* out_cls, T* out_bbox,
                  const T* in_cls, const T* in_bbox, const T* anchor, const int* clip_range,
                  int batch_size, int num_bbox, int num_outbbox, int num_classes, int num_ratios,
                  bool use_segmoid_cls,
                  float* mean, float* std,
                  cudaStream_t stream){
    SMeanStd mean_std;
    memcpy(&mean_std.mean[0], mean, sizeof(float)*4);
    memcpy(&mean_std.std[0], std, sizeof(float)*4);
    const size_t input_size = batch_size*num_outbbox*num_classes;
    delta2bbox_kernel<T><<<GET_BLOCKS(input_size), CUDA_NUM_THREADS,0,stream>>>(
      out_cls, out_bbox,
      in_cls, in_bbox, anchor, clip_range,
      batch_size, num_bbox, num_outbbox, num_classes, num_ratios,
      use_segmoid_cls, 
      mean_std
    );
  }

  template
  void delta2bbox<float>(float* out_cls, float* out_bbox,
                  const float* in_cls, const float* in_bbox, const float* anchor, const int* clip_range,
                  int batch_size, int num_bbox, int num_outbbox, int num_classes, int num_ratios,
                  bool use_segmoid_cls,
                  float* mean, float* std,
                  cudaStream_t stream);
}
}
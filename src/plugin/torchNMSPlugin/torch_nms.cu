#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include "torch_nms.h"
#include "amir_cuda_util/cuda_util.h"


namespace amirstan
{
namespace plugin
{
  using namespace amirstan::cuda;

  template <typename T>
  struct Bbox
  {
      T xmin, ymin, xmax, ymax;
      Bbox(T xmin, T ymin, T xmax, T ymax)
          : xmin(xmin)
          , ymin(ymin)
          , xmax(xmax)
          , ymax(ymax)
      {
      }
      Bbox() = default;
  };
  
  template <typename T_BBOX>
  __device__ T_BBOX bboxSize(
      const Bbox<T_BBOX>& bbox,
      const bool normalized,
      T_BBOX offset)
  {
      if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin)
      {
          // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
          return 0;
      }
      else
      {
          T_BBOX width = bbox.xmax - bbox.xmin;
          T_BBOX height = bbox.ymax - bbox.ymin;
          if (normalized)
          {
              return width * height;
          }
          else
          {
              // If bbox is not within range [0, 1].
              return (width + offset) * (height + offset);
          }
      }
  }

  template <typename T_BBOX>
  __device__ void intersectBbox(
      const Bbox<T_BBOX>& bbox1,
      const Bbox<T_BBOX>& bbox2,
      Bbox<T_BBOX>* intersect_bbox)
  {
      if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin || bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin)
      {
          // Return [0, 0, 0, 0] if there is no intersection.
          intersect_bbox->xmin = T_BBOX(0);
          intersect_bbox->ymin = T_BBOX(0);
          intersect_bbox->xmax = T_BBOX(0);
          intersect_bbox->ymax = T_BBOX(0);
      }
      else
      {
          intersect_bbox->xmin = max(bbox1.xmin, bbox2.xmin);
          intersect_bbox->ymin = max(bbox1.ymin, bbox2.ymin);
          intersect_bbox->xmax = min(bbox1.xmax, bbox2.xmax);
          intersect_bbox->ymax = min(bbox1.ymax, bbox2.ymax);
      }
  }

  template <typename T_BBOX>
  __device__ float jaccardOverlap(
      const Bbox<T_BBOX>& bbox1,
      const Bbox<T_BBOX>& bbox2,
      const bool normalized,
      T_BBOX offset)
  {
      Bbox<T_BBOX> intersect_bbox;
      intersectBbox(bbox1, bbox2, &intersect_bbox);
      float intersect_width, intersect_height;
      if (normalized)
      {
          intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
          intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;
      }
      else
      {
          intersect_width = intersect_bbox.xmax - intersect_bbox.xmin + offset;
          intersect_height = intersect_bbox.ymax - intersect_bbox.ymin + offset;
      }
      if (intersect_width > 0 && intersect_height > 0)
      {
          float intersect_size = intersect_width * intersect_height;
          float bbox1_size = bboxSize(bbox1, normalized, offset);
          float bbox2_size = bboxSize(bbox2, normalized, offset);
          return intersect_size / (bbox1_size + bbox2_size - intersect_size);
      }
      else
      {
          return 0.;
      }
  }


  template<typename T>
  __global__ void arange(T *output, const size_t size){
    CUDA_KERNEL_LOOP(index, size){
      output[index] = T(index);
    }
  }

  
  template <typename T_SCORE, typename T_BBOX, int TSIZE>
  __global__ void nms_score_kernel(T_SCORE* scores, const T_BBOX* bbox_data, const int* score_index, int num_boxes, float iou_threshold){

    extern __shared__ bool kept_bboxinfo_flag[];

    const int max_idx = num_boxes;

    // local thread data
    int loc_bboxIndex[TSIZE];
    Bbox<T_BBOX> loc_bbox[TSIZE];

    __syncthreads();
    #pragma unroll
    for (int t = 0; t < TSIZE; t++)
    {

      const int cur_idx = threadIdx.x + blockDim.x * t;
      const int item_idx = cur_idx;

      if (item_idx < max_idx)
      {
        loc_bboxIndex[t] = score_index[item_idx];

        if (loc_bboxIndex[t] >= 0)
        // if (loc_bboxIndex[t] != -1)
        {
          const int bbox_data_idx = loc_bboxIndex[t];

          loc_bbox[t].xmin = bbox_data[bbox_data_idx * 4 + 0];
          loc_bbox[t].ymin = bbox_data[bbox_data_idx * 4 + 1];
          loc_bbox[t].xmax = bbox_data[bbox_data_idx * 4 + 2];
          loc_bbox[t].ymax = bbox_data[bbox_data_idx * 4 + 3];
          kept_bboxinfo_flag[cur_idx] = true;
        }
        else
        {
            kept_bboxinfo_flag[cur_idx] = false;
        }
      }
      else
      {
          kept_bboxinfo_flag[cur_idx] = false;
      }
    }

    int ref_item_idx = 0;
    int ref_bbox_idx = score_index[0];


    while (ref_item_idx < max_idx)
    {
        Bbox<T_BBOX> ref_bbox;
        ref_bbox.xmin = bbox_data[ref_bbox_idx * 4 + 0];
        ref_bbox.ymin = bbox_data[ref_bbox_idx * 4 + 1];
        ref_bbox.xmax = bbox_data[ref_bbox_idx * 4 + 2];
        ref_bbox.ymax = bbox_data[ref_bbox_idx * 4 + 3];

        // Eliminate shared memory RAW hazard  
        __syncthreads();

        for (int t = 0; t < TSIZE; t++)
        {
            const int cur_idx = threadIdx.x + blockDim.x * t;
            const int item_idx = cur_idx;

            if ((kept_bboxinfo_flag[cur_idx]) && (item_idx > ref_item_idx))
            {
                // TODO: may need to add bool normalized as argument, HERE true means normalized
                if (jaccardOverlap(ref_bbox, loc_bbox[t], false, T_BBOX(0)) > iou_threshold)
                {
                    kept_bboxinfo_flag[cur_idx] = false;
                }
            }
        }
        __syncthreads();

        do
        {
            ref_item_idx++;
        } while (ref_item_idx < max_idx && !kept_bboxinfo_flag[ref_item_idx]);

        ref_bbox_idx = score_index[ref_item_idx];
    }


    // store data
    for (int t = 0; t < TSIZE; t++)
    {
        const int cur_idx = threadIdx.x + blockDim.x * t;
        const int write_item_idx = loc_bboxIndex[t];
        /*
          * If not not keeping the bbox
          * Set the score to -1
          */
        if (cur_idx < max_idx)
        {
          scores[write_item_idx] = kept_bboxinfo_flag[cur_idx] ? scores[write_item_idx] : T_SCORE(-1.0f);
        }
    }
  }


  template<typename T_SCORE>
  __global__ void remove_negative_score(int *output, const T_SCORE *scores,  const size_t size){
    CUDA_KERNEL_LOOP(index, size){
      output[index] = scores[index]>0. ? output[index]:-1;
    }
  }

  template <typename T>
  size_t nms_workspace_size(int num_boxes){
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;
    T* scores = nullptr;
    T* tmp_score = nullptr;
    int* input_index = nullptr;
    int* output = nullptr;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                            scores, tmp_score, input_index, output,
                                            num_boxes, 0, sizeof(T)*8);
    return temp_storage_bytes;
  }

  template size_t nms_workspace_size<float>(int num_boxes);


  template <typename T>
  void torch_nms(int *output, const T* bboxes, const T* scores,
                  int num_boxes, float iou_threshold, void* workspace,
                  cudaStream_t stream){
    int *input_index = (int*)workspace;
    T* tmp_score = (T*)(input_index + num_boxes);
    T* final_score = (T*)(tmp_score + num_boxes);
    void* tmp_sort = (void*)(final_score + num_boxes);
    
    // arange
    arange<<<GET_BLOCKS(num_boxes), CUDA_NUM_THREADS, 0, stream>>>(input_index, num_boxes);
    
    // first sort
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                              scores, tmp_score, input_index, output,
                                              num_boxes, 0, sizeof(T)*8, stream);
    d_temp_storage = tmp_sort;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                              scores, tmp_score, input_index, output,
                                              num_boxes, 0, sizeof(T)*8, stream);
    
    cudaMemcpyAsync((void*)final_score, (void*)scores, num_boxes*sizeof(T), cudaMemcpyDeviceToDevice, stream);

    // do nms
    #define NMS_SCORE_KERNEL(tsize) nms_score_kernel<T, T, (tsize)>

    void (*kernel[8])(T*, const T*, const int*, int, float)
    = {
      NMS_SCORE_KERNEL(1), NMS_SCORE_KERNEL(2), NMS_SCORE_KERNEL(3), NMS_SCORE_KERNEL(4), 
      NMS_SCORE_KERNEL(5), NMS_SCORE_KERNEL(6), NMS_SCORE_KERNEL(7), NMS_SCORE_KERNEL(8)
    };

    const int t_size = (num_boxes + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    kernel[t_size - 1]<<<1, CUDA_NUM_THREADS, CUDA_NUM_THREADS * t_size * sizeof(bool), stream>>>(final_score, bboxes, output, num_boxes, iou_threshold);


    // second sort
    d_temp_storage = nullptr;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                              final_score, tmp_score, input_index, output,
                                              num_boxes, 0, sizeof(T)*8, stream);
    d_temp_storage = tmp_sort;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                              final_score, tmp_score, input_index, output,
                                              num_boxes, 0, sizeof(T)*8, stream);

    // mark index of negative score as -1
    remove_negative_score<<<GET_BLOCKS(num_boxes), CUDA_NUM_THREADS, 0, stream>>>(output, tmp_score, num_boxes);

  }
    
  template void torch_nms<float>(int *output, const float* bboxes, const float* scores,
                  int num_boxes, float iou_threshold, void* workspace,
                  cudaStream_t stream);
}   // namespace plugin
}   // namespace amirstan
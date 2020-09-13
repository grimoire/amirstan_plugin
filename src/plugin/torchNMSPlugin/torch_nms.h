#pragma once

namespace amirstan
{
namespace plugin
{

    template <typename T>
    void torch_nms(int *output, const T* bboxes, const T* scores,
                    int num_boxes, float iou_threshold, void* workspace,
                    cudaStream_t stream);

}
}
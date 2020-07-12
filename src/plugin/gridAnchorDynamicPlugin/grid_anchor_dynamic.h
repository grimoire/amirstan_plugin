#pragma once

namespace amirstan
{
namespace plugin
{

template <typename T>
void grid_anchor_dynamic(T *output, const T* base_anchor, 
                        int width, int height, 
                        int stride,
                        int num_base_anchor,
                        cudaStream_t stream);

}
} // namespace amirstan
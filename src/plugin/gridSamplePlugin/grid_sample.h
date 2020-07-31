#pragma once

namespace amirstan
{
namespace plugin
{

  enum class GridSamplerInterpolation {Bilinear, Nearest};
  enum class GridSamplerPadding {Zeros, Border, Reflection};

    template <typename T>
    void grid_sample(T *output, const T* input, const T* grid, 
                    int* output_dims, int* input_dims, int *grid_dims, int nb_dims,
                    GridSamplerInterpolation interp, GridSamplerPadding padding,
                    bool align_corners,
                    cudaStream_t stream);

}
}
#pragma once

namespace amirstan
{
namespace plugin
{

template <typename T>
void arange_mesh_grid(T *output,
                        const int* output_dims, int nb_dims,
                        int slice_dim, float start, float stride,
                        cudaStream_t stream);

}
}
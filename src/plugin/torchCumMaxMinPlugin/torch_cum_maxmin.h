#pragma once

namespace amirstan
{
namespace plugin
{

    template <typename T>
    void torch_cum_maxmin(T *output, int *index, const T* input,
                    int* input_dims, int nb_dims,
                    int cum_dim, int cum_type,
                    cudaStream_t stream);

}
}
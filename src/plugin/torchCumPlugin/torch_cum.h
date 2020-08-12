#pragma once

namespace amirstan
{
namespace plugin
{

    template <typename T>
    void torch_cum(T *output, const T* input,
                    int* input_dims, int nb_dims,
                    int cum_dim, int cum_type,
                    cudaStream_t stream);

}
}
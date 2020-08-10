#pragma once

namespace amirstan
{
namespace plugin
{

    template <typename T>
    void torch_flip(T *output, const T* input,
                    int* input_dims, int nb_dims,
                    int* flip_dims, int nb_flip_dims,
                    cudaStream_t stream);

}
}
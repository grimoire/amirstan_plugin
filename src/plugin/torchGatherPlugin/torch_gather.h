#pragma once

namespace amirstan
{
namespace plugin
{

template <typename T>
void torch_gather(T *output, const T* input, const int* index, 
                        int dim, int* input_dims, int *index_dims, int nb_dims,
                        cudaStream_t stream);

}
}
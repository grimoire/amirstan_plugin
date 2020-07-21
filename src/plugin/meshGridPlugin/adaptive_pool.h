#pragma once

namespace amirstan
{
namespace plugin
{

enum PoolType{
    MAX=0,
    AVERAGE=1
};

template <typename T>
void adaptive_pool(T *output, const T* input, 
                    int* input_dims, int* output_dims, int nb_dims,
                    int nb_reduce_dims,
                    PoolType pool_type,
                    cudaStream_t stream);

}
}
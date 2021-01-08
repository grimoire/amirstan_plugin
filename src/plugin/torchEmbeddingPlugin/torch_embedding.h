
#pragma once

namespace amirstan {
namespace plugin {
template <typename T>
void torch_embedding(T* output, const int* index, const T* weight,
                     int num_indices, int embedding_dim, cudaStream_t stream);
}
}  // namespace amirstan
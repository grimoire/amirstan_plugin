#pragma once

namespace amirstan {
namespace common {
inline size_t getAlignedSize(size_t origin_size, size_t aligned_number = 16) {
  return size_t((origin_size + aligned_number - 1) / aligned_number) *
         aligned_number;
}
}  // namespace common
}  // namespace amirstan
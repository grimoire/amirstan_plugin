#pragma once

namespace amirstan
{
namespace plugin
{
    template <class T>
    int CARAFEForwardLaucher(const T* features, const T* masks,
                         const int kernel_size, const int group_size,
                         const int scale_factor, const int batch_size,
                         const int channels, const int input_height,
                         const int input_width, const int output_height,
                         const int output_width, const int mask_channels,
                         T* rfeatures, T* routput,
                         T* rmasks, T* output,
                         cudaStream_t stream);
}
} // namespace amirstan
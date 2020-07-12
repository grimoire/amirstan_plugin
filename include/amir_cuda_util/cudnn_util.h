#include <cudnn.h>

namespace amirstan
{
namespace cudnn
{
template <class T>
void cudnnBatchNormTrain(cudnnHandle_t handle,
                         const T &input,
                         int batch_size,
                         int channels,
                         int width,
                         int height,
                         const T &weight, const T &bias,
                         const T &running_mean, const T &running_var,
                         T exponentialAverageFactor,
                         T epsilon,
                         T &result_mean, T &result_var);
}
} // namespace amirstan
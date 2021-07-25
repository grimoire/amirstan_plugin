
#include "torchBmmPlugin.h"

#include <assert.h>

#include <chrono>

#include "amirCommon.h"
#include "amir_cuda_util/cuda_util.h"
#include "common.h"
#include "serialize.hpp"

namespace amirstan {
namespace plugin {

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"TorchBmmPluginDynamic"};
}  // namespace

TorchBmmPluginDynamic::TorchBmmPluginDynamic(const std::string &name)
    : PluginDynamicBase(name) {}

TorchBmmPluginDynamic::TorchBmmPluginDynamic(const std::string name,
                                             const void *data, size_t length)
    : PluginDynamicBase(name) {}

nvinfer1::IPluginV2DynamicExt *TorchBmmPluginDynamic::clone() const
    PLUGIN_NOEXCEPT {
  TorchBmmPluginDynamic *plugin = new TorchBmmPluginDynamic(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TorchBmmPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = inputs[1].d[2];
  return ret;
}

bool TorchBmmPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  if (pos == 0) {
    return (inOut[0].type == nvinfer1::DataType::kFLOAT &&
            inOut[0].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }
}

void TorchBmmPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  // Validate input arguments
}

size_t TorchBmmPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const PLUGIN_NOEXCEPT {
  return 0;
}

int TorchBmmPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs,
                                   void *const *outputs, void *workSpace,
                                   cudaStream_t stream) PLUGIN_NOEXCEPT {
  if (m_cuda_stream != stream) {
    cublasSetStream(m_cublas_handle, stream);
    m_cuda_stream = stream;
  }

  const float *A = (float *)inputs[0];
  const float *B = (float *)inputs[1];
  float *C = (float *)outputs[0];
  const int batch_size = inputDesc[0].dims.d[0];
  const int m = inputDesc[0].dims.d[1];
  const int k = inputDesc[0].dims.d[2];
  const int n = inputDesc[1].dims.d[2];
  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const long long int strideA = m * k;
  const long long int strideB = k * n;
  const long long int strideC = m * n;
  cublasSgemmStridedBatched(m_cublas_handle,           // handle
                            CUBLAS_OP_N, CUBLAS_OP_N,  // transform
                            n, m, k,                   // n, m, k
                            &alpha,                    // alpha
                            B, ldb, strideB,           // config B
                            A, lda, strideA,           // config A
                            &beta,                     // beta
                            C, ldc, strideC,           // config C
                            batch_size);

  return 0;
}

nvinfer1::DataType TorchBmmPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const PLUGIN_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *TorchBmmPluginDynamic::getPluginType() const PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TorchBmmPluginDynamic::getPluginVersion() const PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

int TorchBmmPluginDynamic::getNbOutputs() const PLUGIN_NOEXCEPT { return 1; }

void TorchBmmPluginDynamic::attachToContext(
    cudnnContext *cudnnContext, cublasContext *cublasContext,
    nvinfer1::IGpuAllocator *gpuAllocator) PLUGIN_NOEXCEPT {
  m_cublas_handle = cublasContext;
}

size_t TorchBmmPluginDynamic::getSerializationSize() const PLUGIN_NOEXCEPT {
  return 0;
}

void TorchBmmPluginDynamic::serialize(void *buffer) const PLUGIN_NOEXCEPT {}

////////////////////// creator /////////////////////////////

TorchBmmPluginDynamicCreator::TorchBmmPluginDynamicCreator() {
  mPluginAttributes = std::vector<PluginField>({});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TorchBmmPluginDynamicCreator::getPluginName() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TorchBmmPluginDynamicCreator::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

IPluginV2 *TorchBmmPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) PLUGIN_NOEXCEPT {
  TorchBmmPluginDynamic *plugin = new TorchBmmPluginDynamic(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

IPluginV2 *TorchBmmPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) PLUGIN_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new TorchBmmPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

}  // namespace plugin
}  // namespace amirstan

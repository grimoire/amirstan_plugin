
#include "plugin/torchBmmPlugin/torchBmmPlugin.h"

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

PluginFieldCollection TorchBmmPluginDynamicCreator::mFC{};
std::vector<PluginField> TorchBmmPluginDynamicCreator::mPluginAttributes({});

TorchBmmPluginDynamic::TorchBmmPluginDynamic(const std::string &name)
    : mLayerName(name) {
  initialize();
}

TorchBmmPluginDynamic::TorchBmmPluginDynamic(const std::string name,
                                             const void *data, size_t length)
    : mLayerName(name) {
  initialize();
}

nvinfer1::IPluginV2DynamicExt *TorchBmmPluginDynamic::clone() const {
  TorchBmmPluginDynamic *plugin = new TorchBmmPluginDynamic(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TorchBmmPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = inputs[1].d[2];
  return ret;
}

bool TorchBmmPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) {
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
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {
  // Validate input arguments
}

size_t TorchBmmPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  return 0;
}

int TorchBmmPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs,
                                   void *const *outputs, void *workSpace,
                                   cudaStream_t stream) {
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
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *TorchBmmPluginDynamic::getPluginType() const { return PLUGIN_NAME; }

const char *TorchBmmPluginDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

int TorchBmmPluginDynamic::getNbOutputs() const { return 1; }

int TorchBmmPluginDynamic::initialize() {
  cublasCreate(&m_cublas_handle);
  return 0;
}

void TorchBmmPluginDynamic::terminate() { cublasDestroy(m_cublas_handle); }

size_t TorchBmmPluginDynamic::getSerializationSize() const { return 0; }

void TorchBmmPluginDynamic::serialize(void *buffer) const {}

void TorchBmmPluginDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void TorchBmmPluginDynamic::setPluginNamespace(const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *TorchBmmPluginDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

TorchBmmPluginDynamicCreator::TorchBmmPluginDynamicCreator() {
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TorchBmmPluginDynamicCreator::getPluginName() const {
  return PLUGIN_NAME;
}

const char *TorchBmmPluginDynamicCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const PluginFieldCollection *TorchBmmPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

IPluginV2 *TorchBmmPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) {
  TorchBmmPluginDynamic *plugin = new TorchBmmPluginDynamic(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

IPluginV2 *TorchBmmPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new TorchBmmPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void TorchBmmPluginDynamicCreator::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *TorchBmmPluginDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}

}  // namespace plugin
}  // namespace amirstan

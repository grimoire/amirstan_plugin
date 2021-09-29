#include "torchEmbeddingPlugin.h"

#include <assert.h>

#include <chrono>

#include "amirCommon.h"
#include "amir_cuda_util/cuda_util.h"
#include "common.h"
#include "serialize.hpp"
#include "torch_embedding.h"

namespace amirstan {
namespace plugin {

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"TorchEmbeddingPluginDynamic"};
}  // namespace

TorchEmbeddingPluginDynamic::TorchEmbeddingPluginDynamic(
    const std::string &name, int numEmbeddings, int embeddingDim)
    : PluginDynamicBase(name),
      mNumEmbeddings(numEmbeddings),
      mEmbeddingDim(embeddingDim) {}

TorchEmbeddingPluginDynamic::TorchEmbeddingPluginDynamic(const std::string name,
                                                         const void *data,
                                                         size_t length)
    : PluginDynamicBase(name) {
  deserialize_value(&data, &length, &mNumEmbeddings);
  deserialize_value(&data, &length, &mEmbeddingDim);
}

nvinfer1::IPluginV2DynamicExt *TorchEmbeddingPluginDynamic::clone() const
    PLUGIN_NOEXCEPT {
  TorchEmbeddingPluginDynamic *plugin = new TorchEmbeddingPluginDynamic(
      mLayerName, mNumEmbeddings, mEmbeddingDim);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TorchEmbeddingPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  ret.nbDims = inputs[0].nbDims + 1;
  for (int i = 0; i < inputs[0].nbDims; ++i) {
    ret.d[i] = inputs[0].d[i];
  }
  ret.d[inputs[0].nbDims] = exprBuilder.constant(mEmbeddingDim);
  return ret;
}

bool TorchEmbeddingPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  const auto *in = inOut;
  const auto *out = inOut + nbInputs;

  switch (pos) {
    case 0:
      return in[0].type == nvinfer1::DataType::kINT32 &&
             in[0].format == nvinfer1::TensorFormat::kLINEAR;
    case 1:
      return in[1].type == nvinfer1::DataType::kFLOAT &&
             in[1].format == nvinfer1::TensorFormat::kLINEAR;
    case 2:
      return out[0].type == nvinfer1::DataType::kFLOAT &&
             out[0].format == nvinfer1::TensorFormat::kLINEAR;
  }
  return false;
}

void TorchEmbeddingPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) PLUGIN_NOEXCEPT {}

size_t TorchEmbeddingPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const PLUGIN_NOEXCEPT {
  return 0;
}

int TorchEmbeddingPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace,
    cudaStream_t stream) PLUGIN_NOEXCEPT {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  int num_indices = 1;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    num_indices *= input_dims.d[i];
  }

  auto data_type = inputDesc[0].type;
  const void *mDevWeight = inputs[1];

  switch (data_type) {
    case nvinfer1::DataType::kINT32:
      torch_embedding<float>((float *)outputs[0], (int *)inputs[0],
                             (float *)mDevWeight, num_indices, mEmbeddingDim,
                             stream);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType TorchEmbeddingPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const PLUGIN_NOEXCEPT {
  return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods
const char *TorchEmbeddingPluginDynamic::getPluginType() const PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TorchEmbeddingPluginDynamic::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

int TorchEmbeddingPluginDynamic::getNbOutputs() const PLUGIN_NOEXCEPT {
  return 1;
}

size_t TorchEmbeddingPluginDynamic::getSerializationSize() const
    PLUGIN_NOEXCEPT {
  return sizeof(mNumEmbeddings) + sizeof(mEmbeddingDim);
}

void TorchEmbeddingPluginDynamic::serialize(void *buffer) const
    PLUGIN_NOEXCEPT {
  serialize_value(&buffer, mNumEmbeddings);
  serialize_value(&buffer, mEmbeddingDim);
}

////////////////////// creator /////////////////////////////

TorchEmbeddingPluginDynamicCreator::TorchEmbeddingPluginDynamicCreator() {
  mPluginAttributes = std::vector<PluginField>(
      {PluginField("num_embeddings"), PluginField("embedding_dim")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TorchEmbeddingPluginDynamicCreator::getPluginName() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TorchEmbeddingPluginDynamicCreator::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

IPluginV2 *TorchEmbeddingPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) PLUGIN_NOEXCEPT {
  int numEmbeddings = 0;
  int embeddingDim = 0;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("num_embeddings") == 0) {
      numEmbeddings = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("embedding_dim") == 0) {
      embeddingDim = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }

  TorchEmbeddingPluginDynamic *plugin =
      new TorchEmbeddingPluginDynamic(name, numEmbeddings, embeddingDim);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

IPluginV2 *TorchEmbeddingPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) PLUGIN_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new TorchEmbeddingPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

}  // namespace plugin
}  // namespace amirstan
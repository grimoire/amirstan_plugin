
#include "torchCumPlugin.h"

#include <assert.h>

#include <chrono>

#include "amirCommon.h"
#include "amir_cuda_util/cuda_util.h"
#include "common.h"
#include "serialize.hpp"
#include "torch_cum.h"

namespace amirstan {
namespace plugin {

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"TorchCumPluginDynamic"};
}  // namespace

TorchCumPluginDynamic::TorchCumPluginDynamic(const std::string &name, int dim,
                                             int cumType)
    : PluginDynamicBase(name), mDim(dim), mCumType(cumType) {}

TorchCumPluginDynamic::TorchCumPluginDynamic(const std::string name,
                                             const void *data, size_t length)
    : PluginDynamicBase(name) {
  deserialize_value(&data, &length, &mDim);
  deserialize_value(&data, &length, &mCumType);
}

nvinfer1::IPluginV2DynamicExt *TorchCumPluginDynamic::clone() const
    PLUGIN_NOEXCEPT {
  TorchCumPluginDynamic *plugin =
      new TorchCumPluginDynamic(mLayerName, mDim, mCumType);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TorchCumPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT {
  nvinfer1::DimsExprs ret = inputs[0];
  return ret;
}

bool TorchCumPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  const auto *in = inOut;
  const auto *out = inOut + nbInputs;

  switch (pos) {
    case 0:
      return (in[0].type == nvinfer1::DataType::kFLOAT &&
              in[0].format == nvinfer1::TensorFormat::kLINEAR) or
             (in[0].type == nvinfer1::DataType::kINT32 &&
              in[0].format == nvinfer1::TensorFormat::kLINEAR);
    case 1:
      return out[0].type == in[0].type && out[0].format == in[0].format;
  }
  return false;
}

void TorchCumPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  // Validate input arguments
}

size_t TorchCumPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const PLUGIN_NOEXCEPT {
  return 0;
}

int TorchCumPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs,
                                   void *const *outputs, void *workSpace,
                                   cudaStream_t stream) PLUGIN_NOEXCEPT {
  nvinfer1::Dims input_dims = inputDesc[0].dims;

  auto data_type = inputDesc[0].type;

  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      torch_cum<float>((float *)outputs[0], (float *)inputs[0],
                       &(input_dims.d[0]), input_dims.nbDims, mDim, mCumType,
                       stream);
      break;
    case nvinfer1::DataType::kINT32:
      torch_cum<int>((int *)outputs[0], (int *)inputs[0], &(input_dims.d[0]),
                     input_dims.nbDims, mDim, mCumType, stream);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType TorchCumPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const PLUGIN_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *TorchCumPluginDynamic::getPluginType() const PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TorchCumPluginDynamic::getPluginVersion() const PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

int TorchCumPluginDynamic::getNbOutputs() const PLUGIN_NOEXCEPT { return 1; }

size_t TorchCumPluginDynamic::getSerializationSize() const PLUGIN_NOEXCEPT {
  return sizeof(mDim) + sizeof(mCumType);
}

void TorchCumPluginDynamic::serialize(void *buffer) const PLUGIN_NOEXCEPT {
  serialize_value(&buffer, mDim);
  serialize_value(&buffer, mCumType);
}

////////////////////// creator /////////////////////////////

TorchCumPluginDynamicCreator::TorchCumPluginDynamicCreator() {
  mPluginAttributes = std::vector<PluginField>(
      {PluginField("mode"), PluginField("dim"), PluginField("cum_type")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TorchCumPluginDynamicCreator::getPluginName() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TorchCumPluginDynamicCreator::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

IPluginV2 *TorchCumPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) PLUGIN_NOEXCEPT {
  int dim = 0;
  int cumType = 0;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("dim") == 0) {
      dim = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("cum_type") == 0) {
      cumType = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }

  TorchCumPluginDynamic *plugin = new TorchCumPluginDynamic(name, dim, cumType);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

IPluginV2 *TorchCumPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) PLUGIN_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new TorchCumPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

}  // namespace plugin
}  // namespace amirstan

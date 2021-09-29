
#include "gridAnchorDynamicPlugin.h"

#include <assert.h>
#include <math.h>

#include <chrono>

#include "amirCommon.h"
#include "common.h"
#include "grid_anchor_dynamic.h"
#include "serialize.hpp"

namespace amirstan {
namespace plugin {

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"GridAnchorDynamicPluginDynamic"};
}  // namespace

GridAnchorDynamicPluginDynamic::GridAnchorDynamicPluginDynamic(
    const std::string &name, int stride)
    : PluginDynamicBase(name), mStride(stride) {}

GridAnchorDynamicPluginDynamic::GridAnchorDynamicPluginDynamic(
    const std::string name, const void *data, size_t length)
    : PluginDynamicBase(name) {
  deserialize_value(&data, &length, &mStride);
}

nvinfer1::IPluginV2DynamicExt *GridAnchorDynamicPluginDynamic::clone() const
    PLUGIN_NOEXCEPT {
  GridAnchorDynamicPluginDynamic *plugin =
      new GridAnchorDynamicPluginDynamic(mLayerName, mStride);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs GridAnchorDynamicPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 2;

  auto area = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
                                    *inputs[0].d[2], *inputs[0].d[3]);

  ret.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *area,
                                   *(inputs[1].d[0]));
  ret.d[1] = exprBuilder.constant(4);

  return ret;
}

bool GridAnchorDynamicPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  if (pos == 0) {
    return inOut[0].type == nvinfer1::DataType::kFLOAT &&
           inOut[0].format == nvinfer1::TensorFormat::kLINEAR;
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }
}

void GridAnchorDynamicPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  // Validate input arguments
}

size_t GridAnchorDynamicPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const PLUGIN_NOEXCEPT {
  return 0;
}

int GridAnchorDynamicPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace,
    cudaStream_t stream) PLUGIN_NOEXCEPT {
  int inputHeight = inputDesc[0].dims.d[2];
  int inputWidth = inputDesc[0].dims.d[3];

  int mNumBaseAnchor = inputDesc[1].dims.d[0];

  grid_anchor_dynamic<float>((float *)outputs[0], (float *)inputs[1],
                             inputWidth, inputHeight, mStride, mNumBaseAnchor,
                             stream);

  return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType GridAnchorDynamicPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const PLUGIN_NOEXCEPT {
  return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods
const char *GridAnchorDynamicPluginDynamic::getPluginType() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *GridAnchorDynamicPluginDynamic::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

int GridAnchorDynamicPluginDynamic::getNbOutputs() const PLUGIN_NOEXCEPT {
  return 1;
}

size_t GridAnchorDynamicPluginDynamic::getSerializationSize() const
    PLUGIN_NOEXCEPT {
  return sizeof(mStride);
}

void GridAnchorDynamicPluginDynamic::serialize(void *buffer) const
    PLUGIN_NOEXCEPT {
  serialize_value(&buffer, mStride);
}
////////////////////// creator /////////////////////////////

GridAnchorDynamicPluginDynamicCreator::GridAnchorDynamicPluginDynamicCreator() {
  mPluginAttributes = std::vector<PluginField>(
      {PluginField("base_size"), PluginField("stride")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *GridAnchorDynamicPluginDynamicCreator::getPluginName() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *GridAnchorDynamicPluginDynamicCreator::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

IPluginV2 *GridAnchorDynamicPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) PLUGIN_NOEXCEPT {
  int stride = 0;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("stride") == 0) {
      stride = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }

  GridAnchorDynamicPluginDynamic *plugin;
  plugin = new GridAnchorDynamicPluginDynamic(name, stride);

  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

IPluginV2 *GridAnchorDynamicPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) PLUGIN_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin =
      new GridAnchorDynamicPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

}  // namespace plugin
}  // namespace amirstan

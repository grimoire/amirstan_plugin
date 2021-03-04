
#include "plugin/groupNormPlugin/groupNormPlugin.h"

#include <assert.h>

#include <chrono>

#include "amirCommon.h"
#include "amir_cuda_util/common_util.h"
#include "common.h"
#include "group_norm.h"
#include "serialize.hpp"

namespace amirstan {
namespace plugin {

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"GroupNormPluginDynamic"};
}  // namespace

PluginFieldCollection GroupNormPluginDynamicCreator::mFC{};
std::vector<PluginField> GroupNormPluginDynamicCreator::mPluginAttributes(
    {PluginField("num_groups"), PluginField("eps")});

GroupNormPluginDynamic::GroupNormPluginDynamic(const std::string &name,
                                               int num_groups, float eps)
    : mLayerName(name), mNumGroups(num_groups), mEps(eps) {}

GroupNormPluginDynamic::GroupNormPluginDynamic(const std::string name,
                                               const void *data, size_t length)
    : mLayerName(name) {
  deserialize_value(&data, &length, &mNumGroups);
  deserialize_value(&data, &length, &mEps);
}

nvinfer1::IPluginV2DynamicExt *GroupNormPluginDynamic::clone() const {
  GroupNormPluginDynamic *plugin =
      new GroupNormPluginDynamic(mLayerName, mNumGroups, mEps);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs GroupNormPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  return inputs[0];
}

bool GroupNormPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) {
  if (pos == 0) {
    return (inOut[0].type == nvinfer1::DataType::kFLOAT &&
            inOut[0].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }

  return false;
}

void GroupNormPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {
  // Validate input arguments
}

size_t GroupNormPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  size_t wordSize = samplesCommon::getElementSize(inputs[0].type);
  int batch_size = inputs[0].dims.d[0];
  int channel_size = inputs[0].dims.d[1];
  int width = inputs[0].dims.d[2];
  int height = inputs[0].dims.d[3];
  size_t mean_size =
      amirstan::common::getAlignedSize(batch_size * mNumGroups * wordSize);
  size_t var_size = mean_size;
  size_t mean_var_size = amirstan::common::getAlignedSize(
      batch_size * channel_size * width * height * wordSize);

  size_t final_size = mean_size + var_size + 100;
  return final_size;
}

int GroupNormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace, cudaStream_t stream) {
  int batch_size = inputDesc[0].dims.d[0];
  int inputChannel = inputDesc[0].dims.d[1];
  int inputHeight = inputDesc[0].dims.d[2];
  int inputWidth = inputDesc[0].dims.d[3];

  auto data_type = inputDesc[0].type;

  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      compute_group_norm<float>(
          (float *)outputs[0], (float *)inputs[0], batch_size, mNumGroups,
          inputChannel, inputWidth * inputHeight, mEps, (float *)inputs[1],
          (float *)inputs[2], stream, workSpace);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType GroupNormPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *GroupNormPluginDynamic::getPluginType() const {
  return PLUGIN_NAME;
}

const char *GroupNormPluginDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

int GroupNormPluginDynamic::getNbOutputs() const { return 1; }

int GroupNormPluginDynamic::initialize() { return 0; }

void GroupNormPluginDynamic::terminate() {}

size_t GroupNormPluginDynamic::getSerializationSize() const {
  return sizeof(mNumGroups) + sizeof(mEps);
}

void GroupNormPluginDynamic::serialize(void *buffer) const {
  serialize_value(&buffer, mNumGroups);
  serialize_value(&buffer, mEps);
}

void GroupNormPluginDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void GroupNormPluginDynamic::setPluginNamespace(const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *GroupNormPluginDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

GroupNormPluginDynamicCreator::GroupNormPluginDynamicCreator() {
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *GroupNormPluginDynamicCreator::getPluginName() const {
  return PLUGIN_NAME;
}

const char *GroupNormPluginDynamicCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const PluginFieldCollection *GroupNormPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

IPluginV2 *GroupNormPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) {
  int num_groups = 0;
  float eps = 1e-5;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("num_groups") == 0) {
      num_groups = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("eps") == 0) {
      eps = static_cast<const float *>(fc->fields[i].data)[0];
    }
  }

  GroupNormPluginDynamic *plugin =
      new GroupNormPluginDynamic(name, num_groups, eps);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

IPluginV2 *GroupNormPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new GroupNormPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void GroupNormPluginDynamicCreator::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *GroupNormPluginDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}

}  // namespace plugin
}  // namespace amirstan

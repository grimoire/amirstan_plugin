
#include "deformablePoolPlugin.h"

#include <assert.h>

#include <chrono>

#include "amirCommon.h"
#include "common.h"
#include "deform_roi_pool.h"
#include "serialize.hpp"

namespace amirstan {
namespace plugin {

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"DeformablePoolPluginDynamic"};
}  // namespace

DeformablePoolPluginDynamic::DeformablePoolPluginDynamic(
    const std::string &name, const nvinfer1::Dims &outSize, float spatialScale,
    int samplingRatio, float gamma)
    : PluginDynamicBase(name),
      mOutSize(outSize),
      mSpatialScale(spatialScale),
      mSamplingRatio(samplingRatio),
      mGamma(gamma) {}

DeformablePoolPluginDynamic::DeformablePoolPluginDynamic(const std::string name,
                                                         const void *data,
                                                         size_t length)
    : PluginDynamicBase(name) {
  deserialize_value(&data, &length, &mOutSize);
  deserialize_value(&data, &length, &mSpatialScale);
  deserialize_value(&data, &length, &mSamplingRatio);
  deserialize_value(&data, &length, &mGamma);
}

nvinfer1::IPluginV2DynamicExt *DeformablePoolPluginDynamic::clone() const
    PLUGIN_NOEXCEPT {
  DeformablePoolPluginDynamic *plugin = new DeformablePoolPluginDynamic(
      mLayerName, mOutSize, mSpatialScale, mSamplingRatio, mGamma);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs DeformablePoolPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT {
  assert(nbInputs == 2 || nbInputs == 3);

  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[1].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = exprBuilder.constant(mOutSize.d[0]);
  ret.d[3] = exprBuilder.constant(mOutSize.d[1]);

  return ret;
}

bool DeformablePoolPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  // assert(0 <= pos && pos < 2);
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void DeformablePoolPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  // Validate input arguments
  assert(nbOutputs == 1);
  assert(nbInputs == 2 || nbInputs == 3);
}

size_t DeformablePoolPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const PLUGIN_NOEXCEPT {
  return 0;
}

int DeformablePoolPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace,
    cudaStream_t stream) PLUGIN_NOEXCEPT {
  int channels = inputDesc[0].dims.d[1];
  int height = inputDesc[0].dims.d[2];
  int width = inputDesc[0].dims.d[3];

  float *input = (float *)inputs[0];
  float *rois = (float *)inputs[1];

  float *offset = NULL;
  if (inputDesc[2].dims.nbDims != 0) {
    offset = (float *)inputs[2];
  }

  float *output = (float *)outputs[0];

  int output_size = 1;
  for (int i = 0; i < outputDesc[0].dims.nbDims; ++i) {
    output_size *= outputDesc[0].dims.d[i];
  }

  int pooled_height = mOutSize.d[0];
  int pooled_width = mOutSize.d[1];

  deform_roi_pool_forward(input, rois, offset, output, pooled_height,
                          pooled_width, output_size, channels, height, width,
                          mSpatialScale, mSamplingRatio, mGamma, stream);

  return 0;
}

nvinfer1::DataType DeformablePoolPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const PLUGIN_NOEXCEPT {
  return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods
const char *DeformablePoolPluginDynamic::getPluginType() const PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *DeformablePoolPluginDynamic::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

int DeformablePoolPluginDynamic::getNbOutputs() const PLUGIN_NOEXCEPT {
  return 1;
}

size_t DeformablePoolPluginDynamic::getSerializationSize() const
    PLUGIN_NOEXCEPT {
  return sizeof(mOutSize) + sizeof(mSpatialScale) + sizeof(mSamplingRatio) +
         sizeof(mGamma);
}

void DeformablePoolPluginDynamic::serialize(void *buffer) const
    PLUGIN_NOEXCEPT {
  serialize_value(&buffer, mOutSize);
  serialize_value(&buffer, mSpatialScale);
  serialize_value(&buffer, mSamplingRatio);
  serialize_value(&buffer, mGamma);
}

////////////////////// creator /////////////////////////////

DeformablePoolPluginDynamicCreator::DeformablePoolPluginDynamicCreator() {
  mPluginAttributes = std::vector<PluginField>(
      {PluginField("out_size"), PluginField("spatial_scale"),
       PluginField("sampling_ratio"), PluginField("gamma")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *DeformablePoolPluginDynamicCreator::getPluginName() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *DeformablePoolPluginDynamicCreator::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

IPluginV2 *DeformablePoolPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) PLUGIN_NOEXCEPT {
  nvinfer1::Dims outSize;
  outSize.nbDims = 2;
  outSize.d[0] = 7;
  outSize.d[1] = 7;
  float spatialScale = 1.;
  int samplingRatio = 0.;
  float gamma = 0.1;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("out_size") == 0) {
      outSize.nbDims = 2;
      outSize.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      outSize.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }

    if (field_name.compare("spatial_scale") == 0) {
      spatialScale = static_cast<const float *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("sampling_ratio") == 0) {
      samplingRatio = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("gamma") == 0) {
      gamma = static_cast<const float *>(fc->fields[i].data)[0];
    }
  }

  DeformablePoolPluginDynamic *plugin = new DeformablePoolPluginDynamic(
      name, outSize, spatialScale, samplingRatio, gamma);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

IPluginV2 *DeformablePoolPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) PLUGIN_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new DeformablePoolPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

}  // namespace plugin
}  // namespace amirstan

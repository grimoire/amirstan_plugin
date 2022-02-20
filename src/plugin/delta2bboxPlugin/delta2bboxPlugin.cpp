
#include "delta2bboxPlugin.h"

#include <assert.h>

#include <chrono>

#include "amirCommon.h"
#include "common.h"
#include "delta2bbox.h"
#include "serialize.hpp"

namespace amirstan {
namespace plugin {

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"Delta2BBoxPluginDynamic"};
}  // namespace

Delta2BBoxPluginDynamic::Delta2BBoxPluginDynamic(
    const std::string &name, int minNumBBox,
    const std::vector<float> &targetMeans, const std::vector<float> &targetStds)
    : PluginDynamicBase(name),
      mMinNumBBox(minNumBBox),
      mTargetMeans(targetMeans),
      mTargetStds(targetStds) {}

Delta2BBoxPluginDynamic::Delta2BBoxPluginDynamic(const std::string name,
                                                 const void *data,
                                                 size_t length)
    : PluginDynamicBase(name) {
  deserialize_value(&data, &length, &mMinNumBBox);

  const int mean_std_size = 4;

  const char *d = static_cast<const char *>(data);

  float *mean_data =
      (float *)deserToHost<char>(d, mean_std_size * sizeof(float));
  mTargetMeans =
      std::vector<float>(&mean_data[0], &mean_data[0] + mean_std_size);
  float *std_data =
      (float *)deserToHost<char>(d, mean_std_size * sizeof(float));
  mTargetStds = std::vector<float>(&std_data[0], &std_data[0] + mean_std_size);
  initialize();
}

nvinfer1::IPluginV2DynamicExt *Delta2BBoxPluginDynamic::clone() const
    PLUGIN_NOEXCEPT {
  Delta2BBoxPluginDynamic *plugin = new Delta2BBoxPluginDynamic(
      mLayerName, mMinNumBBox, mTargetMeans, mTargetStds);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs Delta2BBoxPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT {
  assert(nbInputs == 3 || nbInputs == 4);

  auto out_bbox = inputs[0].d[1];

  auto final_bbox =
      exprBuilder.operation(nvinfer1::DimensionOperation::kMAX,
                            *exprBuilder.constant(mMinNumBBox), *out_bbox);

  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = final_bbox;

  switch (outputIndex) {
    case 0:
      ret.d[2] = inputs[0].d[2];
      break;
    case 1:
      ret.d[2] = inputs[1].d[2];
      break;
    default:
      break;
  }

  return ret;
}

bool Delta2BBoxPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  const auto *in = inOut;
  const auto *out = inOut + nbInputs;
  switch (pos) {
    case 0:
      return in[0].type == nvinfer1::DataType::kFLOAT &&
             in[0].format == nvinfer1::TensorFormat::kLINEAR;
    case 1:
      return in[1].type == nvinfer1::DataType::kFLOAT &&
             in[1].format == nvinfer1::TensorFormat::kLINEAR;
    case 2:
      return in[2].type == in[1].type && in[2].format == in[1].format;
  }

  // output
  switch (pos - nbInputs) {
    case 0:
      return out[0].type == in[0].type && out[0].format == in[0].format;
    case 1:
      return out[1].type == in[1].type && out[1].format == in[1].format;
  }

  if (nbInputs == 4 && pos == 3) {
    return in[3].type == nvinfer1::DataType::kINT32 &&
           in[3].format == nvinfer1::TensorFormat::kLINEAR;
  }

  return false;
}

void Delta2BBoxPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  mNeedClipRange = nbInputs == 4;
  // Validate input arguments
}

size_t Delta2BBoxPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const PLUGIN_NOEXCEPT {
  return 0;
}

int Delta2BBoxPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace,
    cudaStream_t stream) PLUGIN_NOEXCEPT {
  int batch_size = inputDesc[0].dims.d[0];
  int num_inboxes = inputDesc[0].dims.d[1];
  int num_classes = inputDesc[0].dims.d[2];
  int num_outboxes = outputDesc[0].dims.d[1];

  int dim_inboxes = inputDesc[1].dims.nbDims;
  int num_box_classes = inputDesc[1].dims.d[dim_inboxes - 1] / 4;

  void *out_cls = outputs[0];
  void *out_bbox = outputs[1];
  const void *in_cls = inputs[0];
  const void *in_bbox = inputs[1];
  const void *anchor = inputs[2];

  int *clip_range = mNeedClipRange ? (int *)(inputs[3]) : nullptr;

  if (num_inboxes != num_outboxes) {
    fill_zeros<float, float>((float *)out_cls, (float *)out_bbox,
                             outputDesc[0].dims.d[0] * outputDesc[0].dims.d[1] *
                                 outputDesc[0].dims.d[2],
                             outputDesc[1].dims.d[0] * outputDesc[1].dims.d[1] *
                                 outputDesc[1].dims.d[2],
                             stream);
  }
  cudaMemcpyAsync(out_cls, in_cls,
                  batch_size * num_inboxes * num_classes * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  delta2bbox<float>((float *)out_bbox, (float *)in_bbox, (float *)anchor,
                    clip_range, batch_size, num_inboxes, num_box_classes,
                    mTargetMeans.data(), mTargetStds.data(), stream);
  return 0;
}

nvinfer1::DataType Delta2BBoxPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const PLUGIN_NOEXCEPT {
  switch (index) {
    case 0:
      return inputTypes[0];
    default:
      return inputTypes[1];
  }
}

// IPluginV2 Methods
const char *Delta2BBoxPluginDynamic::getPluginType() const PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *Delta2BBoxPluginDynamic::getPluginVersion() const PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

int Delta2BBoxPluginDynamic::getNbOutputs() const PLUGIN_NOEXCEPT { return 2; }

size_t Delta2BBoxPluginDynamic::getSerializationSize() const PLUGIN_NOEXCEPT {
  return mTargetMeans.size() * sizeof(float) +
         mTargetStds.size() * sizeof(float) + sizeof(mMinNumBBox);
}

void Delta2BBoxPluginDynamic::serialize(void *buffer) const PLUGIN_NOEXCEPT {
  serialize_value(&buffer, mMinNumBBox);

  const int mean_std_size = 4;

  char *d = static_cast<char *>(buffer);
  serFromHost(d, mTargetMeans.data(), mean_std_size);
  serFromHost(d, mTargetStds.data(), mean_std_size);
}

////////////////////// creator /////////////////////////////

Delta2BBoxPluginDynamicCreator::Delta2BBoxPluginDynamicCreator() {
  mPluginAttributes = std::vector<PluginField>({PluginField("min_num_bbox"),
                                                PluginField("target_means"),
                                                PluginField("target_stds")});

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *Delta2BBoxPluginDynamicCreator::getPluginName() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *Delta2BBoxPluginDynamicCreator::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

IPluginV2 *Delta2BBoxPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) PLUGIN_NOEXCEPT {
  int minNumBBox = 1000;
  std::vector<float> targetMeans;
  std::vector<float> targetStds;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("min_num_bbox") == 0) {
      minNumBBox = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("target_means") == 0) {
      int data_size = fc->fields[i].length;
      const float *data_start = static_cast<const float *>(fc->fields[i].data);
      targetMeans = std::vector<float>(data_start, data_start + data_size);
    }

    if (field_name.compare("target_stds") == 0) {
      int data_size = fc->fields[i].length;
      const float *data_start = static_cast<const float *>(fc->fields[i].data);
      targetStds = std::vector<float>(data_start, data_start + data_size);
    }
  }

  if (targetMeans.size() != 4) {
    std::cerr << "target mean must contain 4 elements" << std::endl;
  }

  if (targetStds.size() != 4) {
    std::cerr << "target std must contain 4 elements" << std::endl;
  }

  Delta2BBoxPluginDynamic *plugin =
      new Delta2BBoxPluginDynamic(name, minNumBBox, targetMeans, targetStds);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

IPluginV2 *Delta2BBoxPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) PLUGIN_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new Delta2BBoxPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

}  // namespace plugin
}  // namespace amirstan

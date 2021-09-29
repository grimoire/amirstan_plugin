
#include "torchNMSPlugin.h"

#include <assert.h>

#include <chrono>

#include "amirCommon.h"
#include "amir_cuda_util/common_util.h"
#include "amir_cuda_util/cuda_util.h"
#include "common.h"
#include "serialize.hpp"
#include "torch_nms.h"

namespace amirstan {
namespace plugin {

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"TorchNMSPluginDynamic"};
}  // namespace

TorchNMSPluginDynamic::TorchNMSPluginDynamic(const std::string &name,
                                             float iouThreshold)
    : PluginDynamicBase(name), mIouThreshold(iouThreshold) {}

TorchNMSPluginDynamic::TorchNMSPluginDynamic(const std::string name,
                                             const void *data, size_t length)
    : PluginDynamicBase(name) {
  deserialize_value(&data, &length, &mIouThreshold);
}

nvinfer1::IPluginV2DynamicExt *TorchNMSPluginDynamic::clone() const
    PLUGIN_NOEXCEPT {
  TorchNMSPluginDynamic *plugin =
      new TorchNMSPluginDynamic(mLayerName, mIouThreshold);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TorchNMSPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT {
  nvinfer1::DimsExprs ret = inputs[1];
  return ret;
}

bool TorchNMSPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  const auto *in = inOut;
  const auto *out = inOut + nbInputs;

  switch (pos) {
    case 0:
      return (in[0].type == nvinfer1::DataType::kFLOAT &&
              in[0].format == nvinfer1::TensorFormat::kLINEAR);
    case 1:
      return (in[1].type == nvinfer1::DataType::kFLOAT &&
              in[1].format == nvinfer1::TensorFormat::kLINEAR);
    case 2:
      return (out[0].type == nvinfer1::DataType::kINT32 &&
              out[0].format == nvinfer1::TensorFormat::kLINEAR);
  }
  return false;
}

void TorchNMSPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  // Validate input arguments
}

size_t TorchNMSPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const PLUGIN_NOEXCEPT {
  int num_bbox = inputs[1].dims.d[0];
  auto data_type = inputs[1].type;
  size_t tmp_score_workspace = 0;
  size_t final_score_workspace = 0;
  size_t nms_workspace = 0;
  size_t index_workspace =
      amirstan::common::getAlignedSize(num_bbox * sizeof(int));
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      tmp_score_workspace = num_bbox * sizeof(float);
      final_score_workspace = num_bbox * sizeof(float);
      nms_workspace = nms_workspace_size<float>(num_bbox) + 1;
      break;
    default:
      break;
  }
  tmp_score_workspace = amirstan::common::getAlignedSize(tmp_score_workspace);
  final_score_workspace =
      amirstan::common::getAlignedSize(final_score_workspace);
  nms_workspace = amirstan::common::getAlignedSize(nms_workspace);
  return tmp_score_workspace + final_score_workspace + index_workspace +
         nms_workspace;
}

int TorchNMSPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs,
                                   void *const *outputs, void *workSpace,
                                   cudaStream_t stream) PLUGIN_NOEXCEPT {
  nvinfer1::Dims score_dims = inputDesc[1].dims;

  int num_boxes = score_dims.d[0];

  auto data_type = inputDesc[1].type;
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      torch_nms<float>((int *)outputs[0], (float *)inputs[0],
                       (float *)inputs[1], num_boxes, mIouThreshold, workSpace,
                       stream);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType TorchNMSPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const PLUGIN_NOEXCEPT {
  return nvinfer1::DataType::kINT32;
}

// IPluginV2 Methods
const char *TorchNMSPluginDynamic::getPluginType() const PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TorchNMSPluginDynamic::getPluginVersion() const PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

int TorchNMSPluginDynamic::getNbOutputs() const PLUGIN_NOEXCEPT { return 1; }

size_t TorchNMSPluginDynamic::getSerializationSize() const PLUGIN_NOEXCEPT {
  return sizeof(mIouThreshold);
}

void TorchNMSPluginDynamic::serialize(void *buffer) const PLUGIN_NOEXCEPT {
  serialize_value(&buffer, mIouThreshold);
}

////////////////////// creator /////////////////////////////

TorchNMSPluginDynamicCreator::TorchNMSPluginDynamicCreator() {
  mPluginAttributes = std::vector<PluginField>({PluginField("iou_threshold")});

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TorchNMSPluginDynamicCreator::getPluginName() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TorchNMSPluginDynamicCreator::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

IPluginV2 *TorchNMSPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) PLUGIN_NOEXCEPT {
  float iou_threshold = 0.;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("iou_threshold") == 0) {
      iou_threshold = static_cast<const float *>(fc->fields[i].data)[0];
    }
  }

  TorchNMSPluginDynamic *plugin =
      new TorchNMSPluginDynamic(name, iou_threshold);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

IPluginV2 *TorchNMSPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) PLUGIN_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new TorchNMSPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

}  // namespace plugin
}  // namespace amirstan

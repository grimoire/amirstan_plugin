
#include "torchUnfoldPlugin.h"

#include <assert.h>

#include <chrono>

#include "amirCommon.h"
#include "amir_cuda_util/cuda_util.h"
#include "common.h"
#include "serialize.hpp"
#include "torch_unfold.h"

namespace amirstan {
namespace plugin {

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"TorchUnfoldPluginDynamic"};
}  // namespace

TorchUnfoldPluginDynamic::TorchUnfoldPluginDynamic(
    const std::string &name, const std::vector<int32_t> &kernelSize,
    const std::vector<int32_t> &dilation, const std::vector<int32_t> &padding,
    const std::vector<int32_t> &stride)
    : PluginDynamicBase(name),
      mKernelSize(kernelSize),
      mDilation(dilation),
      mPadding(padding),
      mStride(stride) {}

TorchUnfoldPluginDynamic::TorchUnfoldPluginDynamic(const std::string name,
                                                   const void *data,
                                                   size_t length)
    : PluginDynamicBase(name) {
  deserialize_value(&data, &length, &mKernelSize);
  deserialize_value(&data, &length, &mDilation);
  deserialize_value(&data, &length, &mPadding);
  deserialize_value(&data, &length, &mStride);
}

nvinfer1::IPluginV2DynamicExt *TorchUnfoldPluginDynamic::clone() const
    PLUGIN_NOEXCEPT {
  TorchUnfoldPluginDynamic *plugin = new TorchUnfoldPluginDynamic(
      mLayerName, mKernelSize, mDilation, mPadding, mStride);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

static const nvinfer1::IDimensionExpr *conv_edge_expr(
    const nvinfer1::IDimensionExpr *input, int32_t kernel_size,
    int32_t dilation, int32_t padding, int32_t stride,
    nvinfer1::IExprBuilder &exprBuilder) {
  int32_t offset = 2 * padding - dilation * (kernel_size - 1) - 1;
  auto edge = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *input,
                                    *exprBuilder.constant(offset));
  edge = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *edge,
                               *exprBuilder.constant(stride));
  edge = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *edge,
                               *exprBuilder.constant(1));
  return edge;
}

nvinfer1::DimsExprs TorchUnfoldPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = exprBuilder.operation(
      nvinfer1::DimensionOperation::kPROD, *inputs[0].d[1],
      *exprBuilder.constant(mKernelSize[0] * mKernelSize[1]));

  auto w = conv_edge_expr(inputs[0].d[3], mKernelSize[0], mDilation[0],
                          mPadding[0], mStride[0], exprBuilder);

  auto h = conv_edge_expr(inputs[0].d[2], mKernelSize[1], mDilation[1],
                          mPadding[1], mStride[1], exprBuilder);
  ret.d[2] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *w, *h);

  return ret;
}

bool TorchUnfoldPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  switch (pos) {
    case 0:
      return (inOut[0].type == nvinfer1::DataType::kFLOAT &&
              inOut[0].format == nvinfer1::TensorFormat::kLINEAR) ||
             (inOut[0].type == nvinfer1::DataType::kINT32 &&
              inOut[0].format == nvinfer1::TensorFormat::kLINEAR);
    case 1:
      return inOut[1].type == inOut[0].type &&
             inOut[1].format == inOut[0].format;
  }
  return false;
}

void TorchUnfoldPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) PLUGIN_NOEXCEPT {}

size_t TorchUnfoldPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const PLUGIN_NOEXCEPT {
  return 0;
}

int TorchUnfoldPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace,
    cudaStream_t stream) PLUGIN_NOEXCEPT {
  nvinfer1::Dims input_dims = inputDesc[0].dims;

  auto data_type = inputDesc[0].type;

  int64_t batch_size = input_dims.d[0];
  int64_t channels = input_dims.d[1];
  int64_t height = input_dims.d[2];
  int64_t width = input_dims.d[3];

  int64_t out_height =
      (height + 2 * mPadding[1] - (mDilation[1] * (mKernelSize[1] - 1) + 1)) /
          mStride[1] +
      1;
  int64_t out_width =
      (width + 2 * mPadding[0] - (mDilation[0] * (mKernelSize[0] - 1) + 1)) /
          mStride[0] +
      1;

  for (size_t i = 0; i < size_t(batch_size); ++i) {
    const void *input = nullptr;
    void *output = nullptr;
    switch (data_type) {
      case nvinfer1::DataType::kFLOAT:
        input = (float *)inputs[0] + i * channels * height * width;
        output = (float *)outputs[0] + i * channels * out_height * out_width *
                                           mKernelSize[0] * mKernelSize[1];
        torch_unfold<float>((float *)output, (float *)input, channels, height,
                            width, out_height, out_width, mKernelSize[1],
                            mKernelSize[0], mPadding[1], mPadding[0],
                            mStride[1], mStride[0], mDilation[1], mDilation[0],
                            stream);
        break;
      case nvinfer1::DataType::kINT32:
        input = (int32_t *)inputs[0] + i * channels * height * width;
        output = (int32_t *)outputs[0] + i * channels * out_height * out_width *
                                             mKernelSize[0] * mKernelSize[1];
        torch_unfold<int32_t>((int32_t *)output, (int32_t *)input, channels,
                              height, width, out_height, out_width,
                              mKernelSize[1], mKernelSize[0], mPadding[1],
                              mPadding[0], mStride[1], mStride[0], mDilation[1],
                              mDilation[0], stream);
        break;
      default:
        return 1;
        break;
    }
  }

  // switch (data_type) {
  //   case nvinfer1::DataType::kFLOAT:
  //     torch_gather<float>((float *)outputs[0], (float *)inputs[0],
  //                         (int *)inputs[1], mDim, &(input_dims.d[0]),
  //                         &(index_dims.d[0]), input_dims.nbDims, stream);
  //     break;

  //   case nvinfer1::DataType::kHALF:
  //     torch_gather<half>((half *)outputs[0], (half *)inputs[0],
  //                        (int *)inputs[1], mDim, &(input_dims.d[0]),
  //                        &(index_dims.d[0]), input_dims.nbDims, stream);
  //     break;

  //   case nvinfer1::DataType::kINT32:
  //     torch_gather<int>((int *)outputs[0], (int *)inputs[0], (int
  //     *)inputs[1],
  //                       mDim, &(input_dims.d[0]), &(index_dims.d[0]),
  //                       input_dims.nbDims, stream);
  //     break;
  //   default:
  //     return 1;
  //     break;
  // }

  return 0;
}

nvinfer1::DataType TorchUnfoldPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const PLUGIN_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *TorchUnfoldPluginDynamic::getPluginType() const PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TorchUnfoldPluginDynamic::getPluginVersion() const PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

int TorchUnfoldPluginDynamic::getNbOutputs() const PLUGIN_NOEXCEPT { return 1; }

size_t TorchUnfoldPluginDynamic::getSerializationSize() const PLUGIN_NOEXCEPT {
  return serialized_size(mKernelSize) + serialized_size(mDilation) +
         serialized_size(mPadding) + serialized_size(mStride);
}

void TorchUnfoldPluginDynamic::serialize(void *buffer) const PLUGIN_NOEXCEPT {
  serialize_value(&buffer, mKernelSize);
  serialize_value(&buffer, mDilation);
  serialize_value(&buffer, mPadding);
  serialize_value(&buffer, mStride);
}

////////////////////// creator /////////////////////////////

TorchUnfoldPluginDynamicCreator::TorchUnfoldPluginDynamicCreator() {
  mPluginAttributes = std::vector<PluginField>(
      {PluginField("kernel_size"), PluginField("dilation"),
       PluginField("padding"), PluginField("stride")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TorchUnfoldPluginDynamicCreator::getPluginName() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TorchUnfoldPluginDynamicCreator::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

IPluginV2 *TorchUnfoldPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) PLUGIN_NOEXCEPT {
  std::vector<int32_t> kernelSize;
  std::vector<int32_t> dilation;
  std::vector<int32_t> padding;
  std::vector<int32_t> stride;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("kernel_size") == 0) {
      int data_size = fc->fields[i].length;
      const int *data_start = static_cast<const int *>(fc->fields[i].data);
      kernelSize = std::vector<int32_t>(data_start, data_start + data_size);
    }

    if (field_name.compare("dilation") == 0) {
      int data_size = fc->fields[i].length;
      const int *data_start = static_cast<const int *>(fc->fields[i].data);
      dilation = std::vector<int32_t>(data_start, data_start + data_size);
    }

    if (field_name.compare("padding") == 0) {
      int data_size = fc->fields[i].length;
      const int *data_start = static_cast<const int *>(fc->fields[i].data);
      padding = std::vector<int32_t>(data_start, data_start + data_size);
    }

    if (field_name.compare("stride") == 0) {
      int data_size = fc->fields[i].length;
      const int *data_start = static_cast<const int *>(fc->fields[i].data);
      stride = std::vector<int32_t>(data_start, data_start + data_size);
    }
  }

  TorchUnfoldPluginDynamic *plugin =
      new TorchUnfoldPluginDynamic(name, kernelSize, dilation, padding, stride);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

IPluginV2 *TorchUnfoldPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) PLUGIN_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new TorchUnfoldPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}
}  // namespace plugin
}  // namespace amirstan

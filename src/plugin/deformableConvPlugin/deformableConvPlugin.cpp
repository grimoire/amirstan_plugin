
#include "deformableConvPlugin.h"

#include <assert.h>

#include <chrono>

#include "amirCommon.h"
#include "amir_cuda_util/common_util.h"
#include "common.h"
#include "deform_conv_cuda.h"
#include "serialize.hpp"

namespace amirstan {
namespace plugin {

namespace {
static const char *DCN_VERSION{"1"};
static const char *DCN_NAME{"DeformableConvPluginDynamic"};
}  // namespace

DeformableConvPluginDynamic::DeformableConvPluginDynamic(
    const std::string &name, const nvinfer1::Dims &stride,
    const nvinfer1::Dims &padding, const nvinfer1::Dims &dilation,
    const int deformableGroup, const int group)
    : PluginDynamicBase(name),
      mStride(stride),
      mPadding(padding),
      mDilation(dilation),
      mDeformableGroup(deformableGroup),
      mGroup(group) {
  initialize();
}

DeformableConvPluginDynamic::DeformableConvPluginDynamic(const std::string name,
                                                         const void *data,
                                                         size_t length)
    : PluginDynamicBase(name) {
  deserialize_value(&data, &length, &mStride);
  deserialize_value(&data, &length, &mPadding);
  deserialize_value(&data, &length, &mDilation);
  deserialize_value(&data, &length, &mDeformableGroup);
  deserialize_value(&data, &length, &mGroup);

  initialize();
}

nvinfer1::IPluginV2DynamicExt *DeformableConvPluginDynamic::clone() const
    PLUGIN_NOEXCEPT {
  DeformableConvPluginDynamic *plugin = new DeformableConvPluginDynamic(
      mLayerName, mStride, mPadding, mDilation, mDeformableGroup, mGroup);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs DeformableConvPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[2].d[0];

  ret.d[2] = inputs[1].d[2];
  ret.d[3] = inputs[1].d[3];

  return ret;
}

bool DeformableConvPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  if (pos == 0) {
    return inOut[0].type == DataType::kFLOAT &&
           inOut[0].format == nvinfer1::TensorFormat::kLINEAR;
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }
}

void DeformableConvPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) PLUGIN_NOEXCEPT {}

size_t DeformableConvPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const PLUGIN_NOEXCEPT {
  int sizeof_dtype = samplesCommon::getElementSize(outputs[0].type);

  int batch_size = inputs[0].dims.d[0];
  int nInputPlane = inputs[0].dims.d[1];

  int nOutputPlane = outputs[0].dims.d[1];
  int outputHeight = outputs[0].dims.d[2];
  int outputWidth = outputs[0].dims.d[3];

  int kW = inputs[2].dims.d[2];
  int kH = inputs[2].dims.d[3];
  int im2col_step = std::min(32, batch_size);

  size_t col_size = amirstan::common::getAlignedSize(
      nInputPlane * kW * kH * im2col_step * outputHeight * outputWidth *
      sizeof_dtype);

  size_t out_size = 0;
  if (im2col_step != 1)
    out_size = amirstan::common::getAlignedSize(
        batch_size * nOutputPlane * outputHeight * outputWidth * sizeof_dtype);

  return col_size + out_size;
}

int DeformableConvPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace,
    cudaStream_t stream) PLUGIN_NOEXCEPT {
  int batch_size = inputDesc[0].dims.d[0];
  int inputChannel = inputDesc[0].dims.d[1];
  int inputHeight = inputDesc[0].dims.d[2];
  int inputWidth = inputDesc[0].dims.d[3];
  int mOutDim = inputDesc[2].dims.d[0];
  int kW = inputDesc[2].dims.d[2];
  int kH = inputDesc[2].dims.d[3];

  DCN_PARAMS dcn_params;
  dcn_params.cublas_handle = m_cublas_handle;
  dcn_params.batchSize = batch_size;
  dcn_params.inputChannel = inputChannel;
  dcn_params.inputW = inputWidth;
  dcn_params.inputH = inputHeight;
  dcn_params.outputChannel = mOutDim;
  dcn_params.kernelW = kW;
  dcn_params.kernelH = kH;
  dcn_params.strideW = mStride.d[0];
  dcn_params.strideH = mStride.d[1];
  dcn_params.padW = mPadding.d[0];
  dcn_params.padH = mPadding.d[1];
  dcn_params.dilationW = mDilation.d[0];
  dcn_params.dilationH = mDilation.d[1];
  dcn_params.group = mGroup;
  dcn_params.deformable_group = mDeformableGroup;
  dcn_params.im2col_step = std::min(32, batch_size);

  deform_conv_forward_cuda((float *)inputs[0], (float *)inputs[2], nullptr,
                           (float *)inputs[1], (float *)outputs[0], workSpace,
                           dcn_params, stream);

  return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType DeformableConvPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const PLUGIN_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *DeformableConvPluginDynamic::getPluginType() const PLUGIN_NOEXCEPT {
  return DCN_NAME;
}

const char *DeformableConvPluginDynamic::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return DCN_VERSION;
}

int DeformableConvPluginDynamic::getNbOutputs() const PLUGIN_NOEXCEPT {
  return 1;
}

size_t DeformableConvPluginDynamic::getSerializationSize() const
    PLUGIN_NOEXCEPT {
  return sizeof(mStride) + sizeof(mPadding) + sizeof(mDilation) +
         sizeof(mDeformableGroup) + sizeof(mGroup);
}

void DeformableConvPluginDynamic::serialize(void *buffer) const
    PLUGIN_NOEXCEPT {
  serialize_value(&buffer, mStride);
  serialize_value(&buffer, mPadding);
  serialize_value(&buffer, mDilation);
  serialize_value(&buffer, mDeformableGroup);
  serialize_value(&buffer, mGroup);
}

void DeformableConvPluginDynamic::attachToContext(
    cudnnContext *cudnnContext, cublasContext *cublasContext,
    nvinfer1::IGpuAllocator *gpuAllocator) PLUGIN_NOEXCEPT {
  m_cublas_handle = cublasContext;
}

void DeformableConvPluginDynamic::detachFromContext() PLUGIN_NOEXCEPT {}

////////////////////// creator /////////////////////////////

DeformableConvPluginDynamicCreator::DeformableConvPluginDynamicCreator() {
  mPluginAttributes = std::vector<PluginField>(
      {PluginField("stride"), PluginField("padding"), PluginField("dilation"),
       PluginField("deformable_group"), PluginField("group")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *DeformableConvPluginDynamicCreator::getPluginName() const
    PLUGIN_NOEXCEPT {
  return DCN_NAME;
}

const char *DeformableConvPluginDynamicCreator::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return DCN_VERSION;
}

static nvinfer1::Dims createDims2d(int nbDims, int w, int h) {
  nvinfer1::Dims dim;
  dim.nbDims = nbDims;
  dim.d[1] = h;
  dim.d[0] = w;
  return dim;
}

IPluginV2 *DeformableConvPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) PLUGIN_NOEXCEPT {
  nvinfer1::Dims stride = createDims2d(2, 1, 1);
  nvinfer1::Dims padding = createDims2d(2, 0, 0);
  nvinfer1::Dims dilation = createDims2d(2, 1, 1);
  int deformableGroup = 1;
  int group = 1;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("deformable_group") == 0) {
      deformableGroup = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("group") == 0) {
      group = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("stride") == 0) {
      stride.nbDims = 2;
      stride.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      stride.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }

    if (field_name.compare("padding") == 0) {
      padding.nbDims = 2;
      padding.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      padding.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }

    if (field_name.compare("dilation") == 0) {
      dilation.nbDims = 2;
      dilation.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      dilation.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }
  }

  DeformableConvPluginDynamic *plugin = new DeformableConvPluginDynamic(
      name, stride, padding, dilation, deformableGroup, group);
  return plugin;
}

IPluginV2 *DeformableConvPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) PLUGIN_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new DeformableConvPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

}  // namespace plugin
}  // namespace amirstan

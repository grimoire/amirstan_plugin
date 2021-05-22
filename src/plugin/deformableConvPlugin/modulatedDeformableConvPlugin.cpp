
#include "plugin/deformableConvPlugin/modulatedDeformableConvPlugin.h"

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
static const char *DCN_NAME{"ModulatedDeformableConvPluginDynamic"};
}  // namespace

PluginFieldCollection ModulatedDeformableConvPluginDynamicCreator::mFC{};
std::vector<PluginField>
    ModulatedDeformableConvPluginDynamicCreator::mPluginAttributes(
        {PluginField("stride"), PluginField("padding"), PluginField("dilation"),
         PluginField("deformable_group"), PluginField("group")});

ModulatedDeformableConvPluginDynamic::ModulatedDeformableConvPluginDynamic(
    const std::string &name, const nvinfer1::Dims stride,
    const nvinfer1::Dims padding, const nvinfer1::Dims dilation,
    const int deformableGroup, const int group)
    : mLayerName(name),
      mStride(stride),
      mPadding(padding),
      mDilation(dilation),
      mDeformableGroup(deformableGroup),
      mGroup(group) {
  mWithBias = false;
  initialize();
}

ModulatedDeformableConvPluginDynamic::ModulatedDeformableConvPluginDynamic(
    const std::string name, const void *data, size_t length)
    : mLayerName(name) {
  deserialize_value(&data, &length, &mStride);
  deserialize_value(&data, &length, &mPadding);
  deserialize_value(&data, &length, &mDilation);
  deserialize_value(&data, &length, &mDeformableGroup);
  deserialize_value(&data, &length, &mGroup);
  mWithBias = false;
  initialize();
}

nvinfer1::IPluginV2DynamicExt *ModulatedDeformableConvPluginDynamic::clone()
    const {
  ModulatedDeformableConvPluginDynamic *plugin =
      new ModulatedDeformableConvPluginDynamic(
          mLayerName, mStride, mPadding, mDilation, mDeformableGroup, mGroup);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs ModulatedDeformableConvPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[3].d[0];

  ret.d[2] = inputs[1].d[2];
  ret.d[3] = inputs[1].d[3];

  return ret;
}

bool ModulatedDeformableConvPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) {
  if (pos == 0) {
    return inOut[0].type == DataType::kFLOAT &&
           inOut[0].format == nvinfer1::TensorFormat::kLINEAR;
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }
}

void ModulatedDeformableConvPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {
  if (nbInputs == 5) {
    mWithBias = true;
  }
}

size_t ModulatedDeformableConvPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  /// ALPHA
  int sizeof_dtype = samplesCommon::getElementSize(outputs[0].type);

  int batch_size = inputs[0].dims.d[0];
  int nInputPlane = inputs[0].dims.d[1];
  int inputHeight = inputs[0].dims.d[2];
  int inputWidth = inputs[0].dims.d[3];

  int nOutputPlane = outputs[0].dims.d[1];
  int outputHeight = outputs[0].dims.d[2];
  int outputWidth = outputs[0].dims.d[3];

  int kW = inputs[3].dims.d[2];
  int kH = inputs[3].dims.d[3];
  int im2col_step = std::min(32, batch_size);

  size_t col_size = amirstan::common::getAlignedSize(
      nInputPlane * kW * kH * outputHeight * outputWidth * sizeof_dtype);

  return col_size;
}

int ModulatedDeformableConvPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace, cudaStream_t stream) {
  /// ALPHA

  int batch_size = inputDesc[0].dims.d[0];
  int inputChannel = inputDesc[0].dims.d[1];
  int inputHeight = inputDesc[0].dims.d[2];
  int inputWidth = inputDesc[0].dims.d[3];

  int mOutDim = inputDesc[3].dims.d[0];
  int kW = inputDesc[3].dims.d[2];
  int kH = inputDesc[3].dims.d[3];

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

  float *weight = (float *)inputs[3];
  float *bias = mWithBias ? (float *)inputs[4] : nullptr;

  modulated_deform_conv_cuda_forward(
      (float *)inputs[0], weight, bias, (float *)inputs[1], (float *)inputs[2],
      (float *)outputs[0], workSpace, dcn_params, stream);

  return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType ModulatedDeformableConvPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *ModulatedDeformableConvPluginDynamic::getPluginType() const {
  return DCN_NAME;
}

const char *ModulatedDeformableConvPluginDynamic::getPluginVersion() const {
  return DCN_VERSION;
}

int ModulatedDeformableConvPluginDynamic::getNbOutputs() const { return 1; }

int ModulatedDeformableConvPluginDynamic::initialize() { return 0; }

void ModulatedDeformableConvPluginDynamic::terminate() {}

size_t ModulatedDeformableConvPluginDynamic::getSerializationSize() const {
  return sizeof(mStride) + sizeof(mPadding) + sizeof(mDilation) +
         sizeof(mDeformableGroup) + sizeof(mGroup);
}

void ModulatedDeformableConvPluginDynamic::serialize(void *buffer) const {
  serialize_value(&buffer, mStride);
  serialize_value(&buffer, mPadding);
  serialize_value(&buffer, mDilation);
  serialize_value(&buffer, mDeformableGroup);
  serialize_value(&buffer, mGroup);
}

void ModulatedDeformableConvPluginDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void ModulatedDeformableConvPluginDynamic::attachToContext(
    cudnnContext *cudnnContext, cublasContext *cublasContext,
    nvinfer1::IGpuAllocator *gpuAllocator) {
  m_cublas_handle = cublasContext;
}

void ModulatedDeformableConvPluginDynamic::detachFromContext() {}

void ModulatedDeformableConvPluginDynamic::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *ModulatedDeformableConvPluginDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

ModulatedDeformableConvPluginDynamicCreator::
    ModulatedDeformableConvPluginDynamicCreator() {
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *ModulatedDeformableConvPluginDynamicCreator::getPluginName() const {
  return DCN_NAME;
}

const char *ModulatedDeformableConvPluginDynamicCreator::getPluginVersion()
    const {
  return DCN_VERSION;
}

const PluginFieldCollection *
ModulatedDeformableConvPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

IPluginV2 *ModulatedDeformableConvPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) {
  nvinfer1::Dims stride{2, {1, 1}};
  nvinfer1::Dims padding{2, {0, 0}};
  nvinfer1::Dims dilation{2, {1, 1}};
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

  ModulatedDeformableConvPluginDynamic *plugin =
      new ModulatedDeformableConvPluginDynamic(name, stride, padding, dilation,
                                               deformableGroup, group);

  return plugin;
}

IPluginV2 *ModulatedDeformableConvPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin =
      new ModulatedDeformableConvPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void ModulatedDeformableConvPluginDynamicCreator::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *ModulatedDeformableConvPluginDynamicCreator::getPluginNamespace()
    const {
  return mNamespace.c_str();
}

}  // namespace plugin
}  // namespace amirstan

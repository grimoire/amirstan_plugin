
#include "adaptivePoolPlugin.h"

#include <assert.h>

#include <chrono>

#include "adaptive_pool.h"
#include "amirCommon.h"
#include "amir_cuda_util/cuda_util.h"
#include "common.h"
#include "serialize.hpp"

namespace amirstan {
namespace plugin {

enum class PoolingType : int32_t {
  kMAX = 0,      // Maximum over elements
  kAVERAGE = 1,  // Average over elements. If the tensor is padded, the count
                 // includes the padding
  kMAX_AVERAGE_BLEND = 2  // Blending between max and average pooling:
                          // (1-blendFactor)*maxPool + blendFactor*avgPool
};

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"AdaptivePoolPluginDynamic"};
}  // namespace

AdaptivePoolPluginDynamic::AdaptivePoolPluginDynamic(
    const std::string &name, const nvinfer1::Dims &outputSize, int poolingType)
    : PluginDynamicBase(name),
      mOutputSize(outputSize),
      mPoolingType(poolingType) {}

AdaptivePoolPluginDynamic::AdaptivePoolPluginDynamic(const std::string name,
                                                     const void *data,
                                                     size_t length)
    : PluginDynamicBase(name) {
  deserialize_value(&data, &length, &mOutputSize);
  deserialize_value(&data, &length, &mPoolingType);
}

nvinfer1::IPluginV2DynamicExt *AdaptivePoolPluginDynamic::clone() const
    PLUGIN_NOEXCEPT {
  AdaptivePoolPluginDynamic *plugin =
      new AdaptivePoolPluginDynamic(mLayerName, mOutputSize, mPoolingType);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs AdaptivePoolPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  int nbdims_input = inputs[0].nbDims;
  ret.nbDims = nbdims_input;

  if (nbInputs > 1) {
    int reduce_dims = inputs[1].nbDims;
    for (int i = 0; i < nbdims_input - reduce_dims; ++i) {
      ret.d[i] = inputs[0].d[i];
    }

    for (int i = nbdims_input - reduce_dims; i < nbdims_input; ++i) {
      ret.d[i] = inputs[1].d[i - nbdims_input + reduce_dims];
    }

  } else {
    int reduce_dims = mOutputSize.nbDims;

    for (int i = 0; i < nbdims_input - reduce_dims; ++i) {
      ret.d[i] = inputs[0].d[i];
    }
    for (int i = nbdims_input - reduce_dims; i < nbdims_input; ++i) {
      int o_size = mOutputSize.d[i - nbdims_input + reduce_dims];
      if (o_size > 0) {
        ret.d[i] = exprBuilder.constant(o_size);
      } else {
        ret.d[i] = inputs[0].d[i];
      }
    }
  }

  return ret;
}

bool AdaptivePoolPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
}

void AdaptivePoolPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  // Validate input arguments
  assert(nbOutputs == 1);
}

size_t AdaptivePoolPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const PLUGIN_NOEXCEPT {
  return 0;
}

int AdaptivePoolPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace,
    cudaStream_t stream) PLUGIN_NOEXCEPT {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  nvinfer1::Dims output_dims = outputDesc[0].dims;

  amirstan::plugin::PoolType pool_type = amirstan::plugin::PoolType::MAX;
  switch (mPoolingType) {
    case int(PoolingType::kMAX):
      pool_type = amirstan::plugin::PoolType::MAX;
      break;
    case int(PoolingType::kAVERAGE):
      pool_type = amirstan::plugin::PoolType::AVERAGE;
      break;
    default:
      break;
  }

  amirstan::plugin::adaptive_pool<float>((float *)outputs[0],
                                         (float *)inputs[0], &(input_dims.d[0]),
                                         &(output_dims.d[0]), input_dims.nbDims,
                                         mOutputSize.nbDims, pool_type, stream);

  return 0;
}

nvinfer1::DataType AdaptivePoolPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const PLUGIN_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *AdaptivePoolPluginDynamic::getPluginType() const PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *AdaptivePoolPluginDynamic::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

int AdaptivePoolPluginDynamic::getNbOutputs() const PLUGIN_NOEXCEPT {
  return 1;
}

size_t AdaptivePoolPluginDynamic::getSerializationSize() const PLUGIN_NOEXCEPT {
  return sizeof(mOutputSize) + sizeof(mPoolingType);
}

void AdaptivePoolPluginDynamic::serialize(void *buffer) const PLUGIN_NOEXCEPT {
  serialize_value(&buffer, mOutputSize);
  serialize_value(&buffer, mPoolingType);
}

////////////////////// creator /////////////////////////////

AdaptivePoolPluginDynamicCreator::AdaptivePoolPluginDynamicCreator() {
  mPluginAttributes = std::vector<nvinfer1::PluginField>(
      {nvinfer1::PluginField("output_size"),
       nvinfer1::PluginField("pooling_type")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *AdaptivePoolPluginDynamicCreator::getPluginName() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *AdaptivePoolPluginDynamicCreator::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

IPluginV2 *AdaptivePoolPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) PLUGIN_NOEXCEPT {
  nvinfer1::Dims outputSize;
  int poolingType = 0;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("output_size") == 0) {
      outputSize.nbDims = fc->fields[i].length;
      memcpy(&(outputSize.d[0]), static_cast<const int *>(fc->fields[i].data),
             fc->fields[i].length * sizeof(int));
    }

    if (field_name.compare("pooling_type") == 0) {
      poolingType = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }

  AdaptivePoolPluginDynamic *plugin =
      new AdaptivePoolPluginDynamic(name, outputSize, poolingType);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

IPluginV2 *AdaptivePoolPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) PLUGIN_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new AdaptivePoolPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}
}  // namespace plugin
}  // namespace amirstan

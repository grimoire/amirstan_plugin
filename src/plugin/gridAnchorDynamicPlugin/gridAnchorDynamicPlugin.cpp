
#include "plugin/gridAnchorDynamicPlugin/gridAnchorDynamicPlugin.h"

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
} // namespace

PluginFieldCollection GridAnchorDynamicPluginDynamicCreator::mFC{};
std::vector<PluginField>
    GridAnchorDynamicPluginDynamicCreator::mPluginAttributes(
        {PluginField("base_size"), PluginField("stride")});

GridAnchorDynamicPluginDynamic::GridAnchorDynamicPluginDynamic(
    const std::string &name, int stride)
    : mLayerName(name), mStride(stride) {}

// void set_base_anchor(int rid, int sid, float x_ctr, float y_ctr,
//                      const std::vector<float> &h_ratios,
//                      const std::vector<float> &w_ratios,
//                      const std::vector<float> &scales, int anchor_count,
//                      int baseSize, std::shared_ptr<float> &hostBaseAnchor) {
//   float scale = scales[sid];
//   float h_ratio = h_ratios[rid];
//   float w_ratio = w_ratios[rid];
//   float ws = float(baseSize) * w_ratio * scale;
//   float hs = float(baseSize) * h_ratio * scale;

//   float half_ws = (ws) / 2;
//   float half_hs = (hs) / 2;

//   hostBaseAnchor.get()[anchor_count * 4] = x_ctr - half_ws;
//   hostBaseAnchor.get()[anchor_count * 4 + 1] = y_ctr - half_hs;
//   hostBaseAnchor.get()[anchor_count * 4 + 2] = x_ctr + half_ws;
//   hostBaseAnchor.get()[anchor_count * 4 + 3] = y_ctr + half_hs;
// }

// void GridAnchorDynamicPluginDynamic::generateBaseAnchor() {
//   int num_scales = mScales.size();
//   int num_ratios = mRatios.size();
//   float x_ctr = mCenterX;
//   float y_ctr = mCenterY;
//   if (x_ctr < 0) {
//     x_ctr = (mBaseSize - 1) / 2;
//   }
//   if (y_ctr < 0) {
//     y_ctr = (mBaseSize - 1) / 2;
//   }

//   std::vector<float> h_ratios, w_ratios;
//   h_ratios.resize(num_ratios);
//   w_ratios.resize(num_ratios);
//   std::transform(mRatios.begin(), mRatios.end(), h_ratios.begin(),
//                  [](float x) -> float { return sqrt(x); });
//   std::transform(h_ratios.begin(), h_ratios.end(), w_ratios.begin(),
//                  [](float x) -> float { return 1 / x; });
//   mNumBaseAnchor = num_scales * num_ratios;
//   mHostBaseAnchor = std::shared_ptr<float>(
//       (float *)malloc(mNumBaseAnchor * 4 * sizeof(float)));
//   int anchor_count = 0;
//   if (mScaleMajor) {
//     for (int rid = 0; rid < num_ratios; ++rid) {
//       for (int sid = 0; sid < num_scales; ++sid) {
//         set_base_anchor(rid, sid, x_ctr, y_ctr, h_ratios, w_ratios, mScales,
//                         anchor_count, mBaseSize, mHostBaseAnchor);
//         anchor_count += 1;
//       }
//     }
//   } else {
//     for (int sid = 0; sid < num_scales; ++sid) {
//       for (int rid = 0; rid < num_ratios; ++rid) {
//         set_base_anchor(rid, sid, x_ctr, y_ctr, h_ratios, w_ratios, mScales,
//                         anchor_count, mBaseSize, mHostBaseAnchor);
//         anchor_count += 1;
//       }
//     }
//   }
// }

GridAnchorDynamicPluginDynamic::GridAnchorDynamicPluginDynamic(
    const std::string name, const void *data, size_t length)
    : mLayerName(name) {
  deserialize_value(&data, &length, &mStride);
}

nvinfer1::IPluginV2DynamicExt *GridAnchorDynamicPluginDynamic::clone() const {
  GridAnchorDynamicPluginDynamic *plugin =
      new GridAnchorDynamicPluginDynamic(mLayerName, mStride);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs GridAnchorDynamicPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
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
    int nbOutputs) {
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
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {
  // Validate input arguments
}

size_t GridAnchorDynamicPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  return 0;
}

int GridAnchorDynamicPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace, cudaStream_t stream) {
  int batch_size = inputDesc[0].dims.d[0];
  int inputChannel = inputDesc[0].dims.d[1];
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
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods
const char *GridAnchorDynamicPluginDynamic::getPluginType() const {
  return PLUGIN_NAME;
}

const char *GridAnchorDynamicPluginDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

int GridAnchorDynamicPluginDynamic::getNbOutputs() const { return 1; }

int GridAnchorDynamicPluginDynamic::initialize() { return 0; }

void GridAnchorDynamicPluginDynamic::terminate() {}

size_t GridAnchorDynamicPluginDynamic::getSerializationSize() const {
  return sizeof(mStride);
}

void GridAnchorDynamicPluginDynamic::serialize(void *buffer) const {
  const void *start = buffer;
  serialize_value(&buffer, mStride);
}

void GridAnchorDynamicPluginDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void GridAnchorDynamicPluginDynamic::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *GridAnchorDynamicPluginDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

GridAnchorDynamicPluginDynamicCreator::GridAnchorDynamicPluginDynamicCreator() {
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *GridAnchorDynamicPluginDynamicCreator::getPluginName() const {
  return PLUGIN_NAME;
}

const char *GridAnchorDynamicPluginDynamicCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const PluginFieldCollection *
GridAnchorDynamicPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

IPluginV2 *GridAnchorDynamicPluginDynamicCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) {
  int stride;

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
    const char *name, const void *serialData, size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin =
      new GridAnchorDynamicPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void GridAnchorDynamicPluginDynamicCreator::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *GridAnchorDynamicPluginDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan

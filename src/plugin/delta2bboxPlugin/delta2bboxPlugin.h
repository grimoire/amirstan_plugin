#pragma once

#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>

#include "../common/layer_base.h"


namespace amirstan {
namespace plugin {

class Delta2BBoxPluginDynamic : public PluginDynamicBase {
 public:
  Delta2BBoxPluginDynamic(const std::string &name, int minNumBBox,
                          const std::vector<float> &targetMeans,
                          const std::vector<float> &targetStds);

  Delta2BBoxPluginDynamic(const std::string name, const void *data,
                          size_t length);

  // It doesn't make sense to make Delta2BBoxPluginDynamic without arguments, so
  // we delete default constructor.
  Delta2BBoxPluginDynamic() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const PLUGIN_NOEXCEPT override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
      nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT override;
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc *inOut,
                                 int nbInputs,
                                 int nbOutputs) PLUGIN_NOEXCEPT override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *outputs,
                       int nbOutputs) PLUGIN_NOEXCEPT override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs,
                          int nbOutputs) const PLUGIN_NOEXCEPT override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
              const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace,
              cudaStream_t stream) PLUGIN_NOEXCEPT override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType *inputTypes,
      int nbInputs) const PLUGIN_NOEXCEPT override;

  // IPluginV2 Methods
  const char *getPluginType() const PLUGIN_NOEXCEPT override;
  const char *getPluginVersion() const PLUGIN_NOEXCEPT override;
  int getNbOutputs() const PLUGIN_NOEXCEPT override;
  size_t getSerializationSize() const PLUGIN_NOEXCEPT override;
  void serialize(void *buffer) const PLUGIN_NOEXCEPT override;

 private:
  int mMinNumBBox;
  std::vector<float> mTargetMeans;
  std::vector<float> mTargetStds;
  bool mNeedClipRange;
};

class Delta2BBoxPluginDynamicCreator : public PluginCreatorBase {
 public:
  Delta2BBoxPluginDynamicCreator();

  const char *getPluginName() const PLUGIN_NOEXCEPT override;

  const char *getPluginVersion() const PLUGIN_NOEXCEPT override;

  nvinfer1::IPluginV2 *createPlugin(const char *name,
                                    const nvinfer1::PluginFieldCollection *fc)
      PLUGIN_NOEXCEPT override;

  nvinfer1::IPluginV2 *deserializePlugin(
      const char *name, const void *serialData,
      size_t serialLength) PLUGIN_NOEXCEPT override;
};

}  // namespace plugin
}  // namespace amirstan

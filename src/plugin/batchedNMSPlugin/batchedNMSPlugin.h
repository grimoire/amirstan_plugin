/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_BATCHED_NMS_PLUGIN_CUSTOM_H
#define TRT_BATCHED_NMS_PLUGIN_CUSTOM_H
#include <string>
#include <vector>

#include "../common/layer_base.h"
#include "NvInferPluginUtils.h"

namespace amirstan {
namespace plugin {

class BatchedNMSPluginCustom : public PluginDynamicBase {
 public:
  BatchedNMSPluginCustom(nvinfer1::plugin::NMSParameters param);

  BatchedNMSPluginCustom(const void* data, size_t length);

  ~BatchedNMSPluginCustom() PLUGIN_NOEXCEPT override = default;

  int getNbOutputs() const PLUGIN_NOEXCEPT override;

  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) PLUGIN_NOEXCEPT override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const PLUGIN_NOEXCEPT override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workSpace,
              cudaStream_t stream) PLUGIN_NOEXCEPT override;

  size_t getSerializationSize() const PLUGIN_NOEXCEPT override;

  void serialize(void* buffer) const PLUGIN_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* outputs,
                       int nbOutputs) PLUGIN_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) PLUGIN_NOEXCEPT override;

  const char* getPluginType() const PLUGIN_NOEXCEPT override;

  const char* getPluginVersion() const PLUGIN_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt* clone() const PLUGIN_NOEXCEPT override;

  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* inputType,
      int nbInputs) const PLUGIN_NOEXCEPT override;
  void setClipParam(bool clip);

 private:
  nvinfer1::plugin::NMSParameters param{};
  int boxesSize{};
  int scoresSize{};
  int numPriors{};
  bool mClipBoxes{};
};

class BatchedNMSPluginCustomCreator : public PluginCreatorBase {
 public:
  BatchedNMSPluginCustomCreator();

  ~BatchedNMSPluginCustomCreator() PLUGIN_NOEXCEPT override = default;

  const char* getPluginName() const PLUGIN_NOEXCEPT override;

  const char* getPluginVersion() const PLUGIN_NOEXCEPT override;

  nvinfer1::IPluginV2Ext* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) PLUGIN_NOEXCEPT override;

  nvinfer1::IPluginV2Ext* deserializePlugin(
      const char* name, const void* serialData,
      size_t serialLength) PLUGIN_NOEXCEPT override;

 private:
  nvinfer1::plugin::NMSParameters params;
  bool mClipBoxes;
};

}  // namespace plugin
}  // namespace amirstan

#endif  // TRT_BATCHED_NMS_PLUGIN_CUSTOM_H

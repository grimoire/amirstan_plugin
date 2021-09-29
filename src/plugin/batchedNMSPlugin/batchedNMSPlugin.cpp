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

#include "batchedNMSPlugin.h"

#include <cstring>

#include "batchedNMSInference.h"
#include "nmsUtils.h"
#include "serialize.hpp"

using namespace nvinfer1;
using amirstan::plugin::BatchedNMSPluginCustom;
using amirstan::plugin::BatchedNMSPluginCustomCreator;
using nvinfer1::plugin::NMSParameters;

namespace {
static const char* NMS_PLUGIN_VERSION{"1"};
static const char* NMS_PLUGIN_NAME{"BatchedNMS_TRT_CUSTOM"};
}  // namespace

BatchedNMSPluginCustom::BatchedNMSPluginCustom(NMSParameters params)
    : PluginDynamicBase(NMS_PLUGIN_NAME), param(params) {}

BatchedNMSPluginCustom::BatchedNMSPluginCustom(const void* data, size_t length)
    : PluginDynamicBase(NMS_PLUGIN_NAME) {
  deserialize_value(&data, &length, &param);
  deserialize_value(&data, &length, &boxesSize);
  deserialize_value(&data, &length, &scoresSize);
  deserialize_value(&data, &length, &numPriors);
  deserialize_value(&data, &length, &mClipBoxes);
}

int BatchedNMSPluginCustom::getNbOutputs() const PLUGIN_NOEXCEPT { return 4; }

nvinfer1::DimsExprs BatchedNMSPluginCustom::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) PLUGIN_NOEXCEPT {
  ASSERT(nbInputs == 2);
  ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());
  ASSERT(inputs[0].nbDims == 4);
  ASSERT(inputs[1].nbDims == 3);

  nvinfer1::DimsExprs ret;
  switch (outputIndex) {
    case 0:
      ret.nbDims = 2;
      break;
    case 1:
      ret.nbDims = 3;
      break;
    case 2:
    case 3:
      ret.nbDims = 2;
      break;
    default:
      break;
  }

  ret.d[0] = inputs[0].d[0];
  if (outputIndex == 0) {
    ret.d[1] = exprBuilder.constant(1);
  }
  if (outputIndex > 0) {
    ret.d[1] = exprBuilder.constant(param.keepTopK);
  }
  if (outputIndex == 1) {
    ret.d[2] = exprBuilder.constant(4);
  }

  return ret;
}

size_t BatchedNMSPluginCustom::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const PLUGIN_NOEXCEPT {
  size_t batch_size = inputs[0].dims.d[0];
  size_t boxes_size =
      inputs[0].dims.d[1] * inputs[0].dims.d[2] * inputs[0].dims.d[3];
  size_t score_size = inputs[1].dims.d[1] * inputs[1].dims.d[2];
  size_t num_priors = inputs[0].dims.d[1];
  return detectionInferenceWorkspaceSize(
      param.shareLocation, batch_size, boxes_size, score_size, param.numClasses,
      num_priors, param.topK, DataType::kFLOAT, DataType::kFLOAT);
}

int BatchedNMSPluginCustom::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workSpace,
    cudaStream_t stream) PLUGIN_NOEXCEPT {
  const void* const locData = inputs[0];
  const void* const confData = inputs[1];

  void* keepCount = outputs[0];
  void* nmsedBoxes = outputs[1];
  void* nmsedScores = outputs[2];
  void* nmsedClasses = outputs[3];

  size_t batch_size = inputDesc[0].dims.d[0];
  size_t boxes_size =
      inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
  size_t score_size = inputDesc[1].dims.d[1] * inputDesc[1].dims.d[2];
  size_t num_priors = inputDesc[0].dims.d[1];

  pluginStatus_t status = nmsInference(
      stream, batch_size, boxes_size, score_size, param.shareLocation,
      param.backgroundLabelId, num_priors, param.numClasses, param.topK,
      param.keepTopK, param.scoreThreshold, param.iouThreshold,
      DataType::kFLOAT, locData, DataType::kFLOAT, confData, keepCount,
      nmsedBoxes, nmsedScores, nmsedClasses, workSpace, param.isNormalized,
      false, mClipBoxes);
  ASSERT(status == STATUS_SUCCESS);

  return 0;
}

size_t BatchedNMSPluginCustom::getSerializationSize() const PLUGIN_NOEXCEPT {
  // NMSParameters, boxesSize,scoresSize,numPriors
  return sizeof(NMSParameters) + sizeof(int) * 3 + sizeof(bool);
}

void BatchedNMSPluginCustom::serialize(void* buffer) const PLUGIN_NOEXCEPT {
  serialize_value(&buffer, param);
  serialize_value(&buffer, boxesSize);
  serialize_value(&buffer, scoresSize);
  serialize_value(&buffer, numPriors);
  serialize_value(&buffer, mClipBoxes);
}

void BatchedNMSPluginCustom::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  // Validate input arguments
}

bool BatchedNMSPluginCustom::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  if (pos == 2) {
    return inOut[pos].type == nvinfer1::DataType::kINT32 &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

const char* BatchedNMSPluginCustom::getPluginType() const PLUGIN_NOEXCEPT {
  return NMS_PLUGIN_NAME;
}

const char* BatchedNMSPluginCustom::getPluginVersion() const PLUGIN_NOEXCEPT {
  return NMS_PLUGIN_VERSION;
}

IPluginV2DynamicExt* BatchedNMSPluginCustom::clone() const PLUGIN_NOEXCEPT {
  auto* plugin = new BatchedNMSPluginCustom(param);
  plugin->boxesSize = boxesSize;
  plugin->scoresSize = scoresSize;
  plugin->numPriors = numPriors;
  plugin->setPluginNamespace(mNamespace.c_str());
  plugin->setClipParam(mClipBoxes);
  return plugin;
}

nvinfer1::DataType BatchedNMSPluginCustom::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes,
    int nbInputs) const PLUGIN_NOEXCEPT {
  if (index == 0) {
    return nvinfer1::DataType::kINT32;
  }
  return inputTypes[0];
}

void BatchedNMSPluginCustom::setClipParam(bool clip) { mClipBoxes = clip; }

// bool BatchedNMSPluginCustom::isOutputBroadcastAcrossBatch(int outputIndex,
// const bool* inputIsBroadcasted, int nbInputs) const
// {
//     return false;
// }

// bool BatchedNMSPluginCustom::canBroadcastInputAcrossBatch(int inputIndex)
// const
// {
//     return false;
// }

BatchedNMSPluginCustomCreator::BatchedNMSPluginCustomCreator() : params{} {
  mPluginAttributes = std::vector<PluginField>();
  mPluginAttributes.emplace_back(
      PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("scoreThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("iouThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("clipBoxes", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* BatchedNMSPluginCustomCreator::getPluginName() const
    PLUGIN_NOEXCEPT {
  return NMS_PLUGIN_NAME;
}

const char* BatchedNMSPluginCustomCreator::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return NMS_PLUGIN_VERSION;
}

IPluginV2Ext* BatchedNMSPluginCustomCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) PLUGIN_NOEXCEPT {
  const PluginField* fields = fc->fields;
  mClipBoxes = true;

  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "shareLocation")) {
      params.shareLocation = *(static_cast<const bool*>(fields[i].data));
    } else if (!strcmp(attrName, "backgroundLabelId")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.backgroundLabelId = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "numClasses")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.numClasses = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "topK")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.topK = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "keepTopK")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.keepTopK = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "scoreThreshold")) {
      ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
      params.scoreThreshold = *(static_cast<const float*>(fields[i].data));
    } else if (!strcmp(attrName, "iouThreshold")) {
      ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
      params.iouThreshold = *(static_cast<const float*>(fields[i].data));
    } else if (!strcmp(attrName, "isNormalized")) {
      params.isNormalized = *(static_cast<const bool*>(fields[i].data));
    } else if (!strcmp(attrName, "clipBoxes")) {
      mClipBoxes = *(static_cast<const bool*>(fields[i].data));
    }
  }

  BatchedNMSPluginCustom* plugin = new BatchedNMSPluginCustom(params);
  plugin->setClipParam(mClipBoxes);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

IPluginV2Ext* BatchedNMSPluginCustomCreator::deserializePlugin(
    const char* name, const void* serialData,
    size_t serialLength) PLUGIN_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call NMS::destroy()
  BatchedNMSPluginCustom* plugin =
      new BatchedNMSPluginCustom(serialData, serialLength);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

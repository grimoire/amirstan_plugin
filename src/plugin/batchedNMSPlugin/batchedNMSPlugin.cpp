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
#include "batchedNMSInference.h"
#include "gatherNMSOutputs.h"
#include "kernel.h"
#include "nmsUtils.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include "plugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::BatchedNMSPlugin;
using nvinfer1::plugin::BatchedNMSPluginCreator;
using nvinfer1::plugin::NMSParameters;

namespace
{
static const char* NMS_PLUGIN_VERSION{"1"};
static const char* NMS_PLUGIN_NAME{"BatchedNMS_TRT_CUSTOM"};
} // namespace

PluginFieldCollection BatchedNMSPluginCreator::mFC{};
std::vector<PluginField> BatchedNMSPluginCreator::mPluginAttributes;

BatchedNMSPlugin::BatchedNMSPlugin(NMSParameters params)
    : param(params)
{
}

BatchedNMSPlugin::BatchedNMSPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    param = read<NMSParameters>(d);
    boxesSize = read<int>(d);
    scoresSize = read<int>(d);
    numPriors = read<int>(d);
    mClipBoxes = read<bool>(d);
    ASSERT(d == a + length);
}

int BatchedNMSPlugin::getNbOutputs() const
{
    return 4;
}

int BatchedNMSPlugin::initialize()
{
    return STATUS_SUCCESS;
}

void BatchedNMSPlugin::terminate() {}


nvinfer1::DimsExprs BatchedNMSPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    ASSERT(nbInputs == 2);
    ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    ASSERT(inputs[0].nbDims == 4);
    ASSERT(inputs[1].nbDims == 3);

    nvinfer1::DimsExprs ret; 
    switch(outputIndex){
    case 0:
        ret.nbDims=2;
        break;
    case 1:
        ret.nbDims=3;
        break;
    case 2:
    case 3:
        ret.nbDims=2;
        break;
    default:
        break;
    }

    ret.d[0] = inputs[0].d[0];
    if(outputIndex==0){
        ret.d[1] = exprBuilder.constant(1);
    }
    if(outputIndex>0){
        ret.d[1] = exprBuilder.constant(param.keepTopK);
    }
    if(outputIndex==1){
        ret.d[2] = exprBuilder.constant(4);
    }

    return ret;

}

size_t BatchedNMSPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    size_t batch_size = inputs[0].dims.d[0];
    size_t boxes_size = inputs[0].dims.d[1] * inputs[0].dims.d[2] * inputs[0].dims.d[3];
    size_t score_size = inputs[1].dims.d[1] * inputs[1].dims.d[2];
    size_t num_priors = inputs[0].dims.d[1];
    return detectionInferenceWorkspaceSize(param.shareLocation, batch_size, boxes_size, score_size, param.numClasses,
        num_priors, param.topK, DataType::kFLOAT, DataType::kFLOAT);
}

int BatchedNMSPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{
    
    const void* const locData = inputs[0];
    const void* const confData = inputs[1];

    void* keepCount = outputs[0];
    void* nmsedBoxes = outputs[1];
    void* nmsedScores = outputs[2];
    void* nmsedClasses = outputs[3];
    
    size_t batch_size = inputDesc[0].dims.d[0];
    size_t boxes_size = inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
    size_t score_size = inputDesc[1].dims.d[1] * inputDesc[1].dims.d[2];
    size_t num_priors = inputDesc[0].dims.d[1];

    pluginStatus_t status = nmsInference(stream, batch_size, boxes_size, score_size, param.shareLocation,
        param.backgroundLabelId, num_priors, param.numClasses, param.topK, param.keepTopK, param.scoreThreshold,
        param.iouThreshold, DataType::kFLOAT, locData, DataType::kFLOAT, confData, keepCount, nmsedBoxes, nmsedScores,
        nmsedClasses, workSpace, param.isNormalized, false, mClipBoxes);
    ASSERT(status == STATUS_SUCCESS);

    return 0;
}

size_t BatchedNMSPlugin::getSerializationSize() const
{
    // NMSParameters, boxesSize,scoresSize,numPriors
    return sizeof(NMSParameters) + sizeof(int) * 3 + sizeof(bool);
}

void BatchedNMSPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, param);
    write(d, boxesSize);
    write(d, scoresSize);
    write(d, numPriors);
    write(d, mClipBoxes);
    ASSERT(d == a + getSerializationSize());
}

void BatchedNMSPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
}


bool BatchedNMSPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    if(pos==2){
        return inOut[pos].type == nvinfer1::DataType::kINT32 &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }
    return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
        inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

const char* BatchedNMSPlugin::getPluginType() const
{
    return NMS_PLUGIN_NAME;
}

const char* BatchedNMSPlugin::getPluginVersion() const
{
    return NMS_PLUGIN_VERSION;
}

void BatchedNMSPlugin::destroy()
{
    delete this;
}

IPluginV2DynamicExt* BatchedNMSPlugin::clone() const
{
    auto* plugin = new BatchedNMSPlugin(param);
    plugin->boxesSize = boxesSize;
    plugin->scoresSize = scoresSize;
    plugin->numPriors = numPriors;
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->setClipParam(mClipBoxes);
    return plugin;
}

void BatchedNMSPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char* BatchedNMSPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

nvinfer1::DataType BatchedNMSPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    if (index == 0)
    {
        return nvinfer1::DataType::kINT32;
    }
    return inputTypes[0];
}

void BatchedNMSPlugin::setClipParam(bool clip)
{
    mClipBoxes = clip;
}

// bool BatchedNMSPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
// {
//     return false;
// }

// bool BatchedNMSPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
// {
//     return false;
// }

BatchedNMSPluginCreator::BatchedNMSPluginCreator()
    : params{}
{
    mPluginAttributes.emplace_back(PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("scoreThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iouThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipBoxes", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* BatchedNMSPluginCreator::getPluginName() const
{
    return NMS_PLUGIN_NAME;
}

const char* BatchedNMSPluginCreator::getPluginVersion() const
{
    return NMS_PLUGIN_VERSION;
}

const PluginFieldCollection* BatchedNMSPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* BatchedNMSPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    mClipBoxes = true;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "shareLocation"))
        {
            params.shareLocation = *(static_cast<const bool*>(fields[i].data));
        }
        else if (!strcmp(attrName, "backgroundLabelId"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.backgroundLabelId = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "numClasses"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.numClasses = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "topK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.topK = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "keepTopK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.keepTopK = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "scoreThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.scoreThreshold = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "iouThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.iouThreshold = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "isNormalized"))
        {
            params.isNormalized = *(static_cast<const bool*>(fields[i].data));
        }
        else if (!strcmp(attrName, "clipBoxes"))
        {
            mClipBoxes = *(static_cast<const bool*>(fields[i].data));
        }
    }

    BatchedNMSPlugin* plugin = new BatchedNMSPlugin(params);
    plugin->setClipParam(mClipBoxes);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* BatchedNMSPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call NMS::destroy()
    BatchedNMSPlugin* plugin = new BatchedNMSPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void BatchedNMSPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* BatchedNMSPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
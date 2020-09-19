
#include <assert.h>
#include <chrono>
#include "plugin/torchNMSPlugin/torchNMSPlugin.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "amir_cuda_util/cuda_util.h"
#include "torch_nms.h"


namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"TorchNMSPluginDynamic"};
} // namespace

PluginFieldCollection TorchNMSPluginDynamicCreator::mFC{};
std::vector<PluginField> TorchNMSPluginDynamicCreator::mPluginAttributes({PluginField("iou_threshold")});

TorchNMSPluginDynamic::TorchNMSPluginDynamic(
    const std::string &name,
        float iouThreshold)
    : mLayerName(name),
      mIouThreshold(iouThreshold)
{}

TorchNMSPluginDynamic::TorchNMSPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mIouThreshold);
}

nvinfer1::IPluginV2DynamicExt *TorchNMSPluginDynamic::clone() const
{
    TorchNMSPluginDynamic *plugin = new TorchNMSPluginDynamic(mLayerName,
                                                            mIouThreshold);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs TorchNMSPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    nvinfer1::DimsExprs ret = inputs[1];
    return ret;
}

bool TorchNMSPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;

    switch (pos)
    {
    case 0:
        return (in[0].type == nvinfer1::DataType::kFLOAT && in[0].format == nvinfer1::TensorFormat::kLINEAR);
    case 1:
        return (in[1].type == nvinfer1::DataType::kFLOAT && in[1].format == nvinfer1::TensorFormat::kLINEAR);
    case 2:
        return (out[0].type == nvinfer1::DataType::kINT32 && out[0].format == nvinfer1::TensorFormat::kLINEAR);
    }
}

void TorchNMSPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
}

size_t TorchNMSPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    int num_bbox = inputs[1].dims.d[0]; 
    auto data_type = inputs[1].type;
    size_t score_workspace = 0;
    size_t nms_workspace = 0;
    switch(data_type){
    case nvinfer1::DataType::kFLOAT:
        score_workspace = num_bbox*sizeof(float)*2;
        nms_workspace = nms_workspace_size<float>(num_bbox) + 1;
        break;
    default:
        break;
    }
    return score_workspace
        + num_bbox*sizeof(int)
        + nms_workspace;
}

int TorchNMSPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{

    nvinfer1::Dims score_dims = inputDesc[1].dims;
    nvinfer1::Dims output_dims = outputDesc[0].dims;

    int num_boxes = score_dims.d[0];

    auto data_type = inputDesc[1].type;
    switch(data_type){
    case nvinfer1::DataType::kFLOAT:
        torch_nms<float>((int*)outputs[0], (float*)inputs[0], (float*)inputs[1],
                  num_boxes, mIouThreshold, workSpace,
                  stream);
        break;
    default:
        return 1;
        break;
    }

    return 0;
}

nvinfer1::DataType TorchNMSPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    return nvinfer1::DataType::kINT32;
}

// IPluginV2 Methods
const char *TorchNMSPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *TorchNMSPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int TorchNMSPluginDynamic::getNbOutputs() const
{
    return 1;
}

int TorchNMSPluginDynamic::initialize()
{
    return 0;
}

void TorchNMSPluginDynamic::terminate()
{
}

size_t TorchNMSPluginDynamic::getSerializationSize() const
{
    return sizeof(mIouThreshold);
}

void TorchNMSPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mIouThreshold);
}

void TorchNMSPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void TorchNMSPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *TorchNMSPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

TorchNMSPluginDynamicCreator::TorchNMSPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *TorchNMSPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *TorchNMSPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *TorchNMSPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *TorchNMSPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    float iou_threshold=0.;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if(field_name.compare("iou_threshold")==0){
            iou_threshold = static_cast<const float *>(fc->fields[i].data)[0];
        }
    }

    TorchNMSPluginDynamic *plugin = new TorchNMSPluginDynamic(name,
                                                              iou_threshold);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *TorchNMSPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new TorchNMSPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void TorchNMSPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *TorchNMSPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
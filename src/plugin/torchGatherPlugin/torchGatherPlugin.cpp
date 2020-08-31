
#include <assert.h>
#include <chrono>
#include "plugin/torchGatherPlugin/torchGatherPlugin.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "amir_cuda_util/cuda_util.h"
#include "torch_gather.h"


namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"TorchGatherPluginDynamic"};
} // namespace

PluginFieldCollection TorchGatherPluginDynamicCreator::mFC{};
std::vector<PluginField> TorchGatherPluginDynamicCreator::mPluginAttributes({PluginField("repeat_dims")});

TorchGatherPluginDynamic::TorchGatherPluginDynamic(
    const std::string &name,
    int dim)
    : mLayerName(name),
      mDim(dim)
{}

TorchGatherPluginDynamic::TorchGatherPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mDim);

    initialize();
}

nvinfer1::IPluginV2DynamicExt *TorchGatherPluginDynamic::clone() const
{
    TorchGatherPluginDynamic *plugin = new TorchGatherPluginDynamic(mLayerName,
                                                    mDim);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs TorchGatherPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    return inputs[1];
}

bool TorchGatherPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    assert(0 <= pos && pos < 3);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;

    switch (pos)
    {
    case 0:
        return (in[0].type == nvinfer1::DataType::kFLOAT && in[0].format == nvinfer1::TensorFormat::kLINEAR)
        || (in[0].type == nvinfer1::DataType::kHALF && in[0].format == nvinfer1::TensorFormat::kLINEAR)
        || (in[0].type == nvinfer1::DataType::kINT32 && in[0].format == nvinfer1::TensorFormat::kLINEAR);
    case 1:
        return in[1].type == nvinfer1::DataType::kINT32 && in[1].format == nvinfer1::TensorFormat::kLINEAR;
    case 2:
        return out[0].type == in[0].type &&
               out[0].format == in[0].format;
    }
}

void TorchGatherPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(nbInputs == 2);
}

size_t TorchGatherPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int TorchGatherPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    nvinfer1::Dims index_dims = inputDesc[1].dims;

    auto data_type = inputDesc[0].type;

    switch(data_type){
    case nvinfer1::DataType::kFLOAT:
        torch_gather<float>((float*)outputs[0], (float*)inputs[0], (int*)inputs[1], 
        mDim, &(input_dims.d[0]), &(index_dims.d[0]), input_dims.nbDims, 
        stream);
        break;

    case nvinfer1::DataType::kHALF:
        torch_gather<half>((half*)outputs[0], (half*)inputs[0], (int*)inputs[1], 
        mDim, &(input_dims.d[0]), &(index_dims.d[0]), input_dims.nbDims, 
        stream);
        break;

    case nvinfer1::DataType::kINT32:
        torch_gather<int>((int*)outputs[0], (int*)inputs[0], (int*)inputs[1], 
        mDim, &(input_dims.d[0]), &(index_dims.d[0]), input_dims.nbDims, 
        stream);
        break;
    default:
        return 1;
        break;
    }

    return 0;
}

nvinfer1::DataType TorchGatherPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    assert(nbInputs == 2);
    return inputTypes[0];
}

// IPluginV2 Methods
const char *TorchGatherPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *TorchGatherPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int TorchGatherPluginDynamic::getNbOutputs() const
{
    return 1;
}

int TorchGatherPluginDynamic::initialize()
{
    return 0;
}

void TorchGatherPluginDynamic::terminate()
{
}

size_t TorchGatherPluginDynamic::getSerializationSize() const
{
    return sizeof(mDim);
}

void TorchGatherPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mDim);
}

void TorchGatherPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void TorchGatherPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *TorchGatherPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

TorchGatherPluginDynamicCreator::TorchGatherPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *TorchGatherPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *TorchGatherPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *TorchGatherPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *TorchGatherPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    int dim=0;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("dim") == 0)
        {
            dim = static_cast<const int *>(fc->fields[i].data)[0];
        }
    }

    TorchGatherPluginDynamic *plugin = new TorchGatherPluginDynamic(name,
                                                    dim);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *TorchGatherPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new TorchGatherPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void TorchGatherPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *TorchGatherPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan

#include <assert.h>
#include <chrono>
#include "plugin/torchCumPlugin/torchCumPlugin.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "amir_cuda_util/cuda_util.h"

#include "torch_cum.h"


namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"TorchCumPluginDynamic"};
} // namespace

PluginFieldCollection TorchCumPluginDynamicCreator::mFC{};
std::vector<PluginField> TorchCumPluginDynamicCreator::mPluginAttributes({PluginField("mode"),
                                                                            PluginField("dim"),
                                                                            PluginField("cum_type")});

TorchCumPluginDynamic::TorchCumPluginDynamic(
    const std::string &name, 
    int dim, 
    int cumType)
    : mLayerName(name),
      mDim(dim),
      mCumType(cumType)
{}

TorchCumPluginDynamic::TorchCumPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mDim);
    deserialize_value(&data, &length, &mCumType);
}

nvinfer1::IPluginV2DynamicExt *TorchCumPluginDynamic::clone() const
{
    TorchCumPluginDynamic *plugin = new TorchCumPluginDynamic(mLayerName,
                                                            mDim,
                                                            mCumType);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs TorchCumPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    nvinfer1::DimsExprs ret = inputs[0];
    return ret;
}

bool TorchCumPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;

    switch (pos)
    {
    case 0:
        return (in[0].type == nvinfer1::DataType::kFLOAT && in[0].format == nvinfer1::TensorFormat::kLINEAR);
    case 1:
        return out[0].type == in[0].type && 
                out[0].format == in[0].format;
    }
}

void TorchCumPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
}

size_t TorchCumPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int TorchCumPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{

    nvinfer1::Dims input_dims = inputDesc[0].dims;
    nvinfer1::Dims output_dims = outputDesc[0].dims;

    auto data_type = inputDesc[0].type;

    switch(data_type){
    case nvinfer1::DataType::kFLOAT:
        torch_cum<float>((float*)outputs[0], (float*)inputs[0],
                  &(input_dims.d[0]), input_dims.nbDims,
                  mDim, mCumType,
                  stream);
        break;
    default:
        return 1;
        break;
    }

    return 0;
}

nvinfer1::DataType TorchCumPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

// IPluginV2 Methods
const char *TorchCumPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *TorchCumPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int TorchCumPluginDynamic::getNbOutputs() const
{
    return 1;
}

int TorchCumPluginDynamic::initialize()
{
    return 0;
}

void TorchCumPluginDynamic::terminate()
{
}

size_t TorchCumPluginDynamic::getSerializationSize() const
{
    return sizeof(mDim)
    +sizeof(mCumType);
}

void TorchCumPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mDim);
    serialize_value(&buffer, mCumType);
}

void TorchCumPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void TorchCumPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *TorchCumPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

TorchCumPluginDynamicCreator::TorchCumPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *TorchCumPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *TorchCumPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *TorchCumPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *TorchCumPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    int dim=0;
    int cumType=0;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if(field_name.compare("dim")==0){
            dim = static_cast<const int *>(fc->fields[i].data)[0];
        }


        if(field_name.compare("cum_type")==0){
            cumType = static_cast<const int *>(fc->fields[i].data)[0];
        }

    }

    TorchCumPluginDynamic *plugin = new TorchCumPluginDynamic(name,
                                                                dim,
                                                                cumType);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *TorchCumPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new TorchCumPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void TorchCumPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *TorchCumPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
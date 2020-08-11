
#include <assert.h>
#include <chrono>
#include "plugin/torchCumMaxMinPlugin/torchCumMaxMinPlugin.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "amir_cuda_util/cuda_util.h"

#include "torch_cum_maxmin.h"


namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"TorchCumMaxMinPluginDynamic"};
} // namespace

PluginFieldCollection TorchCumMaxMinPluginDynamicCreator::mFC{};
std::vector<PluginField> TorchCumMaxMinPluginDynamicCreator::mPluginAttributes({PluginField("mode"),
                                                                            PluginField("dim"),
                                                                            PluginField("cum_type")});

TorchCumMaxMinPluginDynamic::TorchCumMaxMinPluginDynamic(
    const std::string &name, 
    int dim, 
    int cumType)
    : mLayerName(name),
      mDim(dim),
      mCumType(cumType)
{}

TorchCumMaxMinPluginDynamic::TorchCumMaxMinPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mDim);
    deserialize_value(&data, &length, &mCumType);
}

nvinfer1::IPluginV2DynamicExt *TorchCumMaxMinPluginDynamic::clone() const
{
    TorchCumMaxMinPluginDynamic *plugin = new TorchCumMaxMinPluginDynamic(mLayerName,
                                                                            mDim,
                                                                            mCumType);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs TorchCumMaxMinPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    nvinfer1::DimsExprs ret = inputs[0];
    return ret;
}

bool TorchCumMaxMinPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
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
    case 2:
        return out[1].type == nvinfer1::DataType::kINT32 && 
                out[1].format == nvinfer1::TensorFormat::kLINEAR;
    }
}

void TorchCumMaxMinPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
}

size_t TorchCumMaxMinPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int TorchCumMaxMinPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{

    nvinfer1::Dims input_dims = inputDesc[0].dims;
    nvinfer1::Dims output_dims = outputDesc[0].dims;

    auto data_type = inputDesc[0].type;

    switch(data_type){
    case nvinfer1::DataType::kFLOAT:
        torch_cum_maxmin<float>((float*)outputs[0],(int*)outputs[1], (float*)inputs[0],
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

nvinfer1::DataType TorchCumMaxMinPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    switch (index)
    {
    case 0:
        return inputTypes[0];
        break;
    
    case 1:
        return nvinfer1::DataType::kINT32;
        break;
    default:
        break;
    }
}

// IPluginV2 Methods
const char *TorchCumMaxMinPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *TorchCumMaxMinPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int TorchCumMaxMinPluginDynamic::getNbOutputs() const
{
    return 2;
}

int TorchCumMaxMinPluginDynamic::initialize()
{
    return 0;
}

void TorchCumMaxMinPluginDynamic::terminate()
{
}

size_t TorchCumMaxMinPluginDynamic::getSerializationSize() const
{
    return sizeof(mDim)
    +sizeof(mCumType);
}

void TorchCumMaxMinPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mDim);
    serialize_value(&buffer, mCumType);
}

void TorchCumMaxMinPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void TorchCumMaxMinPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *TorchCumMaxMinPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

TorchCumMaxMinPluginDynamicCreator::TorchCumMaxMinPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *TorchCumMaxMinPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *TorchCumMaxMinPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *TorchCumMaxMinPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *TorchCumMaxMinPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
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

    TorchCumMaxMinPluginDynamic *plugin = new TorchCumMaxMinPluginDynamic(name,
                                                                dim,
                                                                cumType);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *TorchCumMaxMinPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new TorchCumMaxMinPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void TorchCumMaxMinPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *TorchCumMaxMinPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
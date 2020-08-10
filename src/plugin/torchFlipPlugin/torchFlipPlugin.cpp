
#include <assert.h>
#include <chrono>
#include "plugin/torchFlipPlugin/torchFlipPlugin.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "amir_cuda_util/cuda_util.h"

#include "torch_flip.h"


namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"TorchFlipPluginDynamic"};
} // namespace

PluginFieldCollection TorchFlipPluginDynamicCreator::mFC{};
std::vector<PluginField> TorchFlipPluginDynamicCreator::mPluginAttributes({PluginField("mode"),
                                                                            PluginField("dims")});

TorchFlipPluginDynamic::TorchFlipPluginDynamic(
    const std::string &name, 
    const nvinfer1::Dims &dims)
    : mLayerName(name),
      mDims(dims)
{}

TorchFlipPluginDynamic::TorchFlipPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mDims);
}

nvinfer1::IPluginV2DynamicExt *TorchFlipPluginDynamic::clone() const
{
    TorchFlipPluginDynamic *plugin = new TorchFlipPluginDynamic(mLayerName,
                                                                    mDims);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs TorchFlipPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    nvinfer1::DimsExprs ret = inputs[0];
    return ret;
}

bool TorchFlipPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    assert(0 <= pos && pos < 2);
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

void TorchFlipPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
}

size_t TorchFlipPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int TorchFlipPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{

    nvinfer1::Dims input_dims = inputDesc[0].dims;
    nvinfer1::Dims output_dims = outputDesc[0].dims;

    auto data_type = inputDesc[0].type;

    switch(data_type){
    case nvinfer1::DataType::kFLOAT:
        torch_flip<float>((float*)outputs[0], (float*)inputs[0],
                  &(input_dims.d[0]), input_dims.nbDims,
                  &(mDims.d[0]), mDims.nbDims,
                  stream);
        break;

    // case nvinfer1::DataType::kHALF:
    //     grid_sample<half>((half*)outputs[0], (half*)inputs[0], (half*)inputs[1],
    //                     &(output_dims.d[0]), &(input_dims.d[0]), &(grid_dims.d[0]), input_dims.nbDims,
    //                     intep_mode, padding_mode,
    //                     mAlignCorners,
    //                     stream);
    //     break;
    default:
        return 1;
        break;
    }

    return 0;
}

nvinfer1::DataType TorchFlipPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

// IPluginV2 Methods
const char *TorchFlipPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *TorchFlipPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int TorchFlipPluginDynamic::getNbOutputs() const
{
    return 1;
}

int TorchFlipPluginDynamic::initialize()
{
    return 0;
}

void TorchFlipPluginDynamic::terminate()
{
}

size_t TorchFlipPluginDynamic::getSerializationSize() const
{
    return sizeof(mDims);
}

void TorchFlipPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mDims);
}

void TorchFlipPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void TorchFlipPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *TorchFlipPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

TorchFlipPluginDynamicCreator::TorchFlipPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *TorchFlipPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *TorchFlipPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *TorchFlipPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *TorchFlipPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    nvinfer1::Dims dims;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("dims") == 0)
        {
            dims.nbDims = fc->fields[i].length;
            for(int j=0;j<dims.nbDims;++j){
                dims.d[j] = static_cast<const int *>(fc->fields[i].data)[j];
            }
        }

    }

    TorchFlipPluginDynamic *plugin = new TorchFlipPluginDynamic(name,
                                                                dims);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *TorchFlipPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new TorchFlipPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void TorchFlipPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *TorchFlipPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
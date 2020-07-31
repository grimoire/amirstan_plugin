
#include <assert.h>
#include <chrono>
#include "plugin/gridSamplePlugin/gridSamplePlugin.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "amir_cuda_util/cuda_util.h"

#include "grid_sample.h"


namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"GridSamplePluginDynamic"};
} // namespace

PluginFieldCollection GridSamplePluginDynamicCreator::mFC{};
std::vector<PluginField> GridSamplePluginDynamicCreator::mPluginAttributes({PluginField("mode"),
                                                                            PluginField("padding_mode"),
                                                                            PluginField("align_corners")});

GridSamplePluginDynamic::GridSamplePluginDynamic(
    const std::string &name,
    int mode, 
    int paddingMode, 
    bool alignCorners)
    : mLayerName(name),
      mMode(mode),
      mPaddingMode(paddingMode),
      mAlignCorners(alignCorners)
{}

GridSamplePluginDynamic::GridSamplePluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mMode);
    deserialize_value(&data, &length, &mPaddingMode);
    deserialize_value(&data, &length, &mAlignCorners);
}

nvinfer1::IPluginV2DynamicExt *GridSamplePluginDynamic::clone() const
{
    GridSamplePluginDynamic *plugin = new GridSamplePluginDynamic(mLayerName,
                                                                    mMode,
                                                                    mPaddingMode,
                                                                    mAlignCorners);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs GridSamplePluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    nvinfer1::DimsExprs ret;
    ret.nbDims = inputs[0].nbDims;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[0].d[1];
    for(int i=2;i<ret.nbDims;++i){
        ret.d[i] = inputs[1].d[i-1];
    }
    return ret;
}

bool GridSamplePluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    assert(0 <= pos && pos < 3);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;

    switch (pos)
    {
    case 0:
        return (in[0].type == nvinfer1::DataType::kFLOAT && in[0].format == nvinfer1::TensorFormat::kLINEAR);
    case 1:
        return in[1].type == in[0].type && 
                in[1].format == in[0].format;
    case 2:
        return out[0].type == in[0].type &&
               out[0].format == in[0].format;
    }
}

void GridSamplePluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
}

size_t GridSamplePluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int GridSamplePluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{

    nvinfer1::Dims input_dims = inputDesc[0].dims;
    nvinfer1::Dims grid_dims = inputDesc[1].dims;
    nvinfer1::Dims output_dims = outputDesc[0].dims;

    amirstan::plugin::GridSamplerInterpolation intep_mode = amirstan::plugin::GridSamplerInterpolation::Nearest;
    switch (nvinfer1::ResizeMode(mMode))
    {
    case nvinfer1::ResizeMode::kLINEAR:
        intep_mode = amirstan::plugin::GridSamplerInterpolation::Bilinear;
        break;
    
    case nvinfer1::ResizeMode::kNEAREST:
        intep_mode = amirstan::plugin::GridSamplerInterpolation::Nearest;
        break;
    default:
        break;
    }

    amirstan::plugin::GridSamplerPadding padding_mode = amirstan::plugin::GridSamplerPadding::Zeros;
    switch (mPaddingMode)
    {
    case 0:
        padding_mode = amirstan::plugin::GridSamplerPadding::Zeros;
        break;
    
    case 1:
        padding_mode = amirstan::plugin::GridSamplerPadding::Border;
        break;

    case 2:
        padding_mode = amirstan::plugin::GridSamplerPadding::Reflection;
        break;
    default:
        break;
    }


    auto data_type = inputDesc[0].type;

    switch(data_type){
    case nvinfer1::DataType::kFLOAT:
        grid_sample<float>((float*)outputs[0], (float*)inputs[0], (float*)inputs[1],
                        &(output_dims.d[0]), &(input_dims.d[0]), &(grid_dims.d[0]), input_dims.nbDims,
                        intep_mode, padding_mode,
                        mAlignCorners,
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

nvinfer1::DataType GridSamplePluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

// IPluginV2 Methods
const char *GridSamplePluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *GridSamplePluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int GridSamplePluginDynamic::getNbOutputs() const
{
    return 1;
}

int GridSamplePluginDynamic::initialize()
{
    return 0;
}

void GridSamplePluginDynamic::terminate()
{
}

size_t GridSamplePluginDynamic::getSerializationSize() const
{
    return sizeof(mMode)+
            sizeof(mPaddingMode)+
            sizeof(mAlignCorners);
}

void GridSamplePluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mMode);
    serialize_value(&buffer, mPaddingMode);
    serialize_value(&buffer, mAlignCorners);
}

void GridSamplePluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void GridSamplePluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *GridSamplePluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

GridSamplePluginDynamicCreator::GridSamplePluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *GridSamplePluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *GridSamplePluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *GridSamplePluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *GridSamplePluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    int mode=0;
    int paddingMode=0;
    bool alignCorners=false;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("mode") == 0)
        {
            mode = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("padding_mode") == 0)
        {
            paddingMode = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("align_corners") == 0)
        {
            alignCorners = (bool)(static_cast<const int *>(fc->fields[i].data)[0]);
        }
    }

    GridSamplePluginDynamic *plugin = new GridSamplePluginDynamic(name,
                                                                mode,
                                                                paddingMode,
                                                                alignCorners);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *GridSamplePluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new GridSamplePluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void GridSamplePluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *GridSamplePluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
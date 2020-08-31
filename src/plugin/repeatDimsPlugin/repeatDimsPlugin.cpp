
#include <assert.h>
#include <chrono>
#include "plugin/repeatDimsPlugin/repeatDimsPlugin.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "amir_cuda_util/cuda_util.h"


namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"RepeatDimsPluginDynamic"};
} // namespace

PluginFieldCollection RepeatDimsPluginDynamicCreator::mFC{};
std::vector<PluginField> RepeatDimsPluginDynamicCreator::mPluginAttributes({PluginField("repeat_dims")});

RepeatDimsPluginDynamic::RepeatDimsPluginDynamic(
    const std::string &name,
    const nvinfer1::Dims &repeatDims)
    : mLayerName(name),
      mRepeatDims(repeatDims)
{}

RepeatDimsPluginDynamic::RepeatDimsPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mRepeatDims);

    initialize();
}

nvinfer1::IPluginV2DynamicExt *RepeatDimsPluginDynamic::clone() const
{
    RepeatDimsPluginDynamic *plugin = new RepeatDimsPluginDynamic(mLayerName,
                                                    mRepeatDims);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs RepeatDimsPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    // assert(nbInputs == 1);
    DimsExprs outputDims;
    
    if(nbInputs==1){
        outputDims.nbDims = mRepeatDims.nbDims;
        int dim_offset = mRepeatDims.nbDims - inputs[0].nbDims;
        
        for(int i=0; i<dim_offset; ++i){
            outputDims.d[i] = exprBuilder.constant(mRepeatDims.d[i]);
        }

        for(int i=0; i<inputs[0].nbDims; ++i){
            outputDims.d[i+dim_offset] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, 
            *(inputs[0].d[i]),
            *(exprBuilder.constant(mRepeatDims.d[i+dim_offset])));
        }
    }else{
        outputDims = inputs[1];
    }

    return outputDims;
}

bool RepeatDimsPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    // assert(0 <= pos && pos < 2);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    switch (pos)
    {
    case 0:
        return (in[0].type == nvinfer1::DataType::kFLOAT && in[0].format == nvinfer1::TensorFormat::kLINEAR)
        || (in[0].type == nvinfer1::DataType::kHALF && in[0].format == nvinfer1::TensorFormat::kLINEAR)
        ||(in[0].type == nvinfer1::DataType::kINT32 && in[0].format == nvinfer1::TensorFormat::kLINEAR);
    case 1:
        if(nbInputs==1){
            return out[0].type == in[0].type &&
                out[0].format == in[0].format;
        }else{
            return (in[1].type == nvinfer1::DataType::kFLOAT && in[1].format == nvinfer1::TensorFormat::kLINEAR)
        || (in[1].type == nvinfer1::DataType::kHALF && in[1].format == nvinfer1::TensorFormat::kLINEAR)
        ||(in[1].type == nvinfer1::DataType::kINT32 && in[1].format == nvinfer1::TensorFormat::kLINEAR);
        }
    case 2:
        return out[0].type == in[0].type &&
            out[0].format == in[0].format;
    }
}

void RepeatDimsPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    // assert(nbInputs == 1);
}

size_t RepeatDimsPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int RepeatDimsPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{
    nvinfer1::Dims input_dims;
    nvinfer1::Dims repeat_dims;

    if(mRepeatDims.nbDims!=0){
        repeat_dims = mRepeatDims;
    }else{
        repeat_dims = inputDesc[1].dims;
        int nb_offset = repeat_dims.nbDims - inputDesc[0].dims.nbDims;
        for(int i=nb_offset;i<repeat_dims.nbDims;++i){
            if(inputDesc[0].dims.d[i-nb_offset]!=1){
                repeat_dims.d[i]=1;
            }
        }
    }
    
    input_dims.nbDims = repeat_dims.nbDims;
    int nb_offset = input_dims.nbDims - (inputDesc[0].dims).nbDims;
    for(int i=0;i< nb_offset;++i){
        input_dims.d[i] = 1;
    }
    for(int i=0;i<(inputDesc[0].dims).nbDims; ++i){
        input_dims.d[i+nb_offset] = (inputDesc[0].dims).d[i];
    }

    auto data_type = inputDesc[0].type;

    switch(data_type){
    case nvinfer1::DataType::kFLOAT:
        amirstan::cuda::repeat_dims<float>((float*)outputs[0], (float*)inputs[0], 
        &(input_dims.d[0]), &repeat_dims.d[0], input_dims.nbDims,
        stream);
        break;

    case nvinfer1::DataType::kHALF:
        amirstan::cuda::repeat_dims<half>((half*)outputs[0], (half*)inputs[0], 
        &(input_dims.d[0]), &repeat_dims.d[0], input_dims.nbDims,
        stream);
        break;

    case nvinfer1::DataType::kINT32:
        amirstan::cuda::repeat_dims<int>((int*)outputs[0], (int*)inputs[0], 
        &(input_dims.d[0]), &repeat_dims.d[0], input_dims.nbDims,
        stream);
        break;
    default:
        return 1;
        break;
    }
    return 0;
}

nvinfer1::DataType RepeatDimsPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    // assert(nbInputs == 1);
    return inputTypes[0];
}

// IPluginV2 Methods
const char *RepeatDimsPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *RepeatDimsPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int RepeatDimsPluginDynamic::getNbOutputs() const
{
    return 1;
}

int RepeatDimsPluginDynamic::initialize()
{
    return 0;
}

void RepeatDimsPluginDynamic::terminate()
{
}

size_t RepeatDimsPluginDynamic::getSerializationSize() const
{
    return sizeof(mRepeatDims);
}

void RepeatDimsPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mRepeatDims);
}

void RepeatDimsPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void RepeatDimsPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *RepeatDimsPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

RepeatDimsPluginDynamicCreator::RepeatDimsPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *RepeatDimsPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *RepeatDimsPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *RepeatDimsPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *RepeatDimsPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    nvinfer1::Dims repeat_dims;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("repeat_dims") == 0)
        {
            repeat_dims.nbDims = fc->fields[i].length;
            for(int j=0;j<repeat_dims.nbDims;++j){
                repeat_dims.d[j] = static_cast<const int *>(fc->fields[i].data)[j];
            }
        }
    }

    RepeatDimsPluginDynamic *plugin = new RepeatDimsPluginDynamic(name,
                                                    repeat_dims);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *RepeatDimsPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new RepeatDimsPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void RepeatDimsPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *RepeatDimsPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
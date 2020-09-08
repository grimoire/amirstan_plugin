
#include <assert.h>
#include <chrono>
#include "plugin/groupNormPlugin/groupNormPlugin.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "group_norm.h"

namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"GroupNormPluginDynamic"};
} // namespace

PluginFieldCollection GroupNormPluginDynamicCreator::mFC{};
std::vector<PluginField> GroupNormPluginDynamicCreator::mPluginAttributes({PluginField("num_groups"),
                                                                           PluginField("num_channels"),
                                                                           PluginField("W"),
                                                                           PluginField("B"),
                                                                           PluginField("eps")});

GroupNormPluginDynamic::GroupNormPluginDynamic(
    const std::string &name,
    int num_groups, int num_channels, float eps,
    const nvinfer1::Weights &W, const nvinfer1::Weights &B)
    : mLayerName(name),
      mNumGroups(num_groups),
      mNumChannels(num_channels),
      mEps(eps),
      mW(W),
      mNumParamsW(W.count),
      mB(B),
      mNumParamsB(B.count)
{
    size_t wordSize = samplesCommon::getElementSize(nvinfer1::DataType::kFLOAT);

    mWhost = std::shared_ptr<char>(new char[mNumParamsW * wordSize]);
    memcpy((void *)mWhost.get(), mW.values, mNumParamsW * wordSize);
    mW.values = mWhost.get();

    mBhost = std::shared_ptr<char>(new char[mNumParamsB * wordSize]);
    memcpy((void *)mBhost.get(), mB.values, mNumParamsB * wordSize);
    mB.values = mBhost.get();
    mWdev=nullptr;
    mBdev=nullptr;
    initialize();
}

GroupNormPluginDynamic::GroupNormPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mNumGroups);
    deserialize_value(&data, &length, &mNumChannels);
    deserialize_value(&data, &length, &mEps);
    deserialize_value(&data, &length, &mNumParamsW);
    deserialize_value(&data, &length, &mNumParamsB);

    auto data_type = nvinfer1::DataType::kFLOAT;
    size_t wordSize = samplesCommon::getElementSize(data_type);

    const char *d = static_cast<const char *>(data);
    char *w_data = deserToHost<char>(d, mNumParamsW * wordSize);
    mWhost = std::shared_ptr<char>((char *)w_data);
    mW.values = mWhost.get();

    char *b_data = deserToHost<char>(d, mNumParamsB * wordSize);
    mBhost = std::shared_ptr<char>((char *)b_data);
    mB.values = mBhost.get();

    mW.type = data_type;
    mB.type = data_type;    

    mW.count = mNumParamsW;
    mB.count = mNumParamsB;

    mWdev=nullptr;
    mBdev=nullptr;
    initialize();
}

nvinfer1::IPluginV2DynamicExt *GroupNormPluginDynamic::clone() const
{
    GroupNormPluginDynamic *plugin = new GroupNormPluginDynamic(mLayerName,
                                                                mNumGroups,
                                                                mNumChannels,
                                                                mEps,
                                                                mW,
                                                                mB);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs GroupNormPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    assert(nbInputs == 1);

    return inputs[0];
}

bool GroupNormPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
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
               out[0].format == nvinfer1::TensorFormat::kLINEAR;
    default:
        return false;
    }

    return false;
}

void GroupNormPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(nbInputs == 1);
    // const auto &inDims0 = inputs[0].desc.dims;
}

size_t GroupNormPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    size_t wordSize = samplesCommon::getElementSize(inputs[0].type);
    int batch_size = inputs[0].dims.d[0];
    int channel_size = inputs[0].dims.d[1];
    int width = inputs[0].dims.d[2];
    int height = inputs[0].dims.d[3];
    size_t mean_size = batch_size * mNumGroups * wordSize;
    size_t var_size = mean_size;
    size_t mean_var_size = batch_size*channel_size*width*height*wordSize;

    size_t final_size = mean_size + var_size + 100;
    return final_size;
}

int GroupNormPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{
    int batch_size = inputDesc[0].dims.d[0];
    int inputChannel = inputDesc[0].dims.d[1];
    int inputHeight = inputDesc[0].dims.d[2];
    int inputWidth = inputDesc[0].dims.d[3];

    auto data_type = inputDesc[0].type;

    switch(data_type){
    case nvinfer1::DataType::kFLOAT:
        compute_group_norm<float>((float*)outputs[0], (float*)inputs[0], 
        batch_size, mNumGroups, inputChannel, inputWidth*inputHeight,
        mEps,
        (float*)mWdev, (float*)mBdev,  stream, workSpace);
        break;

    // case nvinfer1::DataType::kHALF:
    //     compute_group_norm<half>((half*)outputs[0], (half*)inputs[0], 
    //     batch_size, mNumGroups, inputChannel, inputWidth*inputHeight,
    //     (half)mEps,
    //     (float*)mWdev, (float*)mBdev,  stream, workSpace);
    //     break;
    default:
        return 1;
        break;
    }

    return 0;
}

nvinfer1::DataType GroupNormPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    assert(nbInputs == 1);
    return inputTypes[0];
}

// IPluginV2 Methods
const char *GroupNormPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *GroupNormPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int GroupNormPluginDynamic::getNbOutputs() const
{
    return 1;
}

int GroupNormPluginDynamic::initialize()
{
    if (mW.values)
    {
        // target size
        size_t wordSize = samplesCommon::getElementSize(nvinfer1::DataType::kFLOAT);


        if (true)
        {
            if(mWdev == nullptr){
                size_t nbWBytes = mW.count * wordSize;
                mW.type = nvinfer1::DataType::kFLOAT;
                CHECK(cudaMalloc((void **)&mWdev, nbWBytes));   
                convertAndCopyToDevice(mW, reinterpret_cast<float *>(mWdev));
            }

            if(mBdev == nullptr){
                size_t nbBBytes = mB.count * wordSize;
                mB.type = nvinfer1::DataType::kFLOAT; 
                CHECK(cudaMalloc((void **)&mBdev, nbBBytes));
                convertAndCopyToDevice(mB, reinterpret_cast<float *>(mBdev));
            }
        }        
        
    }

    return 0;
}

void GroupNormPluginDynamic::terminate()
{
    gLogVerbose << "FC Plugin terminate start" << std::endl;

    if(mWdev!=nullptr){
        cudaFree(mWdev);
        mWdev = nullptr;
    }
    if(mBdev!=nullptr){
        cudaFree(mBdev);
        mBdev = nullptr;
    }

    gLogVerbose << "FC Plugin terminate done" << std::endl;
}

size_t GroupNormPluginDynamic::getSerializationSize() const
{
    size_t wordSize = samplesCommon::getElementSize(nvinfer1::DataType::kFLOAT);
    return wordSize * mNumParamsW + wordSize * mNumParamsB +
           sizeof(mNumChannels) +
           sizeof(mNumGroups) +
           sizeof(mNumParamsW) +
           sizeof(mNumParamsB) +
           sizeof(mEps);
}

void GroupNormPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mNumGroups);
    serialize_value(&buffer, mNumChannels);
    serialize_value(&buffer, mEps);

    serialize_value(&buffer, mNumParamsW);
    serialize_value(&buffer, mNumParamsB);

    size_t wordSize = samplesCommon::getElementSize(nvinfer1::DataType::kFLOAT);

    char *d = static_cast<char *>(buffer);
    serFromHost(d, mW.values, mNumParamsW * wordSize);
    serFromHost(d, mB.values, mNumParamsB * wordSize);
}

void GroupNormPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void GroupNormPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *GroupNormPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

GroupNormPluginDynamicCreator::GroupNormPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *GroupNormPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *GroupNormPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *GroupNormPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *GroupNormPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{
    int num_channels = 0;
    int num_groups = 0;
    float eps = 1e-5;

    nvinfer1::Weights W;
    W.count = 0;
    W.values = nullptr;
    nvinfer1::Weights B;
    B.count = 0;
    B.values = nullptr;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("num_groups") == 0)
        {
            num_groups = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("num_channels") == 0)
        {
            num_channels = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("eps") == 0)
        {
            eps = static_cast<const float *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("W") == 0)
        {
            // gLogVerbose << "Building W...\n";
            W.values = fc->fields[i].data;
            W.count = fc->fields[i].length;
            W.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("B") == 0)
        {
            // gLogVerbose << "Building B...\n";
            B.values = fc->fields[i].data;
            B.count = fc->fields[i].length;
            B.type = fieldTypeToDataType(fc->fields[i].type);
        }
    }

    if (num_groups < 1)
    {
        gLogError << "Invalid num_group" << std::endl;
    }
    if (W.count == 0 || W.values == nullptr || W.count < num_channels)
    {
        gLogError << "Invalid weights" << std::endl;
    }
    if (B.count == 0 || B.values == nullptr || B.count < num_channels)
    {
        gLogError << "Invalid bias" << std::endl;
    }

    GroupNormPluginDynamic *plugin = new GroupNormPluginDynamic(name,
                                                                num_groups,
                                                                num_channels,
                                                                eps,
                                                                W,
                                                                B);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *GroupNormPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new GroupNormPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void GroupNormPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *GroupNormPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
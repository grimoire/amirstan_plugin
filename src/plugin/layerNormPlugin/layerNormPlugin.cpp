
#include <assert.h>
#include <chrono>
#include "plugin/layerNormPlugin/layerNormPlugin.h"
#include "layer_norm.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"

namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"LayerNormPluginDynamic"};
} // namespace

PluginFieldCollection LayerNormPluginDynamicCreator::mFC{};
std::vector<PluginField> LayerNormPluginDynamicCreator::mPluginAttributes({PluginField("normalized_shape"),
                                                                           PluginField("W"),
                                                                           PluginField("B"),
                                                                           PluginField("eps")});

LayerNormPluginDynamic::LayerNormPluginDynamic(
    const std::string &name,
    const nvinfer1::Dims &normalized_shape, float eps,
    const nvinfer1::Weights &W, const nvinfer1::Weights &B)
    : mLayerName(name),
      mNormalizedShape(normalized_shape),
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
}

LayerNormPluginDynamic::LayerNormPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mNormalizedShape);
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
    initialize();
}

nvinfer1::IPluginV2DynamicExt *LayerNormPluginDynamic::clone() const
{
    LayerNormPluginDynamic *plugin = new LayerNormPluginDynamic(mLayerName,
                                                                mNormalizedShape,
                                                                mEps,
                                                                mW,
                                                                mB);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs LayerNormPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    assert(nbInputs == 1);

    return inputs[0];
}

bool LayerNormPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    assert(0 <= pos && pos < 2);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    switch (pos)
    {
    case 0:
        return in[0].type == nvinfer1::DataType::kFLOAT && in[0].format == nvinfer1::TensorFormat::kLINEAR;
    case 1:
        return out[0].type == in[0].type &&
               out[0].format == in[0].format;
    }
}

void LayerNormPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(nbInputs == 1);
    // const auto &inDims0 = inputs[0].desc.dims;
}

size_t LayerNormPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    size_t wordSize = samplesCommon::getElementSize(nvinfer1::DataType::kFLOAT);
    int mean_dims = inputs[0].dims.nbDims - mNormalizedShape.nbDims;
    // size_t mean_size = batch_size * mNumGroups * wordSize;
    size_t mean_size = wordSize;
    for(int i=0;i<mean_dims;++i){
        mean_size *= inputs[0].dims.d[i];
    }
    size_t var_size = mean_size;

    size_t final_size = mean_size + var_size + 100;
    return final_size;
}

int LayerNormPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{

    int batch_size = inputDesc[0].dims.d[0];
    int inputChannel = inputDesc[0].dims.d[1];
    int inputHeight = inputDesc[0].dims.d[2];
    int inputWidth = inputDesc[0].dims.d[3];

    size_t input_size = 1;
    for(int i=0;i<inputDesc[0].dims.nbDims;++i){
        input_size *= inputDesc[0].dims.d[i];
    }

    size_t layer_size = 1;
    for(int i=0;i<mNormalizedShape.nbDims;++i){
        layer_size *= mNormalizedShape.d[i];
    }

    int norm_size = input_size/layer_size;

    compute_layer_norm<float>((float*)outputs[0], (float*)inputs[0], 
    norm_size, layer_size,
    mEps,
     (float*)mWdev, (float*)mBdev,  stream, workSpace);

    return 0;
}

nvinfer1::DataType LayerNormPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    assert(nbInputs == 1);
    return inputTypes[0];
}

// IPluginV2 Methods
const char *LayerNormPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *LayerNormPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int LayerNormPluginDynamic::getNbOutputs() const
{
    return 1;
}

int LayerNormPluginDynamic::initialize()
{
    if (mW.values)
    {
        // target size
        size_t wordSize = samplesCommon::getElementSize(nvinfer1::DataType::kFLOAT);


        if (true)
        {
            size_t nbWBytes = mW.count * wordSize;
            mW.type = nvinfer1::DataType::kFLOAT;
            CHECK(cudaMalloc((void **)&mWdev, nbWBytes));   
            convertAndCopyToDevice(mW, reinterpret_cast<float *>(mWdev));

            size_t nbBBytes = mB.count * wordSize;
            mB.type = nvinfer1::DataType::kFLOAT;
            CHECK(cudaMalloc((void **)&mBdev, nbBBytes));
            convertAndCopyToDevice(mB, reinterpret_cast<float *>(mBdev));
        }        
        
    }

    return 0;
}

void LayerNormPluginDynamic::terminate()
{
    gLogVerbose << "FC Plugin terminate start" << std::endl;

    cudaFree(mWdev);
    cudaFree(mBdev);

    gLogVerbose << "FC Plugin terminate done" << std::endl;
}

size_t LayerNormPluginDynamic::getSerializationSize() const
{
    size_t wordSize = samplesCommon::getElementSize(nvinfer1::DataType::kFLOAT);
    return wordSize * mNumParamsW + wordSize * mNumParamsB +
           sizeof(mNormalizedShape) +
           sizeof(mNumParamsW) +
           sizeof(mNumParamsB) +
           sizeof(mEps);
}

void LayerNormPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mNormalizedShape);
    serialize_value(&buffer, mEps);

    serialize_value(&buffer, mNumParamsW);
    serialize_value(&buffer, mNumParamsB);

    size_t wordSize = samplesCommon::getElementSize(nvinfer1::DataType::kFLOAT);

    char *d = static_cast<char *>(buffer);
    serFromHost(d, mW.values, mNumParamsW * wordSize);
    serFromHost(d, mB.values, mNumParamsB * wordSize);
}

void LayerNormPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void LayerNormPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *LayerNormPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

LayerNormPluginDynamicCreator::LayerNormPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *LayerNormPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *LayerNormPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *LayerNormPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *LayerNormPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    int typeId = -1;
    nvinfer1::Dims normalizedShape;
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

        if (field_name.compare("normalized_shape") == 0)
        {
            int nbDims = fc->fields[i].length;
            normalizedShape.nbDims = nbDims;
            for(int j=0;j<nbDims;++j){
                normalizedShape.d[j] = static_cast<const int *>(fc->fields[i].data)[j];
            }
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

    if (W.count == 0 || W.values == nullptr)
    {
        gLogError << "Invalid weights" << std::endl;
    }
    if (B.count == 0 || B.values == nullptr)
    {
        gLogError << "Invalid bias" << std::endl;
    }

    LayerNormPluginDynamic *plugin = new LayerNormPluginDynamic(name,
                                                                normalizedShape,
                                                                eps,
                                                                W,
                                                                B);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *LayerNormPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new LayerNormPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void LayerNormPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *LayerNormPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan

#include <assert.h>
#include <chrono>
#include "plugin/carafeFeatureReassemblePlugin/carafeFeatureReassemblePlugin.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "carafe_cuda.h"

namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"CarafeFeatureReassemblePluginDynamic"};
} // namespace

PluginFieldCollection CarafeFeatureReassemblePluginDynamicCreator::mFC{};
std::vector<PluginField> CarafeFeatureReassemblePluginDynamicCreator::mPluginAttributes({PluginField("scale_factor"),
                                                                                         PluginField("up_kernel"),
                                                                                         PluginField("up_group")});

CarafeFeatureReassemblePluginDynamic::CarafeFeatureReassemblePluginDynamic(
    const std::string &name,
    int scaleFactor, int upKernel, int upGroup)
    : mLayerName(name),
      mScaleFactor(scaleFactor),
      mUpKernel(upKernel),
      mUpGroup(upGroup)
{
}

CarafeFeatureReassemblePluginDynamic::CarafeFeatureReassemblePluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mScaleFactor);
    deserialize_value(&data, &length, &mUpKernel);
    deserialize_value(&data, &length, &mUpGroup);

    // mW.values = nullptr;
    // initialize();
}

nvinfer1::IPluginV2DynamicExt *CarafeFeatureReassemblePluginDynamic::clone() const
{
    CarafeFeatureReassemblePluginDynamic *plugin = new CarafeFeatureReassemblePluginDynamic(mLayerName,
                                                                                            mScaleFactor,
                                                                                            mUpKernel,
                                                                                            mUpGroup);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

nvinfer1::DimsExprs CarafeFeatureReassemblePluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    assert(nbInputs == 2);
    assert(inputs[0].nbDims == 4);

    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = inputs[0].d[0];

    ret.d[1] = inputs[0].d[1];

    auto kScaleFactor = exprBuilder.constant(mScaleFactor);
    ret.d[2] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
                                     *inputs[0].d[2],
                                     *kScaleFactor);

    ret.d[3] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
                                     *inputs[0].d[3],
                                     *kScaleFactor);

    return ret;
}

bool CarafeFeatureReassemblePluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    assert(0 <= pos && pos < 3);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    switch (pos)
    {
    case 0:
        return in[0].type == nvinfer1::DataType::kFLOAT &&
         in[0].format == nvinfer1::TensorFormat::kLINEAR;
    case 1:
        return in[1].type == nvinfer1::DataType::kFLOAT &&
               in[1].format == nvinfer1::TensorFormat::kLINEAR;
    case 2:
        return out[0].type == nvinfer1::DataType::kFLOAT &&
               out[0].format == nvinfer1::TensorFormat::kLINEAR;
    }
}

void CarafeFeatureReassemblePluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(nbInputs == 2);
    // const auto &inDims0 = inputs[0].desc.dims;
}

size_t CarafeFeatureReassemblePluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{

    int batch_size = inputs[0].dims.d[0];
    int inputHeight = inputs[0].dims.d[2];
    int inputWidth = inputs[0].dims.d[3];

    int numChannel = outputs[0].dims.d[1];
    int outputHeight = outputs[0].dims.d[2];
    int outputWidth = outputs[0].dims.d[3];

    int maskChannel = inputs[1].dims.d[1];

    int sizeof_dtype = samplesCommon::getElementSize(nvinfer1::DataType::kFLOAT);
    size_t rfeature_size = batch_size*numChannel*inputHeight*inputWidth*sizeof_dtype;
    size_t rmask_size = batch_size*maskChannel*outputHeight*outputWidth*sizeof_dtype;
    size_t routput_size = batch_size*numChannel*outputHeight*outputWidth*sizeof_dtype;
    return rfeature_size + rmask_size + routput_size;
}

int CarafeFeatureReassemblePluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                                  const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{
    int batch_size = inputDesc[0].dims.d[0];
    int inputHeight = inputDesc[0].dims.d[2];
    int inputWidth = inputDesc[0].dims.d[3];

    int numChannel = outputDesc[0].dims.d[1];
    int outputHeight = outputDesc[0].dims.d[2];
    int outputWidth = outputDesc[0].dims.d[3];

    int maskChannel = inputDesc[1].dims.d[1];

    int sizeof_dtype = samplesCommon::getElementSize(nvinfer1::DataType::kFLOAT);
    size_t rfeature_size = batch_size*numChannel*inputHeight*inputWidth*sizeof_dtype;
    size_t rmask_size = batch_size*maskChannel*outputHeight*outputWidth*sizeof_dtype;
    size_t routput_size = batch_size*numChannel*outputHeight*outputWidth*sizeof_dtype;

    void* rfeatures = workSpace;
    void* rmasks = workSpace+rfeature_size;
    void* routput = rmasks+routput_size;
    const void* features = inputs[0];
    const void* masks = inputs[1];
    void* output = outputs[0];

    CARAFEForwardLaucher((const float*)features, (const float*)masks,
                        mUpKernel, mUpGroup,
                        mScaleFactor, batch_size,
                        numChannel, inputHeight,
                        inputWidth, outputHeight,
                        outputWidth, maskChannel,
                        (float*)rfeatures, (float*)routput,
                        (float*)rmasks, (float*)output,
                        stream);

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType CarafeFeatureReassemblePluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    assert(nbInputs == 2);
    return inputTypes[0];
}

// IPluginV2 Methods
const char *CarafeFeatureReassemblePluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *CarafeFeatureReassemblePluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int CarafeFeatureReassemblePluginDynamic::getNbOutputs() const
{
    return 1;
}

int CarafeFeatureReassemblePluginDynamic::initialize()
{
    return 0;
}

void CarafeFeatureReassemblePluginDynamic::terminate()
{
}

size_t CarafeFeatureReassemblePluginDynamic::getSerializationSize() const
{
    return sizeof(mScaleFactor) +
           sizeof(mUpKernel) +
           sizeof(mUpGroup);
}

void CarafeFeatureReassemblePluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mScaleFactor);
    serialize_value(&buffer, mUpKernel);
    serialize_value(&buffer, mUpGroup);
}

void CarafeFeatureReassemblePluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void CarafeFeatureReassemblePluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *CarafeFeatureReassemblePluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

CarafeFeatureReassemblePluginDynamicCreator::CarafeFeatureReassemblePluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *CarafeFeatureReassemblePluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *CarafeFeatureReassemblePluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *CarafeFeatureReassemblePluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *CarafeFeatureReassemblePluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{
    int outDims = 0;

    int scaleFactor = 1;
    int upKernel = 1;
    int upGroup = 1;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);
        

        if (field_name.compare("scale_factor") == 0)
        {
            scaleFactor = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("up_kernel") == 0)
        {
            upKernel = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("up_group") == 0)
        {
            upGroup = static_cast<const int *>(fc->fields[i].data)[0];
        }
    }

    CarafeFeatureReassemblePluginDynamic *plugin = new CarafeFeatureReassemblePluginDynamic(name,
                                                                                            scaleFactor,
                                                                                            upKernel,
                                                                                            upGroup);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

IPluginV2 *CarafeFeatureReassemblePluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new CarafeFeatureReassemblePluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void CarafeFeatureReassemblePluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *CarafeFeatureReassemblePluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
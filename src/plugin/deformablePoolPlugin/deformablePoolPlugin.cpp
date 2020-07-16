
#include <assert.h>
#include <chrono>
#include "plugin/deformablePoolPlugin/deformablePoolPlugin.h"

#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "deform_roi_pool.h"

namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"DeformablePoolPluginDynamic"};
} // namespace

PluginFieldCollection DeformablePoolPluginDynamicCreator::mFC{};
std::vector<PluginField> DeformablePoolPluginDynamicCreator::mPluginAttributes({PluginField("out_size"),
                                                                           PluginField("spatial_scale"),
                                                                           PluginField("sampling_ratio"),
                                                                           PluginField("gamma")});

DeformablePoolPluginDynamic::DeformablePoolPluginDynamic(
        const std::string &name, 
        const nvinfer1::Dims &outSize,
        float spatialScale,
        int samplingRatio,
        float gamma)
    : mLayerName(name),
      mOutSize(outSize),
      mSpatialScale(spatialScale),
      mSamplingRatio(samplingRatio),
      mGamma(gamma)
{

}

DeformablePoolPluginDynamic::DeformablePoolPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mOutSize);
    deserialize_value(&data, &length, &mSpatialScale);
    deserialize_value(&data, &length, &mSamplingRatio);
    deserialize_value(&data, &length, &mGamma);
    
}

nvinfer1::IPluginV2DynamicExt *DeformablePoolPluginDynamic::clone() const
{
    DeformablePoolPluginDynamic *plugin = new DeformablePoolPluginDynamic(mLayerName,
                                                                        mOutSize,
                                                                        mSpatialScale,
                                                                        mSamplingRatio,
                                                                        mGamma);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs DeformablePoolPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    assert(nbInputs == 2 || nbInputs==3);

    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = inputs[1].d[0];
    ret.d[1] = inputs[0].d[1];
    ret.d[2] = exprBuilder.constant(mOutSize.d[0]);
    ret.d[3] = exprBuilder.constant(mOutSize.d[1]);

    return ret;
}

bool DeformablePoolPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    // assert(0 <= pos && pos < 2);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
        inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void DeformablePoolPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(nbInputs == 2 || nbInputs==3);

}

size_t DeformablePoolPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int DeformablePoolPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{
    int channels = inputDesc[0].dims.d[1];
    int height = inputDesc[0].dims.d[2];
    int width = inputDesc[0].dims.d[3];

    float* input = (float*)inputs[0];
    float* rois = (float*)inputs[1];

    float* offset = NULL;
    if(inputDesc[2].dims.nbDims!=0){
        offset = (float*)inputs[2];
    }

    float* output = (float*)outputs[0];

    int output_size = 1;
    for(int i=0;i<outputDesc[0].dims.nbDims;++i){
        output_size*=outputDesc[0].dims.d[i];
    }

    int pooled_height = mOutSize.d[0];
    int pooled_width = mOutSize.d[1];
    
    deform_roi_pool_forward(input, rois, offset,
                                output, pooled_height, pooled_width,
                                output_size, channels, height, width,
                                mSpatialScale, mSamplingRatio,
                                mGamma,
                                stream);

    return 0;
}

nvinfer1::DataType DeformablePoolPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods
const char *DeformablePoolPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *DeformablePoolPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int DeformablePoolPluginDynamic::getNbOutputs() const
{
    return 1;
}

int DeformablePoolPluginDynamic::initialize()
{
    return 0;
}

void DeformablePoolPluginDynamic::terminate()
{
    
}

size_t DeformablePoolPluginDynamic::getSerializationSize() const
{
    return sizeof(mOutSize) +
           sizeof(mSpatialScale) +
           sizeof(mSamplingRatio) +
           sizeof(mGamma);
}

void DeformablePoolPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mOutSize);
    serialize_value(&buffer, mSpatialScale);
    serialize_value(&buffer, mSamplingRatio);
    serialize_value(&buffer, mGamma);

}

void DeformablePoolPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void DeformablePoolPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *DeformablePoolPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

DeformablePoolPluginDynamicCreator::DeformablePoolPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *DeformablePoolPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *DeformablePoolPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *DeformablePoolPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *DeformablePoolPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    nvinfer1::Dims outSize = {2, {7, 7}};
    float spatialScale = 1.;
    int samplingRatio = 0.;
    float gamma = 0.1;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("out_size") == 0)
        {
            outSize.nbDims = 2;
            outSize.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
            outSize.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
        }

        if (field_name.compare("spatial_scale") == 0)
        {
            spatialScale = static_cast<const float *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("sampling_ratio") == 0)
        {
            samplingRatio = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("gamma") == 0)
        {
            gamma = static_cast<const float *>(fc->fields[i].data)[0];
        }
    }

    DeformablePoolPluginDynamic *plugin = new DeformablePoolPluginDynamic(name,
                                                                    outSize,
                                                                    spatialScale,
                                                                    samplingRatio,
                                                                    gamma);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *DeformablePoolPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new DeformablePoolPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void DeformablePoolPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *DeformablePoolPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
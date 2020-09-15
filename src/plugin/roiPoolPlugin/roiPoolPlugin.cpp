
#include <assert.h>
#include <chrono>
#include "plugin/roiPoolPlugin/roiPoolPlugin.h"
#include "roi_pool.h"
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
static const char *PLUGIN_NAME{"RoiPoolPluginDynamic"};
} // namespace

PluginFieldCollection RoiPoolPluginDynamicCreator::mFC{};
std::vector<PluginField> RoiPoolPluginDynamicCreator::mPluginAttributes({PluginField("out_size"),
                                                                           PluginField("featmap_strides"),
                                                                           PluginField("roi_scale_factor"),
                                                                           PluginField("finest_scale")});

RoiPoolPluginDynamic::RoiPoolPluginDynamic(
        const std::string &name, 
        int outSize,
        const std::vector<float>& featmapStrides,
        float roiScaleFactor,
        int finestScale)
    : mLayerName(name),
      mOutSize(outSize),
      mFeatmapStrides(featmapStrides),
      mRoiScaleFactor(roiScaleFactor),
      mFinestScale(finestScale)
{

}

RoiPoolPluginDynamic::RoiPoolPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mOutSize);
    deserialize_value(&data, &length, &mRoiScaleFactor);
    deserialize_value(&data, &length, &mFinestScale);
    
    int strides_size = 0;
    deserialize_value(&data, &length, &strides_size);

    const char *d = static_cast<const char *>(data);

    float *strides_data = (float*)deserToHost<char>(d, strides_size * sizeof(float));
    mFeatmapStrides = std::vector<float>(&strides_data[0], &strides_data[0]+strides_size);
    initialize();
}

nvinfer1::IPluginV2DynamicExt *RoiPoolPluginDynamic::clone() const
{
    RoiPoolPluginDynamic *plugin = new RoiPoolPluginDynamic(mLayerName,
                                                            mOutSize,
                                                            mFeatmapStrides,
                                                            mRoiScaleFactor,
                                                            mFinestScale);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs RoiPoolPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    assert(nbInputs == mFeatmapStrides.size()+1);

    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[1].d[1];
    ret.d[2] = exprBuilder.constant(mOutSize);
    ret.d[3] = exprBuilder.constant(mOutSize);

    return ret;
}

bool RoiPoolPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    // assert(0 <= pos && pos < 2);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
        inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void RoiPoolPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
    // assert(nbOutputs == 1);
    assert(nbInputs == mFeatmapStrides.size()+1);

}

size_t RoiPoolPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int RoiPoolPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{
    int num_rois = inputDesc[0].dims.d[0];
    int batch_size = inputDesc[1].dims.d[0];
    int channels = inputDesc[1].dims.d[1];

    const int kMaxFeatMap=10;
    int heights[kMaxFeatMap];
    int widths[kMaxFeatMap];
    float strides[kMaxFeatMap];

    int num_feats = mFeatmapStrides.size();
    for(int i=0;i<num_feats;++i){
        heights[i] = inputDesc[i+1].dims.d[2];
        widths[i] = inputDesc[i+1].dims.d[3];
        strides[i] = mFeatmapStrides[i];
    }

    const void* rois = inputs[0];
    const void *const *feats = inputs+1;

    roi_pool<float>((float*)outputs[0], 
                        (const float*)rois, num_rois,
                        feats, num_feats,
                        batch_size, channels, &heights[0], &widths[0], &strides[0],
                        mOutSize, mRoiScaleFactor, mFinestScale,
                        stream
                        );

    return 0;
}

nvinfer1::DataType RoiPoolPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods
const char *RoiPoolPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *RoiPoolPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int RoiPoolPluginDynamic::getNbOutputs() const
{
    return 1;
}

int RoiPoolPluginDynamic::initialize()
{
    return 0;
}

void RoiPoolPluginDynamic::terminate()
{
    
}

size_t RoiPoolPluginDynamic::getSerializationSize() const
{
    return mFeatmapStrides.size() * sizeof(float) +
           sizeof(mOutSize) +
           sizeof(mRoiScaleFactor) +
           sizeof(mFinestScale) +
           sizeof(int);
}

void RoiPoolPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mOutSize);
    serialize_value(&buffer, mRoiScaleFactor);
    serialize_value(&buffer, mFinestScale);

    int strides_size = mFeatmapStrides.size();
    serialize_value(&buffer, strides_size);

    char *d = static_cast<char *>(buffer);
    serFromHost(d, mFeatmapStrides.data(), strides_size);
}

void RoiPoolPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void RoiPoolPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *RoiPoolPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

RoiPoolPluginDynamicCreator::RoiPoolPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *RoiPoolPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *RoiPoolPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *RoiPoolPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *RoiPoolPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    int outSize = 7;
    std::vector<float> featmapStrides;
    float roiScaleFactor = -1;
    int finestScale=56;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("out_size") == 0)
        {
            outSize = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("roi_scale_factor") == 0)
        {
            roiScaleFactor = static_cast<const float *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("finest_scale") == 0)
        {
            finestScale = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("featmap_strides") == 0)
        {
            int data_size= fc->fields[i].length;
            const float* data_start = static_cast<const float *>(fc->fields[i].data);
            featmapStrides = std::vector<float>(data_start, data_start+data_size);
        }

    }

    if (featmapStrides.size()==0)
    {
        gLogError << "featmap_strides is zero" << std::endl;
    }

    RoiPoolPluginDynamic *plugin = new RoiPoolPluginDynamic(name,
                                                            outSize,
                                                            featmapStrides,
                                                            roiScaleFactor,
                                                            finestScale);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *RoiPoolPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new RoiPoolPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void RoiPoolPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *RoiPoolPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
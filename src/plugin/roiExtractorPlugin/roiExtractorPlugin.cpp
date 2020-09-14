
#include <assert.h>
#include <chrono>
#include "plugin/roiExtractorPlugin/roiExtractorPlugin.h"
// #include "group_norm.h"
#include "roi_extractor.h"
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
static const char *PLUGIN_NAME{"RoiExtractorPluginDynamic"};
} // namespace

PluginFieldCollection RoiExtractorPluginDynamicCreator::mFC{};
std::vector<PluginField> RoiExtractorPluginDynamicCreator::mPluginAttributes({PluginField("out_size"),
                                                                           PluginField("sample_num"),
                                                                           PluginField("featmap_strides"),
                                                                           PluginField("roi_scale_factor"),
                                                                           PluginField("finest_scale"),
                                                                           PluginField("aligned")});

RoiExtractorPluginDynamic::RoiExtractorPluginDynamic(
    const std::string &name, 
        int outSize,
        int sampleNum,
        const std::vector<float>& featmapStrides,
        float roiScaleFactor,
        int finestScale,
        bool aligned)
    : mLayerName(name),
      mOutSize(outSize),
      mSampleNum(sampleNum),
      mFeatmapStrides(featmapStrides),
      mRoiScaleFactor(roiScaleFactor),
      mFinestScale(finestScale),
      mAligned(aligned)
{

}

RoiExtractorPluginDynamic::RoiExtractorPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mOutSize);
    deserialize_value(&data, &length, &mSampleNum);
    deserialize_value(&data, &length, &mRoiScaleFactor);
    deserialize_value(&data, &length, &mFinestScale);
    deserialize_value(&data, &length, &mAligned);
    
    int strides_size = 0;
    deserialize_value(&data, &length, &strides_size);

    const char *d = static_cast<const char *>(data);

    float *strides_data = (float*)deserToHost<char>(d, strides_size * sizeof(float));
    mFeatmapStrides = std::vector<float>(&strides_data[0], &strides_data[0]+strides_size);
    initialize();
}

nvinfer1::IPluginV2DynamicExt *RoiExtractorPluginDynamic::clone() const
{
    RoiExtractorPluginDynamic *plugin = new RoiExtractorPluginDynamic(mLayerName,
                                                                        mOutSize,
                                                                        mSampleNum,
                                                                        mFeatmapStrides,
                                                                        mRoiScaleFactor,
                                                                        mFinestScale,
                                                                        mAligned);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs RoiExtractorPluginDynamic::getOutputDimensions(
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

bool RoiExtractorPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    // assert(0 <= pos && pos < 2);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
        inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void RoiExtractorPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(nbInputs == mFeatmapStrides.size()+1);

}

size_t RoiExtractorPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int RoiExtractorPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
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

    roi_extractor<float>((float*)outputs[0], 
                        (const float*)rois, num_rois,
                        feats, num_feats,
                        batch_size, channels, &heights[0], &widths[0], &strides[0],
                        mOutSize, mSampleNum, mRoiScaleFactor, mFinestScale, mAligned,
                        stream
                        );

    return 0;
}

nvinfer1::DataType RoiExtractorPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods
const char *RoiExtractorPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *RoiExtractorPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int RoiExtractorPluginDynamic::getNbOutputs() const
{
    return 1;
}

int RoiExtractorPluginDynamic::initialize()
{
    return 0;
}

void RoiExtractorPluginDynamic::terminate()
{
    
}

size_t RoiExtractorPluginDynamic::getSerializationSize() const
{
    return mFeatmapStrides.size() * sizeof(float) +
           sizeof(mOutSize) +
           sizeof(mSampleNum) +
           sizeof(mRoiScaleFactor) +
           sizeof(mFinestScale) +
           sizeof(mAligned) +
           sizeof(int);
}

void RoiExtractorPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mOutSize);
    serialize_value(&buffer, mSampleNum);
    serialize_value(&buffer, mRoiScaleFactor);
    serialize_value(&buffer, mFinestScale);
    serialize_value(&buffer, mAligned);

    int strides_size = mFeatmapStrides.size();
    serialize_value(&buffer, strides_size);

    char *d = static_cast<char *>(buffer);
    serFromHost(d, mFeatmapStrides.data(), strides_size);
}

void RoiExtractorPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void RoiExtractorPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *RoiExtractorPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

RoiExtractorPluginDynamicCreator::RoiExtractorPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *RoiExtractorPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *RoiExtractorPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *RoiExtractorPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *RoiExtractorPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    int outSize = 7;
    int sampleNum = 2;
    std::vector<float> featmapStrides;
    float roiScaleFactor = -1;
    int finestScale=56;
    bool aligned=false;

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

        if (field_name.compare("sample_num") == 0)
        {
            sampleNum = static_cast<const int *>(fc->fields[i].data)[0];
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

        if (field_name.compare("aligned") == 0)
        {
            int aligned_int = static_cast<const int *>(fc->fields[i].data)[0];
            aligned = aligned_int!=0;
        }

    }

    if (featmapStrides.size()==0)
    {
        gLogError << "featmap_strides is zero" << std::endl;
    }

    RoiExtractorPluginDynamic *plugin = new RoiExtractorPluginDynamic(name,
                                                                    outSize,
                                                                    sampleNum,
                                                                    featmapStrides,
                                                                    roiScaleFactor,
                                                                    finestScale,
                                                                    aligned);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *RoiExtractorPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new RoiExtractorPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void RoiExtractorPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *RoiExtractorPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
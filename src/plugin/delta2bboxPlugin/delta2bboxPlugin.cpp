
#include <assert.h>
#include <chrono>
#include "plugin/delta2bboxPlugin/delta2bboxPlugin.h"
#include "delta2bbox.h"
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
static const char *PLUGIN_NAME{"Delta2BBoxPluginDynamic"};
} // namespace

PluginFieldCollection Delta2BBoxPluginDynamicCreator::mFC{};
std::vector<PluginField> Delta2BBoxPluginDynamicCreator::mPluginAttributes({PluginField("use_sigmoid_cls"),
                                                                            PluginField("min_num_bbox"),
                                                                            PluginField("num_classes"),
                                                                           PluginField("target_means"),
                                                                           PluginField("target_stds")});

Delta2BBoxPluginDynamic::Delta2BBoxPluginDynamic(
    const std::string &name,
        bool useSigmoidCls,
        int minNumBBox,
        int numClasses,
        const std::vector<float> &targetMeans,
        const std::vector<float> &targetStds)
    : mLayerName(name),
      mUseSigmoidCls(useSigmoidCls),
      mMinNumBBox(minNumBBox),
      mNumClasses(numClasses),
      mTargetMeans(targetMeans),
      mTargetStds(targetStds)
{

}

Delta2BBoxPluginDynamic::Delta2BBoxPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mUseSigmoidCls);
    deserialize_value(&data, &length, &mMinNumBBox);
    deserialize_value(&data, &length, &mNumClasses);
    
    const int mean_std_size=4;

    const char *d = static_cast<const char *>(data);

    float *mean_data = (float*)deserToHost<char>(d, mean_std_size * sizeof(float));
    mTargetMeans = std::vector<float>(&mean_data[0], &mean_data[0]+mean_std_size);
    float *std_data = (float*)deserToHost<char>(d, mean_std_size * sizeof(float));
    mTargetStds = std::vector<float>(&std_data[0], &std_data[0]+mean_std_size);
    initialize();
}

nvinfer1::IPluginV2DynamicExt *Delta2BBoxPluginDynamic::clone() const
{
    Delta2BBoxPluginDynamic *plugin = new Delta2BBoxPluginDynamic(mLayerName,
                                                                        mUseSigmoidCls,
                                                                        mMinNumBBox,
                                                                        mNumClasses,
                                                                        mTargetMeans,
                                                                        mTargetStds);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs Delta2BBoxPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    assert(nbInputs == 3 || nbInputs == 4);

    auto out_bbox = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
                                        *exprBuilder.operation(
                                            nvinfer1::DimensionOperation::kFLOOR_DIV,
                                            *inputs[0].d[1],
                                            *exprBuilder.constant(mNumClasses)
                                        ),
                                        *exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, 
                                                            *(inputs[0].d[2]), 
                                                            *(inputs[0].d[3]))
                                        );
    
    auto final_bbox = exprBuilder.operation(nvinfer1::DimensionOperation::kMAX,
                                        *exprBuilder.constant(mMinNumBBox), *out_bbox
                                        );

    nvinfer1::DimsExprs ret;
    ret.nbDims = outputIndex==0 ? 3 : 4;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = final_bbox;
    ret.d[2] = exprBuilder.constant(mNumClasses);

    if(outputIndex==1){
        ret.d[2] = exprBuilder.constant(1);
        ret.d[3] = exprBuilder.constant(4);
    }

    return ret;
}

bool Delta2BBoxPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    switch(pos){
    case 0:
        return in[0].type == nvinfer1::DataType::kFLOAT && in[0].format == nvinfer1::TensorFormat::kLINEAR;
    case 1:
        return in[1].type == in[0].type && in[1].format == in[0].format;
    case 2:
        return in[2].type == in[0].type && in[2].format == in[0].format;
    }

    // output
    switch(pos-nbInputs){
        case 0:
            return out[0].type == in[0].type && out[0].format == in[0].format;
        case 1:
            return out[1].type == in[0].type && out[1].format == in[0].format;
    }

    if(nbInputs==4 && pos == 3){
        return in[3].type == nvinfer1::DataType::kINT32;
    }
    
    return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
        inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void Delta2BBoxPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
}

size_t Delta2BBoxPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int Delta2BBoxPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{
    int batch_size = inputDesc[0].dims.d[0];
    int num_inboxes = inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
    int num_ratios = inputDesc[0].dims.d[1]/mNumClasses;
    int num_outboxes = outputDesc[0].dims.d[1];

    float* out_cls = (float*)outputs[0];
    float* out_bbox = (float*)outputs[1];
    const float* in_cls = (float*)inputs[0];
    const float* in_bbox = (float*)inputs[1];
    const float* anchor = (float*)inputs[2];

    int* clip_range = nullptr;
    if(inputDesc[3].dims.nbDims>0){
        clip_range = (int*)(inputs[3]) + inputDesc[3].dims.d[0] - 2;
    }

    cudaMemsetAsync(out_cls,
                    0, 
                    outputDesc[0].dims.d[0]*outputDesc[0].dims.d[1]*outputDesc[0].dims.d[2]*sizeof(float), 
                    stream);
    cudaMemsetAsync(out_bbox,
                    0, 
                    outputDesc[1].dims.d[0]*outputDesc[1].dims.d[1]*outputDesc[1].dims.d[2]*outputDesc[1].dims.d[3]*sizeof(float), 
                    stream);
    delta2bbox<float>(out_cls, out_bbox,
                in_cls, in_bbox, anchor, clip_range,
                batch_size, num_inboxes, num_outboxes, mNumClasses, num_ratios,
                mUseSigmoidCls,
                mTargetMeans.data(), mTargetStds.data(),
                stream);
    return 0;
}

nvinfer1::DataType Delta2BBoxPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods
const char *Delta2BBoxPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *Delta2BBoxPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int Delta2BBoxPluginDynamic::getNbOutputs() const
{
    return 2;
}

int Delta2BBoxPluginDynamic::initialize()
{
    return 0;
}

void Delta2BBoxPluginDynamic::terminate()
{
    
}

size_t Delta2BBoxPluginDynamic::getSerializationSize() const
{
    return mTargetMeans.size() * sizeof(float) +
            mTargetStds.size() * sizeof(float) +
           sizeof(mUseSigmoidCls) +
           sizeof(mMinNumBBox) + 
           sizeof(mNumClasses);
}

void Delta2BBoxPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mUseSigmoidCls);
    serialize_value(&buffer, mMinNumBBox);
    serialize_value(&buffer, mNumClasses);
    
    const int mean_std_size = 4;

    char *d = static_cast<char *>(buffer);
    serFromHost(d, mTargetMeans.data(), mean_std_size);
    serFromHost(d, mTargetStds.data(), mean_std_size);
}

void Delta2BBoxPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void Delta2BBoxPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *Delta2BBoxPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

Delta2BBoxPluginDynamicCreator::Delta2BBoxPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *Delta2BBoxPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *Delta2BBoxPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *Delta2BBoxPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *Delta2BBoxPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    bool useSigmoidCls=true;
    int minNumBBox = 1000;
    int numClasses = 1;
    std::vector<float> targetMeans;
    std::vector<float> targetStds;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("use_sigmoid_cls") == 0)
        {
            useSigmoidCls = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("min_num_bbox") == 0)
        {
            minNumBBox = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("num_classes") == 0)
        {
            numClasses = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("target_means") == 0)
        {
            int data_size= fc->fields[i].length;
            const float* data_start = static_cast<const float *>(fc->fields[i].data);
            targetMeans = std::vector<float>(data_start, data_start+data_size);
        }

        if (field_name.compare("target_stds") == 0)
        {
            int data_size= fc->fields[i].length;
            const float* data_start = static_cast<const float *>(fc->fields[i].data);
            targetStds = std::vector<float>(data_start, data_start+data_size);
        }

    }

    if (targetMeans.size()!=4)
    {
        gLogError << "target mean must contain 4 elements" << std::endl;
    }

    if (targetStds.size()!=4)
    {
        gLogError << "target std must contain 4 elements" << std::endl;
    }

    Delta2BBoxPluginDynamic *plugin = new Delta2BBoxPluginDynamic(  name,
                                                                    useSigmoidCls,
                                                                    minNumBBox,
                                                                    numClasses,
                                                                    targetMeans,
                                                                    targetStds);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *Delta2BBoxPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new Delta2BBoxPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void Delta2BBoxPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *Delta2BBoxPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan

#include <assert.h>
#include <chrono>
#include <math.h>
#include "plugin/gridAnchorDynamicPlugin/gridAnchorDynamicPlugin.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "grid_anchor_dynamic.h"

namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"GridAnchorDynamicPluginDynamic"};
} // namespace

PluginFieldCollection GridAnchorDynamicPluginDynamicCreator::mFC{};
std::vector<PluginField> GridAnchorDynamicPluginDynamicCreator::mPluginAttributes({
                                                                                PluginField("base_size"),
                                                                                PluginField("stride"),
                                                                                PluginField("scales"),
                                                                                PluginField("ratios"),
                                                                                PluginField("scale_major"),
                                                                                PluginField("center_x"),
                                                                                PluginField("center_y"),
                                                                                PluginField("base_anchors"),
                                                                                });

GridAnchorDynamicPluginDynamic::GridAnchorDynamicPluginDynamic(
        const std::string &name,
        int baseSize, int stride,
        const std::vector<float> &scales, const std::vector<float> &ratios,
        bool scaleMajor,
        int centerX, int centerY)
    : mLayerName(name),
      mBaseSize(baseSize),
      mStride(stride),
      mScales(scales),
      mRatios(ratios),
      mScaleMajor(scaleMajor),
      mCenterX(centerX),
      mCenterY(centerY)
{
    mIsInitialed = false;
    mDevBaseAnchor = nullptr;
    generateBaseAnchor();
    initialize();
}


GridAnchorDynamicPluginDynamic::GridAnchorDynamicPluginDynamic(
    const std::string &name, 
    int baseSize, int stride,
    const std::vector<float> &baseAnchors
    )
    : mLayerName(name),
      mBaseSize(baseSize),
      mStride(stride),
      mScales(std::vector<float>()),
      mRatios(std::vector<float>()),
      mScaleMajor(true),
      mCenterX(0),
      mCenterY(0)
{
    mIsInitialed = false;
    mDevBaseAnchor = nullptr;
    mNumBaseAnchor = baseAnchors.size()/4;
    mHostBaseAnchor = std::shared_ptr<float>((float*)malloc(mNumBaseAnchor * 4 * sizeof(float)));
    memcpy(mHostBaseAnchor.get(), baseAnchors.data(), mNumBaseAnchor * 4 * sizeof(float));
    initialize();
}

void set_base_anchor(int rid, int sid,
float x_ctr, float y_ctr,
const std::vector<float>& h_ratios, const std::vector<float>& w_ratios, const std::vector<float>& scales,
int anchor_count, int baseSize,  
std::shared_ptr<float>& hostBaseAnchor){
    float scale = scales[sid];
    float h_ratio = h_ratios[rid];
    float w_ratio = w_ratios[rid];
    float ws = float(baseSize)*w_ratio*scale;
    float hs = float(baseSize)*h_ratio*scale;

    float half_ws = (ws)/2;
    float half_hs = (hs)/2;
    
    hostBaseAnchor.get()[anchor_count*4] = x_ctr-half_ws;
    hostBaseAnchor.get()[anchor_count*4+1] = y_ctr-half_hs;
    hostBaseAnchor.get()[anchor_count*4+2] = x_ctr+half_ws;
    hostBaseAnchor.get()[anchor_count*4+3] = y_ctr+half_hs;

}

void GridAnchorDynamicPluginDynamic::generateBaseAnchor(){
    int num_scales = mScales.size();
    int num_ratios = mRatios.size();
    float x_ctr = mCenterX;
    float y_ctr = mCenterY;
    if(x_ctr<0){
        x_ctr = (mBaseSize-1)/2;
    }
    if(y_ctr<0){
        y_ctr = (mBaseSize-1)/2;
    }

    std::vector<float> h_ratios, w_ratios;
    h_ratios.resize(num_ratios);
    w_ratios.resize(num_ratios);
    std::transform(mRatios.begin(), mRatios.end(), h_ratios.begin(), [](float x)->float{ return sqrt(x); });
    std::transform(h_ratios.begin(), h_ratios.end(), w_ratios.begin(), [](float x)->float{ return 1/x; });
    mNumBaseAnchor = num_scales * num_ratios;
    mHostBaseAnchor = std::shared_ptr<float>((float*)malloc(mNumBaseAnchor * 4 * sizeof(float)));
    int anchor_count = 0;
    if(mScaleMajor){
        for(int rid=0;rid<num_ratios;++rid){
            for(int sid=0;sid<num_scales;++sid){
                set_base_anchor(rid, sid,
                 x_ctr, y_ctr,
                  h_ratios, w_ratios, mScales,
                  anchor_count, mBaseSize,
                  mHostBaseAnchor);
                anchor_count+=1;
            }
        }
    }else{
        for(int sid=0;sid<num_scales;++sid){
            for(int rid=0;rid<num_ratios;++rid){
                set_base_anchor(rid, sid,
                 x_ctr, y_ctr,
                  h_ratios, w_ratios, mScales,
                  anchor_count, mBaseSize,
                  mHostBaseAnchor);
                anchor_count+=1;
            }
        }
    }
}

GridAnchorDynamicPluginDynamic::GridAnchorDynamicPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    const void *start = data;
    deserialize_value(&data, &length, &mBaseSize);
    deserialize_value(&data, &length, &mStride);
    deserialize_value(&data, &length, &mScaleMajor);
    deserialize_value(&data, &length, &mCenterX);
    deserialize_value(&data, &length, &mCenterY);
    deserialize_value(&data, &length, &mNumBaseAnchor);

    int scale_size=0;
    int ratio_size=0;
    deserialize_value(&data, &length, &scale_size);
    deserialize_value(&data, &length, &ratio_size);

    const char *d = static_cast<const char *>(data);

    float* scales_data = (float*)deserToHost<char>(d, scale_size*sizeof(float));
    mScales = std::vector<float>(&scales_data[0], &scales_data[0]+scale_size);
    
    float* ratios_data = (float*)deserToHost<char>(d, ratio_size*sizeof(float));
    mRatios = std::vector<float>(&ratios_data[0], &ratios_data[0]+ratio_size);

    // mWdev = deserToDev<char>(d, mNumParamsW * wordSize);
    float* baseanchor_data = (float*)deserToHost<char>(d, mNumBaseAnchor * 4 * sizeof(float));
    mHostBaseAnchor = std::shared_ptr<float>((float *)baseanchor_data);

    assert((start + getSerializationSize()) == d);
    // mW.values = nullptr;
    mIsInitialed = false;
    initialize();
}


nvinfer1::IPluginV2DynamicExt *GridAnchorDynamicPluginDynamic::clone() const
{
    // GridAnchorDynamicPluginDynamic *plugin = new GridAnchorDynamicPluginDynamic(mLayerName,
    //                                                                       mBaseSize, mStride,
    //                                                                       mScales, mRatios,
    //                                                                       mScaleMajor,
    //                                                                       mCenterX, mCenterY);
    std::vector<float> baseAnchors(mHostBaseAnchor.get(), mHostBaseAnchor.get()+mNumBaseAnchor*4);
    GridAnchorDynamicPluginDynamic *plugin = new GridAnchorDynamicPluginDynamic(mLayerName,
                                                                          mBaseSize, mStride,
                                                                          baseAnchors);
    plugin->setPluginNamespace(getPluginNamespace());
    
    return plugin;
}

nvinfer1::DimsExprs GridAnchorDynamicPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    assert(nbInputs == 1);
    assert(inputs[0].nbDims == 4);

    nvinfer1::DimsExprs ret;
    ret.nbDims = 2;

    auto area = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, 
                        *inputs[0].d[2],
                        *inputs[0].d[3]);
    
    ret.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, 
                        *area,
                        *exprBuilder.constant(mNumBaseAnchor));
    ret.d[1] = exprBuilder.constant(4);

    return ret;
}

bool GridAnchorDynamicPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    assert(0 <= pos && pos < 2);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    switch (pos)
    {
    case 0:
        return in[0].type == nvinfer1::DataType::kFLOAT &&
            in[0].format == nvinfer1::TensorFormat::kLINEAR;
        // return true;
    case 1:
        return out[0].type == in[0].type &&
               out[0].format == in[0].format;
    }
}

void GridAnchorDynamicPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(nbInputs == 1);
    // const auto &inDims0 = inputs[0].desc.dims;
}

size_t GridAnchorDynamicPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int GridAnchorDynamicPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                         const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{
    int batch_size = inputDesc[0].dims.d[0];
    int inputChannel = inputDesc[0].dims.d[1];
    int inputHeight = inputDesc[0].dims.d[2];
    int inputWidth = inputDesc[0].dims.d[3];

    grid_anchor_dynamic<float>((float*)outputs[0], mDevBaseAnchor, 
    inputWidth, inputHeight, 
    mStride, mNumBaseAnchor, stream);
    
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType GridAnchorDynamicPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    assert(nbInputs == 1);
    return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods
const char *GridAnchorDynamicPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *GridAnchorDynamicPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int GridAnchorDynamicPluginDynamic::getNbOutputs() const
{
    return 1;
}

int GridAnchorDynamicPluginDynamic::initialize()
{
    if(mNumBaseAnchor>0 && (!mIsInitialed)){
        size_t nbBytes = mNumBaseAnchor * 4 * sizeof(float);
        CHECK(cudaMalloc((void **)&mDevBaseAnchor, nbBytes));
        CHECK(cudaMemcpy((void*)mDevBaseAnchor, (void*)mHostBaseAnchor.get(), nbBytes, cudaMemcpyHostToDevice));
        mIsInitialed=true;
    }
    return 0;
}

void GridAnchorDynamicPluginDynamic::terminate()
{
    if(mDevBaseAnchor!=nullptr && mIsInitialed)
    {
        cudaFree(mDevBaseAnchor);
        mDevBaseAnchor = nullptr;
        mIsInitialed = false;
    }
}

size_t GridAnchorDynamicPluginDynamic::getSerializationSize() const
{
    return sizeof(mBaseSize) +
           sizeof(mStride) +
           sizeof(mScaleMajor) +
           sizeof(mCenterX) +
           sizeof(mCenterY) +
           sizeof(mNumBaseAnchor) + 
           sizeof(int)*2 +
           sizeof(float)*mScales.size() +
           sizeof(float)*mRatios.size() + 
           sizeof(float) * mNumBaseAnchor * 4 ;
}

void GridAnchorDynamicPluginDynamic::serialize(void *buffer) const
{
    const void* start = buffer;
    serialize_value(&buffer, mBaseSize);
    serialize_value(&buffer, mStride);
    serialize_value(&buffer, mScaleMajor);
    serialize_value(&buffer, mCenterX);
    serialize_value(&buffer, mCenterY);
    serialize_value(&buffer, mNumBaseAnchor);
    
    int scale_size = mScales.size();
    int ratio_size = mRatios.size();
    serialize_value(&buffer, scale_size);
    serialize_value(&buffer, ratio_size);

    char *d = static_cast<char *>(buffer);
    serFromHost(d, mScales.data(), scale_size);
    serFromHost(d, mRatios.data(), ratio_size);
    serFromHost(d, mHostBaseAnchor.get(), mNumBaseAnchor * 4);

    assert((start + getSerializationSize()) == d);
}

void GridAnchorDynamicPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void GridAnchorDynamicPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *GridAnchorDynamicPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

GridAnchorDynamicPluginDynamicCreator::GridAnchorDynamicPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *GridAnchorDynamicPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *GridAnchorDynamicPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *GridAnchorDynamicPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *GridAnchorDynamicPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{
    int outDims = 0;
    
    int base_size;
    int stride;
    std::vector<float> scales;
    std::vector<float> ratios;
    bool scale_major=true;
    int center_x=-1;
    int center_y=-1;
    std::vector<float> base_anchors;


    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("base_size") == 0)
        {
            base_size = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("stride") == 0)
        {
            stride = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("scales") == 0)
        {
            int data_size= fc->fields[i].length;
            const float* data_start = static_cast<const float *>(fc->fields[i].data);
            scales = std::vector<float>(data_start, data_start+data_size);
        }

        if (field_name.compare("ratios") == 0)
        {
            int data_size= fc->fields[i].length;
            const float* data_start = static_cast<const float *>(fc->fields[i].data);
            ratios = std::vector<float>(data_start, data_start+data_size);
        }

        if (field_name.compare("scale_major") == 0)
        {
            scale_major = (bool)static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("center_x") == 0)
        {
            center_x = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("center_y") == 0)
        {
            center_y = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("base_anchors") == 0)
        {
            int data_size= fc->fields[i].length;
            const float* data_start = static_cast<const float *>(fc->fields[i].data);
            base_anchors = std::vector<float>(data_start, data_start+data_size);
        }
    }

    GridAnchorDynamicPluginDynamic *plugin;
    if(base_anchors.size()>0){
        plugin = new GridAnchorDynamicPluginDynamic(name, base_size, stride, 
                                                    base_anchors);

    }else{
        plugin = new GridAnchorDynamicPluginDynamic(name, base_size, stride, 
                                                    scales, ratios,
                                                    scale_major, 
                                                    center_x, center_y);
    }
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

IPluginV2 *GridAnchorDynamicPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new GridAnchorDynamicPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void GridAnchorDynamicPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *GridAnchorDynamicPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
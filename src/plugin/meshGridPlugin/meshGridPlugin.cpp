
#include <assert.h>
#include <chrono>
#include "plugin/meshGridPlugin/meshGridPlugin.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "amir_cuda_util/cuda_util.h"
#include "mesh_grid.h"

namespace amirstan
{
namespace plugin
{

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"MeshGridPluginDynamic"};
} // namespace

PluginFieldCollection MeshGridPluginDynamicCreator::mFC{};
std::vector<PluginField> MeshGridPluginDynamicCreator::mPluginAttributes({
                                                                        PluginField("num_inputs"),
                                                                        PluginField("slice_dims"),
                                                                        PluginField("starts"),
                                                                        PluginField("strides")});

MeshGridPluginDynamic::MeshGridPluginDynamic(
        const std::string &name, int numInputs, const nvinfer1::Dims &sliceDims, const std::vector<float> &starts,const std::vector<float> &strides)
    : mLayerName(name),
      mNumInputs(numInputs),
      mSliceDims(sliceDims),
      mStarts(starts),
      mStrides(strides)
{

}

MeshGridPluginDynamic::MeshGridPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mNumInputs);
    deserialize_value(&data, &length, &mSliceDims);
    
    int start_length, stride_length;
    deserialize_value(&data, &length, &start_length);
    deserialize_value(&data, &length, &stride_length);
    
    const char *d = static_cast<const char *>(data);
    float *start_data = (float*)deserToHost<char>(d, start_length * sizeof(float));
    mStarts = std::vector<float>(&start_data[0], &start_data[0]+start_length);
    float *stride_data = (float*)deserToHost<char>(d, stride_length * sizeof(float));
    mStrides = std::vector<float>(&stride_data[0], &stride_data[0]+stride_length);
}

nvinfer1::IPluginV2DynamicExt *MeshGridPluginDynamic::clone() const
{
    MeshGridPluginDynamic *plugin = new MeshGridPluginDynamic(mLayerName,
                                                            mNumInputs,
                                                            mSliceDims,
                                                            mStarts,
                                                            mStrides);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs MeshGridPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    nvinfer1::DimsExprs ret;

    // torch meshgrid
    if(mNumInputs>0){
        assert(mNumInputs == nbInputs);
        ret.nbDims = nbInputs;
        for(int i=0; i<nbInputs; ++i){
            ret.d[i] = inputs[i].d[0];
        }
    }else{
        ret.nbDims = mSliceDims.nbDims;
        for(int i=0; i<ret.nbDims; ++i){
            ret.d[i] = inputs[0].d[mSliceDims.d[i]];
        }
    }

    return ret;
}

bool MeshGridPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;

     if(pos==0)
    {
        return (in[pos].type == nvinfer1::DataType::kFLOAT && in[pos].format == nvinfer1::TensorFormat::kLINEAR)
        || (in[pos].type == nvinfer1::DataType::kHALF && in[pos].format == nvinfer1::TensorFormat::kLINEAR)
        ||(in[pos].type == nvinfer1::DataType::kINT32 && in[pos].format == nvinfer1::TensorFormat::kLINEAR);
    }else{
        return (inOut[pos].type == in[0].type && inOut[pos].format == in[0].format);
    }
}

void MeshGridPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
}

size_t MeshGridPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int MeshGridPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                    const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{

    if(mNumInputs>0){
        // torch mesh grid
        for(int i=0; i<mNumInputs; ++i){
            nvinfer1::Dims input_dims;
            nvinfer1::Dims repeat_dims;
            input_dims.nbDims = mNumInputs;
            repeat_dims.nbDims = mNumInputs;
            for(int j=0;j<mNumInputs;++j){
                input_dims.d[j]=1;
                repeat_dims.d[j]=outputDesc[i].dims.d[j];
            }
            input_dims.d[i]=inputDesc[i].dims.d[0];
            repeat_dims.d[i] = 1;
            

            auto data_type = inputDesc[0].type;

            switch(data_type){
            case nvinfer1::DataType::kFLOAT:
                amirstan::cuda::repeat_dims<float>((float*)outputs[i], (float*)inputs[i], 
                &(input_dims.d[0]), &repeat_dims.d[0], input_dims.nbDims,
                stream);
                break;

            case nvinfer1::DataType::kHALF:
                amirstan::cuda::repeat_dims<half>((half*)outputs[i], (half*)inputs[i], 
                &(input_dims.d[0]), &repeat_dims.d[0], input_dims.nbDims,
                stream);
                break;

            case nvinfer1::DataType::kINT32:
                amirstan::cuda::repeat_dims<int>((int*)outputs[i], (int*)inputs[i], 
                &(input_dims.d[0]), &repeat_dims.d[0], input_dims.nbDims,
                stream);
                break;
            default:
                return 1;
                break;
            }
        }
    }else{
        int num_output = getNbOutputs();
        for(int i=0; i<num_output; ++i){
            auto &output_dims = outputDesc[i].dims;
            auto data_type = outputDesc[i].type;
            float start = mStarts[i];
            float stride = mStrides[i];

            switch(data_type){
            case nvinfer1::DataType::kFLOAT:
                amirstan::plugin::arange_mesh_grid<float>((float *)outputs[i],
                                        &(output_dims.d[0]), output_dims.nbDims,
                                        i, start, stride,
                                        stream);
                break;

            case nvinfer1::DataType::kHALF:
                amirstan::plugin::arange_mesh_grid<half>((half *)outputs[i],
                                        &(output_dims.d[0]), output_dims.nbDims,
                                        i, start, stride,
                                        stream);
                break;

            case nvinfer1::DataType::kINT32:
                amirstan::plugin::arange_mesh_grid<int>((int *)outputs[i],
                                        &(output_dims.d[0]), output_dims.nbDims,
                                        i, start, stride,
                                        stream);
                break;
            default:
                return 1;
                break;
            }
        }
    }

    return 0;
}

nvinfer1::DataType MeshGridPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

// IPluginV2 Methods
const char *MeshGridPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *MeshGridPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int MeshGridPluginDynamic::getNbOutputs() const
{
    if(mNumInputs>0){
        return mNumInputs;
    }else{
        return mSliceDims.nbDims;
    }
}

int MeshGridPluginDynamic::initialize()
{
    return 0;
}

void MeshGridPluginDynamic::terminate()
{
}

size_t MeshGridPluginDynamic::getSerializationSize() const
{
    return sizeof(mNumInputs)
            + sizeof(mSliceDims)
            + sizeof(int)*2
            + mStarts.size() * sizeof(float)
            + mStrides.size() * sizeof(float)
            + sizeof(mStrides);
}

void MeshGridPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mNumInputs);
    serialize_value(&buffer, mSliceDims);

    int start_length = mStarts.size();
    int stride_length = mStrides.size();
    serialize_value(&buffer, start_length);
    serialize_value(&buffer, stride_length);

    char *d = static_cast<char *>(buffer);
    serFromHost(d, mStarts.data(), start_length);
    serFromHost(d, mStrides.data(), stride_length);
    
}

void MeshGridPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void MeshGridPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *MeshGridPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

MeshGridPluginDynamicCreator::MeshGridPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *MeshGridPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *MeshGridPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *MeshGridPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *MeshGridPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    int numInputs=0;
    nvinfer1::Dims sliceDims;
    sliceDims.nbDims=0;
    std::vector<float> starts;
    std::vector<float> strides;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if(field_name.compare("num_inputs")==0){
            numInputs = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("slice_dims") == 0)
        {
            sliceDims.nbDims = fc->fields[i].length;
            memcpy(&(sliceDims.d[0]), static_cast<const int *>(fc->fields[i].data), fc->fields[i].length*sizeof(int));
        }

        if (field_name.compare("starts") == 0)
        {
            int data_size= fc->fields[i].length;
            const float* data_start = static_cast<const float *>(fc->fields[i].data);
            starts = std::vector<float>(data_start, data_start+data_size);
        }

        if (field_name.compare("strides") == 0)
        {
            int data_size= fc->fields[i].length;
            const float* data_start = static_cast<const float *>(fc->fields[i].data);
            strides = std::vector<float>(data_start, data_start+data_size);
        }

    }

    assert(numInputs>0 || sliceDims.nbDims>0);

    MeshGridPluginDynamic *plugin = new MeshGridPluginDynamic(name,
                                                    numInputs,
                                                    sliceDims,
                                                    starts,
                                                    strides);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *MeshGridPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new MeshGridPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void MeshGridPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *MeshGridPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
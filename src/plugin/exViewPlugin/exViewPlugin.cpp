
#include <assert.h>
#include <chrono>

#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "expressionParser.h"
#include "plugin/exViewPlugin/exViewPlugin.h"

namespace amirstan
{
namespace plugin
{

std::vector<std::string> split_str(const std::string& str, const std::string& delim) {  
	std::vector<std::string> res;  
	if("" == str) return res;  
	char * strs = new char[str.length() + 1] ;
    char * old_strs = strs;
	strcpy(strs, str.c_str());   
 
	char * d = new char[delim.length() + 1];  
    char * old_d = d;
	strcpy(d, delim.c_str());  
 
	char *p = strtok(strs, d);  
	while(p) {  
		std::string s = p;  
		res.push_back(s); 
		p = strtok(NULL, d);  
	}  

    delete[] old_strs;
    delete[] old_d;
 
	return res;  
}

namespace
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"ExViewPluginDynamic"};
} // namespace

PluginFieldCollection ExViewPluginDynamicCreator::mFC{};
std::vector<PluginField> ExViewPluginDynamicCreator::mPluginAttributes({PluginField("dim_expression")});

ExViewPluginDynamic::ExViewPluginDynamic(
    const std::string &name, 
    std::vector<std::string> dimExpression)
    : mLayerName(name),
      mDimExpression(dimExpression)
{
}

ExViewPluginDynamic::ExViewPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    int str_length;
    deserialize_value(&data, &length, &str_length);

    const char *d = static_cast<const char *>(data);
    char* chr_dim_exp = new char[str_length+1];
    char *chr_dim_exp_data = deserToHost<char>(d, str_length*sizeof(char));
    memcpy(chr_dim_exp, chr_dim_exp_data, str_length*sizeof(char));
    chr_dim_exp[str_length] = '\0';
    std::string str_dim_exp = chr_dim_exp;
    mDimExpression = split_str(str_dim_exp, ";");

    delete[] chr_dim_exp;
    initialize();
}

nvinfer1::IPluginV2DynamicExt *ExViewPluginDynamic::clone() const
{
    ExViewPluginDynamic *plugin = new ExViewPluginDynamic(mLayerName,
                                                            mDimExpression);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs ExViewPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{

    DimsExprs outputDims;
    outputDims.nbDims = mDimExpression.size();

    for(int i=0;i<mDimExpression.size();++i){
        outputDims.d[i] = parse_expression(mDimExpression[i], inputs + 1, exprBuilder);
    }

    return outputDims;
}

bool ExViewPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut,
                                                                           int nbInputs,
                                                                           int nbOutputs)
{
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;

    if(pos ==0){
        return in[0].type == nvinfer1::DataType::kFLOAT && in[0].format == nvinfer1::TensorFormat::kLINEAR;
    }
    else{
        return inOut[pos].type == in[0].type && inOut[pos].format == in[0].format;
    }

    return true;
}

void ExViewPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
    // const auto &inDims0 = inputs[0].desc.dims;
}

size_t ExViewPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    return 0;
}

int ExViewPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                                        const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{
    
    size_t copy_size = 1;
    const nvinfer1::Dims &dims = outputDesc[0].dims;
    
    for(int i=0;i<dims.nbDims;++i){
        copy_size *= dims.d[i];
    }

    size_t wordSize = samplesCommon::getElementSize(outputDesc[0].type);
    cudaMemcpyAsync(outputs[0], inputs[0], copy_size*wordSize, cudaMemcpyDeviceToDevice, stream);

    return 0;
}

nvinfer1::DataType ExViewPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

// IPluginV2 Methods
const char *ExViewPluginDynamic::getPluginType() const
{
    return PLUGIN_NAME;
}

const char *ExViewPluginDynamic::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

int ExViewPluginDynamic::getNbOutputs() const
{
    return 1;
}

int ExViewPluginDynamic::initialize()
{
    return 0;
}

void ExViewPluginDynamic::terminate()
{
}

size_t ExViewPluginDynamic::getSerializationSize() const
{
    int str_length = 0;
    for(auto str:mDimExpression){
        str_length += str.length();
        str_length += 1;
    }
    return sizeof(int) 
    + str_length*sizeof(char);
}

void ExViewPluginDynamic::serialize(void *buffer) const
{

    int str_length = 0;
    std::string str_dim_exp;
    for(auto str:mDimExpression){
        str_dim_exp += str;
        str_dim_exp += ";";
    }
    str_length = str_dim_exp.length();
    serialize_value(&buffer, str_length);

    char *d = static_cast<char *>(buffer);
    serFromHost(d, str_dim_exp.c_str(), str_length * sizeof(char));

}

void ExViewPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void ExViewPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *ExViewPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

ExViewPluginDynamicCreator::ExViewPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *ExViewPluginDynamicCreator::getPluginName() const
{
    return PLUGIN_NAME;
}

const char *ExViewPluginDynamicCreator::getPluginVersion() const
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection *ExViewPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *ExViewPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{

    std::string str_dim_exp;

    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("dim_expression") == 0)
        {
            const char* chr_dim_exp_data = static_cast<const char *>(fc->fields[i].data);
            int str_length = (fc->fields[i].length);
            char* chr_dim_exp = new char[str_length+1];
            memcpy(chr_dim_exp, chr_dim_exp_data, sizeof(char)*str_length);
            chr_dim_exp[str_length] = '\0';
            str_dim_exp = chr_dim_exp;
            delete[] chr_dim_exp;
        }
    }

    auto dim_exp = split_str(str_dim_exp, ";");
    ExViewPluginDynamic *plugin = new ExViewPluginDynamic(name,
                                                        dim_exp);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2 *ExViewPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new ExViewPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void ExViewPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *ExViewPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan

#include <assert.h>
#include <chrono>
#include "plugin/deformableConvPlugin/modulatedDeformableConvPlugin.h"
#include "common.h"
#include "amirCommon.h"
#include "serialize.hpp"
#include "deform_conv_cuda.h"

namespace amirstan
{
namespace plugin
{

namespace
{
static const char *DCN_VERSION{"1"};
static const char *DCN_NAME{"ModulatedDeformableConvPluginDynamic"};
} // namespace

PluginFieldCollection ModulatedDeformableConvPluginDynamicCreator::mFC{};
std::vector<PluginField> ModulatedDeformableConvPluginDynamicCreator::mPluginAttributes({PluginField("out_dims"),
                                                                                PluginField("type_id"),
                                                                                PluginField("kernel_size"),
                                                                                PluginField("W"),
                                                                                PluginField("B"),
                                                                                PluginField("stride"),
                                                                                PluginField("padding"),
                                                                                PluginField("dilation"),
                                                                                PluginField("deformable_group"),
                                                                                PluginField("group")});

ModulatedDeformableConvPluginDynamic::ModulatedDeformableConvPluginDynamic(
    const std::string &name, const nvinfer1::DataType &type,
    const int outDim, const nvinfer1::Dims &kernelSize,
    const nvinfer1::Weights &W,
    const nvinfer1::Weights &B)
    : mLayerName(name),
      mType(type),
      mOutDim(outDim),
      mKernelSize(kernelSize),
      mW(W),
      mNumParamsW(W.count),
      mB(B),
      mNumParamsB(B.count)
{
    // init params
    mStride = nvinfer1::Dims{2, {1, 1}};
    mPadding = nvinfer1::Dims{2, {0, 0}};
    mDilation = nvinfer1::Dims{2, {1, 1}};
    mDeformableGroup = 1;
    mGroup = 1;

    size_t wordSize = samplesCommon::getElementSize(mType);
    mWhost = std::shared_ptr<char>(new char[mNumParamsW * wordSize]);
    memcpy((void *)mWhost.get(), mW.values, mW.count * wordSize);
    mW.values = mWhost.get();
    mBhost = std::shared_ptr<char>(new char[mNumParamsB * wordSize]);
    memcpy((void *)mBhost.get(), mB.values, mB.count * wordSize);
    mB.values = mBhost.get();
    mWdev=nullptr;
    mBdev=nullptr;
    initialize();
}

ModulatedDeformableConvPluginDynamic::ModulatedDeformableConvPluginDynamic(const std::string name, const void *data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mOutDim);
    deserialize_value(&data, &length, &mKernelSize);
    deserialize_value(&data, &length, &mNumParamsW);
    deserialize_value(&data, &length, &mNumParamsB);

    deserialize_value(&data, &length, &mStride);
    deserialize_value(&data, &length, &mPadding);
    deserialize_value(&data, &length, &mDilation);
    deserialize_value(&data, &length, &mDeformableGroup);
    deserialize_value(&data, &length, &mGroup);

    size_t wordSize = samplesCommon::getElementSize(mType);

    const char *d = static_cast<const char *>(data);
    
    // mWdev = deserToDev<char>(d, mNumParamsW * wordSize);
    char* w_data = deserToHost<char>(d, mNumParamsW * wordSize);
    mWhost = std::shared_ptr<char>((char *)w_data);
    mW.values = mWhost.get();

    mW.count = mNumParamsW;
    mW.type = mType;

    
    char* b_data = deserToHost<char>(d, mNumParamsB * wordSize);
    mBhost = std::shared_ptr<char>((char *)b_data);
    mB.values = mBhost.get();

    mB.count = mNumParamsB;
    mB.type = mType;
    // mW.values = nullptr;
    mWdev=nullptr;
    mBdev=nullptr;
    initialize();
}

void ModulatedDeformableConvPluginDynamic::setStrideNd(nvinfer1::Dims stride)
{
    mStride = stride;
}

nvinfer1::Dims ModulatedDeformableConvPluginDynamic::getStrideNd() const
{
    return mStride;
}

void ModulatedDeformableConvPluginDynamic::setPaddingNd(nvinfer1::Dims padding)
{
    mPadding = padding;
}

nvinfer1::Dims ModulatedDeformableConvPluginDynamic::getPaddingNd() const
{
    return mPadding;
}

void ModulatedDeformableConvPluginDynamic::setDilationNd(nvinfer1::Dims dilation)
{
    mDilation = dilation;
}

nvinfer1::Dims ModulatedDeformableConvPluginDynamic::getDilationNd() const
{
    return mDilation;
}

void ModulatedDeformableConvPluginDynamic::setDeformableGroup(int deformableGroup)
{
    mDeformableGroup = deformableGroup;
}

int ModulatedDeformableConvPluginDynamic::getDeformableGroup()
{
    return mDeformableGroup;
}

void ModulatedDeformableConvPluginDynamic::setGroup(int group)
{
    mGroup = group;
}

int ModulatedDeformableConvPluginDynamic::getGroup()
{
    return mGroup;
}

nvinfer1::IPluginV2DynamicExt *ModulatedDeformableConvPluginDynamic::clone() const
{
    ModulatedDeformableConvPluginDynamic *plugin = new ModulatedDeformableConvPluginDynamic(mLayerName,
                                                                          mType,
                                                                          mOutDim,
                                                                          mKernelSize,
                                                                          mW,
                                                                          mB);
    plugin->setPluginNamespace(getPluginNamespace());
    plugin->setStrideNd(mStride);
    plugin->setPaddingNd(mPadding);
    plugin->setDilationNd(mDilation);
    plugin->setDeformableGroup(mDeformableGroup);
    plugin->setGroup(mGroup);

    return plugin;
}

inline int convDim(int input_size, int kernel_size, int dilation, int padding, int stride)
{
    return int((input_size - dilation * (kernel_size - 1) + 2 * padding - 1) / float(stride) + 1);
}

nvinfer1::DimsExprs ModulatedDeformableConvPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
{
    assert(nbInputs == 3);
    assert(inputs[0].nbDims == 4);
    assert(inputs[1].nbDims == 4);
    assert(inputs[2].nbDims == 4);
    assert(outputIndex == 0);

    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = exprBuilder.constant(mOutDim);

    ret.d[2] = inputs[1].d[2];
    ret.d[3] = inputs[1].d[3];

    return ret;
}

bool ModulatedDeformableConvPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs)
{
    assert(0 <= pos && pos < 4);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    switch (pos)
    {
    case 0:
        return in[0].type== DataType::kFLOAT && in[0].format == nvinfer1::TensorFormat::kLINEAR;
    case 1:
        return in[1].type == in[0].type &&
               in[1].format == nvinfer1::TensorFormat::kLINEAR;
    case 2:
        return in[2].type == in[0].type &&
               in[2].format == nvinfer1::TensorFormat::kLINEAR;
    case 3:
        return out[0].type == in[0].type &&
               out[0].format == nvinfer1::TensorFormat::kLINEAR;
    }
}

void ModulatedDeformableConvPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(nbInputs == 3);
    assert(mType == inputs[0].desc.type);
    // const auto &inDims0 = inputs[0].desc.dims;
}

size_t ModulatedDeformableConvPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const
{
    /// ALPHA
    int sizeof_dtype = samplesCommon::getElementSize(mType);

    int batch_size = inputs[0].dims.d[0];
    int nInputPlane = inputs[0].dims.d[1];
    int inputHeight = inputs[0].dims.d[2];
    int inputWidth = inputs[0].dims.d[3];

    int nOutputPlane = outputs[0].dims.d[1];
    int outputHeight = outputs[0].dims.d[2];
    int outputWidth = outputs[0].dims.d[3];

    int kW = mKernelSize.d[0];
    int kH = mKernelSize.d[1];
    int im2col_step = std::min(int(batch_size), 64);

    size_t col_size = nInputPlane * kW * kH * outputHeight * outputWidth * sizeof_dtype;

    return col_size + 100 * sizeof(float);
}

int ModulatedDeformableConvPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                                         const void *const *inputs, void *const *outputs, void *workSpace, cudaStream_t stream)
{
    /// ALPHA
    if (m_cuda_stream != stream)
    {
        cublasSetStream(m_cublas_handle, stream);
        m_cuda_stream = stream;
    }
    const static int im2col_step = 64;
    // const size_t workspaceSize = getWorkspaceSize(inputDesc, 2, outputDesc, 1);

    int batch_size = inputDesc[0].dims.d[0];
    int inputChannel = inputDesc[0].dims.d[1];
    int inputHeight = inputDesc[0].dims.d[2];
    int inputWidth = inputDesc[0].dims.d[3];

    DCN_PARAMS dcn_params;
    dcn_params.cublas_handle = m_cublas_handle;
    dcn_params.batchSize = batch_size;
    dcn_params.inputChannel = inputChannel;
    dcn_params.inputW = inputWidth;
    dcn_params.inputH = inputHeight;
    dcn_params.outputChannel = mOutDim;
    dcn_params.kernelW = mKernelSize.d[0];
    dcn_params.kernelH = mKernelSize.d[1];
    dcn_params.strideW = mStride.d[0];
    dcn_params.strideH = mStride.d[1];
    dcn_params.padW = mPadding.d[0];
    dcn_params.padH = mPadding.d[1];
    dcn_params.dilationW = mDilation.d[0];
    dcn_params.dilationH = mDilation.d[1];
    dcn_params.group = mGroup;
    dcn_params.deformable_group = mDeformableGroup;
    dcn_params.im2col_step = std::min(64, batch_size);

    modulated_deform_conv_cuda_forward((float *)inputs[0], (float *)mWdev, (float*)mBdev, 
                                        (float *)inputs[1], (float *)inputs[2],
                                        (float *)outputs[0], workSpace,
                                        dcn_params,
                                        stream);


    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType ModulatedDeformableConvPluginDynamic::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    assert(nbInputs == 3);
    return inputTypes[0];
}

// IPluginV2 Methods
const char *ModulatedDeformableConvPluginDynamic::getPluginType() const
{
    return DCN_NAME;
}

const char *ModulatedDeformableConvPluginDynamic::getPluginVersion() const
{
    return DCN_VERSION;
}

int ModulatedDeformableConvPluginDynamic::getNbOutputs() const
{
    return 1;
}

int ModulatedDeformableConvPluginDynamic::initialize()
{
    cublasCreate(&m_cublas_handle);
    if (mW.values && mWdev == nullptr)
    {
        // target size
        size_t wordSize = samplesCommon::getElementSize(mType);
        size_t nbBytes = mW.count * wordSize;
        CHECK(cudaMalloc((void **)&mWdev, nbBytes));

        if (mType == DataType::kFLOAT)
        {
            convertAndCopyToDevice(mW, reinterpret_cast<float *>(mWdev));
        }
    }else{
        mWdev=nullptr;
    }

    if (mB.values && mBdev == nullptr)
    {
        // target size
        size_t wordSize = samplesCommon::getElementSize(mType);
        size_t nbBytes = mB.count * wordSize;
        CHECK(cudaMalloc((void **)&mBdev, nbBytes));

        if (mType == DataType::kFLOAT)
        {
            convertAndCopyToDevice(mB, reinterpret_cast<float *>(mBdev));
        }
    }else{
        mBdev=nullptr;
    }

    return 0;
}

void ModulatedDeformableConvPluginDynamic::terminate()
{
    gLogVerbose << "DCN Plugin terminate start" << std::endl;

    if(mWdev!=nullptr){
        cudaFree(mWdev);
        mWdev=nullptr;
    }
    if(mBdev!=nullptr){
        cudaFree(mBdev);
        mBdev=nullptr;
    }
        cublasDestroy(m_cublas_handle);

    gLogVerbose << "DCN Plugin terminate done" << std::endl;
}

size_t ModulatedDeformableConvPluginDynamic::getSerializationSize() const
{
    size_t wordSize = samplesCommon::getElementSize(mType);
    return wordSize * mNumParamsW +
            wordSize * mNumParamsB +
           sizeof(mType) +
           sizeof(mOutDim) +
           sizeof(mKernelSize) +
           sizeof(mNumParamsW) +
           sizeof(mNumParamsB) +
           sizeof(mStride) +
           sizeof(mPadding) +
           sizeof(mDilation) +
           sizeof(mDeformableGroup) +
           sizeof(mGroup);
}

void ModulatedDeformableConvPluginDynamic::serialize(void *buffer) const
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mOutDim);
    serialize_value(&buffer, mKernelSize);
    serialize_value(&buffer, mNumParamsW);
    serialize_value(&buffer, mNumParamsB);

    serialize_value(&buffer, mStride);
    serialize_value(&buffer, mPadding);
    serialize_value(&buffer, mDilation);
    serialize_value(&buffer, mDeformableGroup);
    serialize_value(&buffer, mGroup);

    size_t wordSize = samplesCommon::getElementSize(mType);
    char *d = static_cast<char *>(buffer);
    serFromHost(d, mW.values, mNumParamsW * wordSize);
    serFromHost(d, mB.values, mNumParamsB * wordSize);
}

void ModulatedDeformableConvPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void ModulatedDeformableConvPluginDynamic::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *ModulatedDeformableConvPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

ModulatedDeformableConvPluginDynamicCreator::ModulatedDeformableConvPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *ModulatedDeformableConvPluginDynamicCreator::getPluginName() const
{
    return DCN_NAME;
}

const char *ModulatedDeformableConvPluginDynamicCreator::getPluginVersion() const
{
    return DCN_VERSION;
}

const PluginFieldCollection *ModulatedDeformableConvPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2 *ModulatedDeformableConvPluginDynamicCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{
    int outDims = 0;
    int typeId = -1;
    nvinfer1::Dims kernelSize;

    nvinfer1::Dims stride{2, {1, 1}};
    nvinfer1::Dims padding{2, {0, 0}};
    nvinfer1::Dims dilation{2, {1, 1}};
    int deformableGroup = 1;
    int group = 1;

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
        if (field_name.compare("type_id") == 0)
        {
            typeId = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("out_dims") == 0)
        {
            outDims = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("deformable_group") == 0)
        {
            deformableGroup = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("group") == 0)
        {
            group = static_cast<const int *>(fc->fields[i].data)[0];
        }

        if (field_name.compare("kernel_size") == 0)
        {
            kernelSize.nbDims = 2;
            kernelSize.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
            kernelSize.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
        }

        if (field_name.compare("stride") == 0)
        {
            stride.nbDims = 2;
            stride.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
            stride.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
        }

        if (field_name.compare("padding") == 0)
        {
            padding.nbDims = 2;
            padding.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
            padding.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
        }

        if (field_name.compare("dilation") == 0)
        {
            dilation.nbDims = 2;
            dilation.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
            dilation.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
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
            // gLogVerbose << "Building W...\n";
            B.values = fc->fields[i].data;
            B.count = fc->fields[i].length;
            B.type = fieldTypeToDataType(fc->fields[i].type);
        }
    }

    if (outDims <= 0)
    {
        gLogError << "Invalid output dimension" << std::endl;
    }
    if (typeId < 0 || typeId > 3)
    {
        gLogError << "Invalid type id" << typeId << std::endl;
    }
    if (W.count == 0 || W.values == nullptr || W.count < outDims)
    {
        gLogError << "Invalid weights" << std::endl;
    }

    DataType type = static_cast<DataType>(typeId);
    ModulatedDeformableConvPluginDynamic *plugin = new ModulatedDeformableConvPluginDynamic(name, type, outDims, kernelSize, W, B);
    plugin->setPluginNamespace(getPluginNamespace());
    plugin->setStrideNd(stride);
    plugin->setPaddingNd(padding);
    plugin->setDilationNd(dilation);
    plugin->setDeformableGroup(deformableGroup);
    plugin->setGroup(group);

    return plugin;
}

IPluginV2 *ModulatedDeformableConvPluginDynamicCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call FCPluginDynamic::destroy()
    auto plugin = new ModulatedDeformableConvPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void ModulatedDeformableConvPluginDynamicCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *ModulatedDeformableConvPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace amirstan
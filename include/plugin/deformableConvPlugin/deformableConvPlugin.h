#pragma once

#include "NvInferPlugin.h"
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <memory>

namespace amirstan
{
namespace plugin
{

class DeformableConvPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    DeformableConvPluginDynamic(
        const std::string &name, const nvinfer1::DataType &type,
        const int outDim, const nvinfer1::Dims &kernelSize,
        const nvinfer1::Weights &W,
        const nvinfer1::Weights &B);

    DeformableConvPluginDynamic(const std::string name, const void *data, size_t length);

    // It doesn't make sense to make DeformableConvPluginDynamic without arguments, so we
    // delete default constructor.
    DeformableConvPluginDynamic() = delete;

    virtual void setStrideNd(nvinfer1::Dims stride);
    virtual nvinfer1::Dims getStrideNd() const;

    virtual void setPaddingNd(nvinfer1::Dims padding);
    virtual nvinfer1::Dims getPaddingNd() const;

    virtual void setDilationNd(nvinfer1::Dims dilation);
    virtual nvinfer1::Dims getDilationNd() const;

    virtual void setDeformableGroup(int deformableGroup);
    virtual int getDeformableGroup();

    virtual void setGroup(int group);
    virtual int getGroup();

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt *clone() const override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder) override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                            const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const override;
    int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const override;

    // IPluginV2 Methods
    const char *getPluginType() const override;
    const char *getPluginVersion() const override;
    int getNbOutputs() const override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void *buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char *pluginNamespace) override;
    const char *getPluginNamespace() const override;

private:
    const std::string mLayerName;
    std::string mNamespace;

    nvinfer1::DataType mType;
    size_t mOutDim; // leading dim
    nvinfer1::Dims mKernelSize;

    nvinfer1::Dims mStride;
    nvinfer1::Dims mPadding;
    nvinfer1::Dims mDilation;
    int mDeformableGroup;
    int mGroup;

    nvinfer1::Weights mW;
    std::shared_ptr<char> mWhost;
    size_t mNumParamsW;
    char *mWdev;

    nvinfer1::Weights mB;
    std::shared_ptr<char> mBhost;
    size_t mNumParamsB;
    char *mBdev;

    cublasHandle_t m_cublas_handle;
    cudaStream_t m_cuda_stream;

protected:
    // To prevent compiler warnings.
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::enqueue;
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
};

class DeformableConvPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    DeformableConvPluginDynamicCreator();

    const char *getPluginName() const override;

    const char *getPluginVersion() const override;

    const nvinfer1::PluginFieldCollection *getFieldNames() override;

    nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) override;

    nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;

    void setPluginNamespace(const char *pluginNamespace) override;

    const char *getPluginNamespace() const override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(DeformableConvPluginDynamicCreator);
} // namespace plugin
} // namespace amirstan
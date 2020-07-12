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

class GroupNormPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    GroupNormPluginDynamic(
        const std::string &name,
        int num_groups, int num_channels, float eps,
        const nvinfer1::Weights &W, const nvinfer1::Weights &B);

    GroupNormPluginDynamic(const std::string name, const void *data, size_t length);

    // It doesn't make sense to make GroupNormPluginDynamic without arguments, so we
    // delete default constructor.
    GroupNormPluginDynamic() = delete;

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

    int mNumChannels;
    int mNumGroups;
    float mEps;

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

class GroupNormPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    GroupNormPluginDynamicCreator();

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

REGISTER_TENSORRT_PLUGIN(GroupNormPluginDynamicCreator);
} // namespace plugin
} // namespace amirstan
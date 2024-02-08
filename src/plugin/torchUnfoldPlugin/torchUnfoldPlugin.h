#pragma once

#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>

#include "../common/layer_base.h"

namespace amirstan {
namespace plugin {

class TorchUnfoldPluginDynamic: public PluginDynamicBase {
public:
    TorchUnfoldPluginDynamic(const std::string&          name,
                             const std::vector<int32_t>& kernelSize,
                             const std::vector<int32_t>& dilation,
                             const std::vector<int32_t>& padding,
                             const std::vector<int32_t>& stride);

    TorchUnfoldPluginDynamic(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make TorchUnfoldPluginDynamic without arguments,
    // so we delete default constructor.
    TorchUnfoldPluginDynamic() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const PLUGIN_NOEXCEPT override;
    nvinfer1::DimsExprs            getOutputDimensions(int                        outputIndex,
                                                       const nvinfer1::DimsExprs* inputs,
                                                       int                        nbInputs,
                                                       nvinfer1::IExprBuilder&    exprBuilder) PLUGIN_NOEXCEPT override;
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
        PLUGIN_NOEXCEPT override;
    void   configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                           int                                      nbInputs,
                           const nvinfer1::DynamicPluginTensorDesc* out,
                           int                                      nbOutputs) PLUGIN_NOEXCEPT override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                            int                               nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs,
                            int                               nbOutputs) const PLUGIN_NOEXCEPT override;
    int    enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                   const nvinfer1::PluginTensorDesc* outputDesc,
                   const void* const*                inputs,
                   void* const*                      outputs,
                   void*                             workspace,
                   cudaStream_t                      stream) PLUGIN_NOEXCEPT override;

    // IPluginV2Ext Methods
    nvinfer1::DataType
    getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const PLUGIN_NOEXCEPT override;

    // IPluginV2 Methods
    const char* getPluginType() const PLUGIN_NOEXCEPT override;
    const char* getPluginVersion() const PLUGIN_NOEXCEPT override;
    int         getNbOutputs() const PLUGIN_NOEXCEPT override;
    size_t      getSerializationSize() const PLUGIN_NOEXCEPT override;
    void        serialize(void* buffer) const PLUGIN_NOEXCEPT override;

private:
    std::vector<int32_t> mKernelSize;
    std::vector<int32_t> mDilation;
    std::vector<int32_t> mPadding;
    std::vector<int32_t> mStride;
};

class TorchUnfoldPluginDynamicCreator: public PluginCreatorBase<TorchUnfoldPluginDynamic> {
public:
    TorchUnfoldPluginDynamicCreator();

    const char* getPluginName() const PLUGIN_NOEXCEPT override;

    const char* getPluginVersion() const PLUGIN_NOEXCEPT override;

    nvinfer1::IPluginV2* createPlugin(const char*                            name,
                                      const nvinfer1::PluginFieldCollection* fc) PLUGIN_NOEXCEPT override;
};

}  // namespace plugin
}  // namespace amirstan

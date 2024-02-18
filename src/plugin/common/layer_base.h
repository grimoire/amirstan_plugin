#pragma once
#include <string>

#include "NvInferRuntime.h"
#include "NvInferVersion.h"

namespace amirstan {
namespace plugin {

#if NV_TENSORRT_MAJOR > 7
#define PLUGIN_NOEXCEPT noexcept
#else
#define PLUGIN_NOEXCEPT
#endif

class PluginDynamicBase: public nvinfer1::IPluginV2DynamicExt {
public:
    PluginDynamicBase(const std::string& name): mLayerName(name) {}
    // IPluginV2 Methods
    const char* getPluginVersion() const PLUGIN_NOEXCEPT override
    {
        return "1";
    }
    int initialize() PLUGIN_NOEXCEPT override
    {
        return 0;
    }
    void terminate() PLUGIN_NOEXCEPT override {}
    void destroy() PLUGIN_NOEXCEPT override
    {
        delete this;
    }
    void setPluginNamespace(const char* pluginNamespace) PLUGIN_NOEXCEPT override
    {
        mNamespace = pluginNamespace;
    }
    const char* getPluginNamespace() const PLUGIN_NOEXCEPT override
    {
        return mNamespace.c_str();
    }

protected:
    const std::string mLayerName;
    std::string       mNamespace;

#if NV_TENSORRT_MAJOR < 8
protected:
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::enqueue;
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
#endif

    // To prevent compiler warnings.
};

template<class PluginType>
class PluginCreatorBase: public nvinfer1::IPluginCreator {
public:
    const char* getPluginVersion() const PLUGIN_NOEXCEPT override
    {
        return "1";
    };

    const nvinfer1::PluginFieldCollection* getFieldNames() PLUGIN_NOEXCEPT override
    {
        return &mFC;
    }

    void setPluginNamespace(const char* pluginNamespace) PLUGIN_NOEXCEPT override
    {
        mNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const PLUGIN_NOEXCEPT override
    {
        return mNamespace.c_str();
    }

    nvinfer1::IPluginV2DynamicExt*
    deserializePlugin(const char* name, const void* serialData, size_t serialLength) PLUGIN_NOEXCEPT override
    {
        PluginType* plugin = new PluginType(name, serialData, serialLength);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }

protected:
    nvinfer1::PluginFieldCollection    mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string                        mNamespace;
};

#define SET_PLUGIN_NAME_VERSION_IMPL(PluginType, PluginCreator, Name, Version)                                         \
    const char* PluginType::getPluginVersion() const PLUGIN_NOEXCEPT                                                   \
    {                                                                                                                  \
        return Version;                                                                                                \
    }                                                                                                                  \
    const char* PluginType::getPluginType() const PLUGIN_NOEXCEPT                                                      \
    {                                                                                                                  \
        return Name;                                                                                                   \
    }                                                                                                                  \
    const char* PluginCreator::getPluginName() const PLUGIN_NOEXCEPT                                                   \
    {                                                                                                                  \
        return Name;                                                                                                   \
    }                                                                                                                  \
    const char* PluginCreator::getPluginVersion() const PLUGIN_NOEXCEPT                                                \
    {                                                                                                                  \
        return Version;                                                                                                \
    }

#define SET_PLUGIN_NAME_VERSION(PluginType, Version)                                                                   \
    SET_PLUGIN_NAME_VERSION_IMPL(PluginType, PluginType##Creator, #PluginType, Version)

}  // namespace plugin
}  // namespace amirstan
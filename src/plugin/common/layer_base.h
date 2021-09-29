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

class PluginDynamicBase : public nvinfer1::IPluginV2DynamicExt {
 public:
  PluginDynamicBase(const std::string &name) : mLayerName(name) {}
  // IPluginV2 Methods
  const char *getPluginVersion() const PLUGIN_NOEXCEPT override { return "1"; }
  int initialize() PLUGIN_NOEXCEPT override { return 0; }
  void terminate() PLUGIN_NOEXCEPT override {}
  void destroy() PLUGIN_NOEXCEPT override { delete this; }
  void setPluginNamespace(const char *pluginNamespace)
      PLUGIN_NOEXCEPT override {
    mNamespace = pluginNamespace;
  }
  const char *getPluginNamespace() const PLUGIN_NOEXCEPT override {
    return mNamespace.c_str();
  }

 protected:
  const std::string mLayerName;
  std::string mNamespace;

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

class PluginCreatorBase : public nvinfer1::IPluginCreator {
 public:
  const char *getPluginVersion() const PLUGIN_NOEXCEPT override { return "1"; };

  const nvinfer1::PluginFieldCollection *getFieldNames()
      PLUGIN_NOEXCEPT override {
    return &mFC;
  }

  void setPluginNamespace(const char *pluginNamespace)
      PLUGIN_NOEXCEPT override {
    mNamespace = pluginNamespace;
  }

  const char *getPluginNamespace() const PLUGIN_NOEXCEPT override {
    return mNamespace.c_str();
  }

 protected:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};
}  // namespace plugin
}  // namespace amirstan
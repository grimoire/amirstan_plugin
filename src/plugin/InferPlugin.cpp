
#include "plugin/adaptivePoolPlugin/adaptivePoolPlugin.h"
#include "plugin/amirInferPlugin.h"
#include "plugin/batchedNMSPlugin/batchedNMSPlugin.h"
#include "plugin/carafeFeatureReassemblePlugin/carafeFeatureReassemblePlugin.h"
#include "plugin/deformableConvPlugin/deformableConvPlugin.h"
#include "plugin/deformableConvPlugin/modulatedDeformableConvPlugin.h"
#include "plugin/deformablePoolPlugin/deformablePoolPlugin.h"
#include "plugin/delta2bboxPlugin/delta2bboxPlugin.h"
#include "plugin/exViewPlugin/exViewPlugin.h"
#include "plugin/gridAnchorDynamicPlugin/gridAnchorDynamicPlugin.h"
#include "plugin/gridSamplePlugin/gridSamplePlugin.h"
#include "plugin/groupNormPlugin/groupNormPlugin.h"
#include "plugin/layerNormPlugin/layerNormPlugin.h"
#include "plugin/meshGridPlugin/meshGridPlugin.h"
#include "plugin/repeatDimsPlugin/repeatDimsPlugin.h"
#include "plugin/roiExtractorPlugin/roiExtractorPlugin.h"
#include "plugin/roiPoolPlugin/roiPoolPlugin.h"
#include "plugin/torchBmmPlugin/torchBmmPlugin.h"
#include "plugin/torchCumMaxMinPlugin/torchCumMaxMinPlugin.h"
#include "plugin/torchCumPlugin/torchCumPlugin.h"
#include "plugin/torchEmbeddingPlugin/torchEmbeddingPlugin.h"
#include "plugin/torchFlipPlugin/torchFlipPlugin.h"
#include "plugin/torchGatherPlugin/torchGatherPlugin.h"
#include "plugin/torchNMSPlugin/torchNMSPlugin.h"

namespace amirstan {
namespace plugin {
REGISTER_TENSORRT_PLUGIN(BatchedNMSPluginCustomCreator);
REGISTER_TENSORRT_PLUGIN(AdaptivePoolPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(CarafeFeatureReassemblePluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(DeformableConvPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(ModulatedDeformableConvPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(DeformablePoolPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(Delta2BBoxPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(ExViewPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(GridAnchorDynamicPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(GridSamplePluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(GroupNormPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(LayerNormPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(MeshGridPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(RepeatDimsPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(RoiExtractorPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(RoiPoolPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchCumMaxMinPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchCumPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchFlipPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchGatherPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchNMSPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchEmbeddingPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchBmmPluginDynamicCreator);
}  // namespace plugin
}  // namespace amirstan

extern "C" {

bool initLibAmirstanInferPlugins() { return true; }
}  // extern "C"

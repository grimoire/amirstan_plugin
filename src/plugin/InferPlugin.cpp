
#include "adaptivePoolPlugin/adaptivePoolPlugin.h"
#include "batchedNMSPlugin/batchedNMSPlugin.h"
#include "carafeFeatureReassemblePlugin/carafeFeatureReassemblePlugin.h"
#include "deformableConvPlugin/deformableConvPlugin.h"
#include "deformableConvPlugin/modulatedDeformableConvPlugin.h"
#include "deformablePoolPlugin/deformablePoolPlugin.h"
#include "delta2bboxPlugin/delta2bboxPlugin.h"
#include "gridAnchorDynamicPlugin/gridAnchorDynamicPlugin.h"
#include "gridSamplePlugin/gridSamplePlugin.h"
#include "groupNormPlugin/groupNormPlugin.h"
#include "meshGridPlugin/meshGridPlugin.h"
#include "plugin/amirInferPlugin.h"
#include "roiExtractorPlugin/roiExtractorPlugin.h"
#include "roiPoolPlugin/roiPoolPlugin.h"
#include "torchBmmPlugin/torchBmmPlugin.h"
#include "torchCumMaxMinPlugin/torchCumMaxMinPlugin.h"
#include "torchCumPlugin/torchCumPlugin.h"
#include "torchEmbeddingPlugin/torchEmbeddingPlugin.h"
#include "torchGatherPlugin/torchGatherPlugin.h"
#include "torchNMSPlugin/torchNMSPlugin.h"
#include "torchUnfoldPlugin/torchUnfoldPlugin.h"

namespace amirstan {
namespace plugin {
REGISTER_TENSORRT_PLUGIN(AdaptivePoolPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(BatchedNMSPluginCustomCreator);
REGISTER_TENSORRT_PLUGIN(CarafeFeatureReassemblePluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(DeformableConvPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(ModulatedDeformableConvPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(DeformablePoolPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(Delta2BBoxPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(GridAnchorDynamicPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(GridSamplePluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(GroupNormPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(MeshGridPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(RoiExtractorPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(RoiPoolPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchBmmPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchCumMaxMinPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchCumPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchEmbeddingPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchGatherPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchNMSPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchUnfoldPluginDynamicCreator);
}  // namespace plugin
}  // namespace amirstan

extern "C" {

bool initLibAmirstanInferPlugins() { return true; }
}  // extern "C"

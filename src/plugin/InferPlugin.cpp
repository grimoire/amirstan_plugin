
#include "plugin/amirInferPlugin.h"
#include "plugin/deformableConvPlugin/deformableConvPlugin.h"
#include "plugin/deformableConvPlugin/modulatedDeformableConvPlugin.h"
#include "plugin/deformablePoolPlugin/deformablePoolPlugin.h"
#include "plugin/groupNormPlugin/groupNormPlugin.h"
#include "plugin/repeatDimsPlugin/repeatDimsPlugin.h"
#include "plugin/exViewPlugin/exViewPlugin.h"
#include "plugin/layerNormPlugin/layerNormPlugin.h"
#include "plugin/gridAnchorDynamicPlugin/gridAnchorDynamicPlugin.h"
#include "plugin/roiExtractorPlugin/roiExtractorPlugin.h"
#include "plugin/delta2bboxPlugin/delta2bboxPlugin.h"
#include "plugin/batchedNMSPlugin/batchedNMSPlugin.h"
#include "plugin/torchGatherPlugin/torchGatherPlugin.h"
#include "plugin/adaptivePoolPlugin/adaptivePoolPlugin.h"
#include "plugin/meshGridPlugin/meshGridPlugin.h"
#include "plugin/carafeFeatureReassemblePlugin/carafeFeatureReassemblePlugin.h"
#include "plugin/gridSamplePlugin/gridSamplePlugin.h"
#include "plugin/torchFlipPlugin/torchFlipPlugin.h"
#include "plugin/torchCumMaxMinPlugin/torchCumMaxMinPlugin.h"
#include "plugin/torchCumPlugin/torchCumPlugin.h"
#include "plugin/torchNMSPlugin/torchNMSPlugin.h"
#include "plugin/roiPoolPlugin/roiPoolPlugin.h"


extern "C"
{
    
    bool initLibAmirstanInferPlugins(){
        return true;
    }
} // extern "C"

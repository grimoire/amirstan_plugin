#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))


/* This is a sample bounding box parsing function for the mmdetection
 * detector models converted to TensorRT with https://github.com/grimoire/mmdetection-to-tensorrt. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseMmdet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferParseObjectInfo> &objectList)
{
  static int numDetectionLayerIndex = -1;
  static int boxesLayerIndex = -1;
  static int scoresLayerIndex = -1;
  static int classesLayerIndex = -1;
  static NvDsInferDimsCHW scoresLayerDims;
  int numDetsToParse;

  /* Find the num_detections layer */
  if (numDetectionLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "num_detections") == 0) {
        numDetectionLayerIndex = i;
        break;
      }
    }
    if (numDetectionLayerIndex == -1) {
      std::cerr << "Could not find num_detection layer buffer while parsing" << std::endl;
      return false;
    }
  }

  /* Find the boxes layer */
  if (boxesLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "boxes") == 0) {
        boxesLayerIndex = i;
        break;
      }
    }
    if (boxesLayerIndex == -1) {
      std::cerr << "Could not find boxes layer buffer while parsing" << std::endl;
      return false;
    }
  }

  /* Find the scores layer */
  if (scoresLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "scores") == 0) {
        scoresLayerIndex = i;
        getDimsCHWFromDims(scoresLayerDims, outputLayersInfo[i].dims);
        break;
      }
    }
    if (scoresLayerIndex == -1) {
      std::cerr << "Could not find scores layer buffer while parsing" << std::endl;
      return false;
    }
  }

  /* Find the classes layer */
  if (classesLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "classes") == 0) {
        classesLayerIndex = i;
        break;
      }
    }
    if (classesLayerIndex == -1) {
      std::cerr << "Could not find classes layer buffer while parsing" << std::endl;
      return false;
    }
  }

  float *bboxes = (float *) outputLayersInfo[boxesLayerIndex].buffer;
  float *classes = (float *) outputLayersInfo[classesLayerIndex].buffer;
  float *scores = (float *) outputLayersInfo[scoresLayerIndex].buffer;
  numDetsToParse = *(int*)(outputLayersInfo[numDetectionLayerIndex].buffer);

  for (int indx = 0; indx < numDetsToParse; indx++)
  {
    float outputX1 = bboxes[indx * 4];
    float outputY1 = bboxes[indx * 4 + 1];
    float outputX2 = bboxes[indx * 4 + 2];
    float outputY2 = bboxes[indx * 4 + 3];
    float this_class = classes[indx];
    float this_score = scores[indx];
    float threshold = this_class>=0? detectionParams.perClassThreshold[this_class] : 1.;

    if (this_score >= threshold)
    {
      NvDsInferParseObjectInfo object;

      object.classId = this_class;
      object.detectionConfidence = this_score;

      object.left = outputX1;
      object.top = outputY1;
      object.width = outputX2 - outputX1;
      object.height = outputY2 - outputY1;

      objectList.push_back(object);
    }
  }
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseMmdet);

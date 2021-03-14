/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_KERNEL_H
#define TRT_KERNEL_H

#include <cassert>
#include <cstdio>

#include "cublas_v2.h"
#include "plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
#define DEBUG_ENABLE 0

pluginStatus_t allClassNMS(cudaStream_t stream, int num, int num_classes,
                           int num_preds_per_class, int top_k,
                           float nms_threshold, bool share_location,
                           bool isNormalized, DataType DT_SCORE,
                           DataType DT_BBOX, void* bbox_data,
                           void* beforeNMS_scores, void* beforeNMS_index_array,
                           void* afterNMS_scores, void* afterNMS_index_array,
                           bool flipXY = false);

size_t detectionForwardBBoxDataSize(int N, int C1, DataType DT_BBOX);

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int N, int C1,
                                       DataType DT_BBOX);

size_t sortScoresPerClassWorkspaceSize(int num, int num_classes,
                                       int num_preds_per_class,
                                       DataType DT_CONF);

size_t sortScoresPerImageWorkspaceSize(int num_images, int num_items_per_image,
                                       DataType DT_SCORE);

pluginStatus_t sortScoresPerImage(cudaStream_t stream, int num_images,
                                  int num_items_per_image, DataType DT_SCORE,
                                  void* unsorted_scores,
                                  void* unsorted_bbox_indices,
                                  void* sorted_scores,
                                  void* sorted_bbox_indices, void* workspace);

pluginStatus_t sortScoresPerClass(cudaStream_t stream, int num, int num_classes,
                                  int num_preds_per_class,
                                  int background_label_id,
                                  float confidence_threshold, DataType DT_SCORE,
                                  void* conf_scores_gpu, void* index_array_gpu,
                                  void* workspace);

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count);

const char* cublasGetErrorString(cublasStatus_t error);

pluginStatus_t permuteData(cudaStream_t stream, int nthreads, int num_classes,
                           int num_data, int num_dim, DataType DT_DATA,
                           bool confSigmoid, const void* data, void* new_data);

size_t detectionForwardPreNMSSize(int N, int C2);

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK);

pluginStatus_t gatherNMSOutputs(cudaStream_t stream, bool shareLocation,
                                int numImages, int numPredsPerClass,
                                int numClasses, int topK, int keepTopK,
                                DataType DT_BBOX, DataType DT_SCORE,
                                const void* indices, const void* scores,
                                const void* bboxData, void* keepCount,
                                void* nmsedBoxes, void* nmsedScores,
                                void* nmsedClasses, bool clipBoxes = true);

#endif

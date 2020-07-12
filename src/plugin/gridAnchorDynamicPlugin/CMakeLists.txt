cmake_minimum_required(VERSION 2.8.11)

set(TARGET_NAME gridAnchorDynamicPlugin)
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_static)

#find_package(CUDA REQUIRED)
enable_language(CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options -fPIC)

file(GLOB_RECURSE DCN_PLUGIN_HEADS *.h *.hpp *.cuh)
file(GLOB DCN_PLUGIN_SOURCES *.cpp *.cu)

source_group("Include" FILES ${DCN_PLUGIN_HEADS})
source_group("Source" FILES ${DCN_PLUGIN_SOURCES})

cuda_add_library(${STATIC_TARGET} STATIC ${DCN_PLUGIN_HEADS} ${DCN_PLUGIN_SOURCES})
target_link_libraries(${STATIC_TARGET} ${CUDA_LIBRARY})
target_link_libraries(${STATIC_TARGET} ${TENSORRT_LIBRARY})
target_link_libraries(${STATIC_TARGET} amir_cuda_util)
# target_link_libraries(dcn_plugin ${CUDA_npp_LIBRARY})
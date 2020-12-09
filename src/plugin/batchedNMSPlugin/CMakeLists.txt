set(TARGET_NAME batchedNMSPlugin)
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_static)

#find_package(CUDA REQUIRED)
enable_language(CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options -fPIC)

file(GLOB_RECURSE PLUGIN_HEADS *.h *.hpp *.cuh)
file(GLOB PLUGIN_SOURCE *.cpp *.cu)

source_group("Include" FILES ${PLUGIN_HEADS})
source_group("Source" FILES ${PLUGIN_SOURCE})

cuda_add_library(${STATIC_TARGET} STATIC ${PLUGIN_HEADS} ${PLUGIN_SOURCE})
target_link_libraries(${STATIC_TARGET} ${CUDA_LIBRARY})
target_link_libraries(${STATIC_TARGET} ${TENSORRT_LIBRARY})
target_link_libraries(${STATIC_TARGET} amir_cuda_util)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set_property(TARGET ${STATIC_TARGET} PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
endif()

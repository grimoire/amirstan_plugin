cmake_minimum_required(VERSION 2.8.11)

#find_package(CUDA REQUIRED)
enable_language(CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options -fPIC)

file(GLOB_RECURSE AMIR_CUDA_UTIL_HEADS *.h *.hpp *.cuh)
file(GLOB AMIR_CUDA_UTIL_SOURCES *.cpp *.cu)

source_group("Include" FILES ${AMIR_CUDA_UTIL_HEADS})
source_group("Source" FILES ${AMIR_CUDA_UTIL_SOURCES})

cuda_add_library(amir_cuda_util STATIC ${AMIR_CUDA_UTIL_HEADS} ${AMIR_CUDA_UTIL_SOURCES})
target_link_libraries(amir_cuda_util ${CUDA_LIBRARY})
# target_link_libraries(amir_cuda_util ${CUDA_npp_LIBRARY})
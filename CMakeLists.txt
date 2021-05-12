cmake_minimum_required (VERSION 3.10)
project (trt_plugin)


# cuda
FIND_PACKAGE(CUDA REQUIRED)
if(NOT CMAKE_CUDA_DEVICE_LINK_LIBRARY)
  set(CMAKE_CUDA_DEVICE_LINK_LIBRARY
    "<CMAKE_CUDA_COMPILER> ${CMAKE_CUDA_HOST_FLAGS} <LANGUAGE_COMPILE_FLAGS> ${CMAKE_CUDA_COMPILE_OPTIONS_PIC} ${_CMAKE_CUDA_EXTRA_DEVICE_LINK_FLAGS} -shared -dlink <OBJECTS> -o <TARGET> <LINK_LIBRARIES>${__IMPLICT_DLINK_FLAGS}")
endif()
if(NOT CMAKE_CUDA_DEVICE_LINK_EXECUTABLE)
  set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE
    "<CMAKE_CUDA_COMPILER> ${CMAKE_CUDA_HOST_FLAGS} <FLAGS> ${CMAKE_CUDA_COMPILE_OPTIONS_PIC} ${_CMAKE_CUDA_EXTRA_DEVICE_LINK_FLAGS} -shared -dlink <OBJECTS> -o <TARGET> <LINK_LIBRARIES>${__IMPLICT_DLINK_FLAGS}")
endif()
INCLUDE_DIRECTORIES(/usr/local/cuda/include)

# cub
if (NOT DEFINED CUB_ROOT_DIR)
  set(CUB_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cub")
endif()
INCLUDE_DIRECTORIES(${CUB_ROOT_DIR})

# tensorrt
if (NOT DEFINED TENSORRT_DIR)
  set(TENSORRT_DIR "$ENV{HOME}/SDK/TensorRT")
endif()
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_DIR} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PARSERS nvparsers
        HINTS ${TENSORRT_DIR} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_DIR} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} 
                     ${TENSORRT_LIBRARY_PARSERS} 
                     ${TENSORRT_LIBRARY_INFER_PLUGIN} 
                     )
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
    message(ERROR
            "Cannot find TensorRT library.")
endif()
INCLUDE_DIRECTORIES(${TENSORRT_INCLUDE_DIR})

# DeepStream
if (NOT DEFINED WITH_DEEPSTREAM)
    set(WITH_DEEPSTREAM false)
endif ()
MESSAGE(STATUS "WITH_DEEPSTREAM: ${WITH_DEEPSTREAM}")

if (WITH_DEEPSTREAM)
    if (NOT DEFINED DeepStream_DIR)
        set(DeepStream_DIR /opt/nvidia/deepstream/deepstream/)
    endif ()
    if (DEFINED DeepStream_DIR)
        include_directories("${DeepStream_DIR}/sources/includes")
    endif (DEFINED DeepStream_DIR)
endif()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_definitions(-std=c++11)
add_definitions(-fPIC)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fopenmp")

############################################################################################
# CUDA targets

if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
else()
  list(APPEND GPU_ARCHS
      35
      53
      61
      70
      75
    )

  string(REGEX MATCH "aarch64" IS_ARM "${TRT_PLATFORM_ID}")
  if (IS_ARM)
    # Xavier (SM72) only supported for aarch64.
    list(APPEND GPU_ARCHS 72)
  endif()

  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
    # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
    list(APPEND GPU_ARCHS 80)
  else()
    message(WARNING "Detected CUDA version is < 11.0. SM80 not supported.")
  endif()

  message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif()
# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")

############################################################################################

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


add_subdirectory (src)

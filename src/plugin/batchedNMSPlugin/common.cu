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
#include <stdint.h>

#include <cub/cub.cuh>

#include "bboxUtils.h"
#include "cublas_v2.h"
#include "kernel.h"

#define CUDA_MEM_ALIGN 256

// ALIGNPTR
int8_t* alignPtr(int8_t* ptr, uintptr_t to)
{
    uintptr_t addr = (uintptr_t)ptr;
    if (addr % to) {
        addr += to - addr % to;
    }
    return (int8_t*)addr;
}

// NEXTWORKSPACEPTR
int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t)ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t*)addr, CUDA_MEM_ALIGN);
}

// CALCULATE TOTAL WORKSPACE SIZE
size_t calculateTotalWorkspaceSize(size_t* workspaces, int count)
{
    size_t total = 0;
    for (int i = 0; i < count; i++) {
        total += workspaces[i];
        if (workspaces[i] % CUDA_MEM_ALIGN) {
            total += CUDA_MEM_ALIGN - (workspaces[i] % CUDA_MEM_ALIGN);
        }
    }
    return total;
}

using nvinfer1::DataType;

// DATA TYPE SIZE
size_t dataTypeSize(const DataType dtype)
{
    switch (dtype) {
        case DataType::kINT8:
            return sizeof(char);
        case DataType::kHALF:
            return sizeof(short);
        case DataType::kFLOAT:
            return sizeof(float);
        default:
            return 0;
    }
}

template<unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void setUniformOffsets_kernel(const int num_segments, const int offset, int* d_offsets)
{
    const int idx = blockIdx.x * nthds_per_cta + threadIdx.x;
    if (idx <= num_segments)
        d_offsets[idx] = idx * offset;
}

void setUniformOffsets(cudaStream_t stream, const int num_segments, const int offset, int* d_offsets)
{
    const int BS = 32;
    const int GS = (num_segments + 1 + BS - 1) / BS;
    setUniformOffsets_kernel<BS><<<GS, BS, 0, stream>>>(num_segments, offset, d_offsets);
}

const char* cublasGetErrorString(cublasStatus_t error)
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        default:
            return "UNKNOW_ERROR";
    }
    return "Unknown cublas status";
}

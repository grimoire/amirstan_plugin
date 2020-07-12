#pragma once

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "cublas_v2.h"
#include "logging.h"
#include "common.h"

extern Logger gLogger;
extern LogStreamConsumer gLogVerbose;
extern LogStreamConsumer gLogInfo;
extern LogStreamConsumer gLogWarning;
extern LogStreamConsumer gLogError;
extern LogStreamConsumer gLogFatal;

namespace amirstan
{

constexpr uint32_t BDIM = 1; // batch dimension
constexpr uint32_t SDIM = 0; // seq len dimension
constexpr uint32_t HDIM = 2; // hidden dimension

inline void convertAndCopyToDevice(const nvinfer1::Weights &src, float *destDev)
{

    size_t wordSize = sizeof(float);
    size_t nbBytes = src.count * wordSize;
    if (src.type == nvinfer1::DataType::kFLOAT)
    {
        gLogVerbose << "Float Weights(Host) => Float Array(Device)" << std::endl;
        CHECK(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
    }
    else
    {
        gLogVerbose << "Half Weights(Host) => Float Array(Device)" << std::endl;
        std::vector<float> tmp(src.count);
        const half *values = reinterpret_cast<const half *>(src.values);

        for (int it = 0; it < tmp.size(); it++)
        {
            tmp[it] = __half2float(values[it]);
        }

        CHECK(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
    }
}

template <typename T>
inline T *deserToDev(const char *&buffer, size_t nbElem)
{
    T *dev = nullptr;
    const size_t len = sizeof(T) * nbElem;
    CHECK(cudaMalloc((void **)&dev, len));
    CHECK(cudaMemcpy(dev, buffer, len, cudaMemcpyHostToDevice));

    buffer += len;
    return dev;
}

template <typename T>
inline void serFromDev(char *&buffer, const T *data, size_t nbElem)
{
    const size_t len = sizeof(T) * nbElem;
    CHECK(cudaMemcpy(buffer, data, len, cudaMemcpyDeviceToHost));
    buffer += len;
}

template <typename T>
inline T *deserToHost(const char *&buffer, size_t nbElem)
{
    T *dev = nullptr;
    const size_t len = sizeof(T) * nbElem;
    dev = (char*)malloc(len);
    memcpy(dev, buffer, len);

    buffer += len;
    return dev;
}

template <typename T>
inline void serFromHost(char *&buffer, const T *data, size_t nbElem)
{
    const size_t len = sizeof(T) * nbElem;
    memcpy(buffer, data, len);
    buffer += len;
}

inline nvinfer1::DataType fieldTypeToDataType(const nvinfer1::PluginFieldType ftype)
{
    switch (ftype)
    {
    case nvinfer1::PluginFieldType::kFLOAT32:
    {
        gLogVerbose << "PluginFieldType is Float32" << std::endl;
        return nvinfer1::DataType::kFLOAT;
    }
    case nvinfer1::PluginFieldType::kFLOAT16:
    {
        gLogVerbose << "PluginFieldType is Float16" << std::endl;
        return nvinfer1::DataType::kHALF;
    }
    case nvinfer1::PluginFieldType::kINT32:
    {
        gLogVerbose << "PluginFieldType is Int32" << std::endl;
        return nvinfer1::DataType::kINT32;
    }
    case nvinfer1::PluginFieldType::kINT8:
    {
        gLogVerbose << "PluginFieldType is Int8" << std::endl;
        return nvinfer1::DataType::kINT8;
    }
    default:
        throw std::invalid_argument("No corresponding datatype for plugin field type");
    }
}
} // namespace amirstan
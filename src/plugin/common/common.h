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

#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H

// For loadLibrary
#ifdef _MSC_VER
// Needed so that the max/min definitions in windows.h do not conflict with
// std::max/min.
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <dlfcn.h>
#endif

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
using namespace plugin;

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#if (!defined(__ANDROID__) && defined(__aarch64__)) || defined(__QNX__)
#define ENABLE_DLA_API 1
#endif

#define CHECK(status)                                    \
  do {                                                   \
    auto ret = (status);                                 \
    if (ret != 0) {                                      \
      std::cerr << "Cuda failure: " << ret << std::endl; \
      abort();                                           \
    }                                                    \
  } while (0)

#define CHECK_RETURN_W_MSG(status, val, errMsg)                        \
  do {                                                                 \
    if (!(status)) {                                                   \
      std::cerr << errMsg << " Error in " << __FILE__ << ", function " \
                << FN_NAME << "(), line " << __LINE__ << std::endl;    \
      return val;                                                      \
    }                                                                  \
  } while (0)

#define CHECK_RETURN(status, val) CHECK_RETURN_W_MSG(status, val, "")

constexpr long double operator"" _GiB(long double val) {
  return val * (1 << 30);
}
constexpr long double operator"" _MiB(long double val) {
  return val * (1 << 20);
}
constexpr long double operator"" _KiB(long double val) {
  return val * (1 << 10);
}

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val) {
  return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val) {
  return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val) {
  return val * (1 << 10);
}

namespace samplesCommon {

inline void* safeCudaMalloc(size_t memSize) {
  void* deviceMem;
  CHECK(cudaMalloc(&deviceMem, memSize));
  if (deviceMem == nullptr) {
    std::cerr << "Out of memory" << std::endl;
    exit(1);
  }
  return deviceMem;
}

inline bool isDebug() { return (std::getenv("TENSORRT_DEBUG") ? true : false); }

template <class Iter>
inline std::vector<size_t> argsort(Iter begin, Iter end, bool reverse = false) {
  std::vector<size_t> inds(end - begin);
  std::iota(inds.begin(), inds.end(), 0);
  if (reverse) {
    std::sort(inds.begin(), inds.end(),
              [&begin](size_t i1, size_t i2) { return begin[i2] < begin[i1]; });
  } else {
    std::sort(inds.begin(), inds.end(),
              [&begin](size_t i1, size_t i2) { return begin[i1] < begin[i2]; });
  }
  return inds;
}

inline void print_version() {
  std::cout << "  TensorRT version: " << NV_TENSORRT_MAJOR << "."
            << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << "."
            << NV_TENSORRT_BUILD << std::endl;
}

inline float getMaxValue(const float* buffer, int64_t size) {
  assert(buffer != nullptr);
  assert(size > 0);
  return *std::max_element(buffer, buffer + size);
}

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    // case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

inline int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

class TimerBase {
 public:
  virtual void start() {}
  virtual void stop() {}
  float microseconds() const noexcept { return mMs * 1000.f; }
  float milliseconds() const noexcept { return mMs; }
  float seconds() const noexcept { return mMs / 1000.f; }
  void reset() noexcept { mMs = 0.f; }

 protected:
  float mMs{0.0f};
};

class GpuTimer : public TimerBase {
 public:
  GpuTimer(cudaStream_t stream) : mStream(stream) {
    CHECK(cudaEventCreate(&mStart));
    CHECK(cudaEventCreate(&mStop));
  }
  ~GpuTimer() {
    CHECK(cudaEventDestroy(mStart));
    CHECK(cudaEventDestroy(mStop));
  }
  void start() { CHECK(cudaEventRecord(mStart, mStream)); }
  void stop() {
    CHECK(cudaEventRecord(mStop, mStream));
    float ms{0.0f};
    CHECK(cudaEventSynchronize(mStop));
    CHECK(cudaEventElapsedTime(&ms, mStart, mStop));
    mMs += ms;
  }

 private:
  cudaEvent_t mStart, mStop;
  cudaStream_t mStream;
};  // class GpuTimer

template <typename Clock>
class CpuTimer : public TimerBase {
 public:
  using clock_type = Clock;

  void start() { mStart = Clock::now(); }
  void stop() {
    mStop = Clock::now();
    mMs += std::chrono::duration<float, std::milli>{mStop - mStart}.count();
  }

 private:
  std::chrono::time_point<Clock> mStart, mStop;
};  // class CpuTimer

using PreciseCpuTimer = CpuTimer<std::chrono::high_resolution_clock>;

inline void loadLibrary(const std::string& path) {
#ifdef _MSC_VER
  void* handle = LoadLibrary(path.c_str());
#else
  void* handle = dlopen(path.c_str(), RTLD_LAZY);
#endif
  if (handle == nullptr) {
#ifdef _MSC_VER
    std::cerr << "Could not load plugin library: " << path << std::endl;
#else
    std::cerr << "Could not load plugin library: " << path
              << ", due to: " << dlerror() << std::endl;
#endif
  }
}

}  // namespace samplesCommon

inline std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims) {
  os << "(";
  for (int i = 0; i < dims.nbDims; ++i) {
    os << (i ? ", " : "") << dims.d[i];
  }
  return os << ")";
}

#endif  // TENSORRT_COMMON_H

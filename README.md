# Amirstan_plugin

Amirstan plugin contains some useful TensorRT plugins.
These plugins are used to support other projects:

- [torch2trt_dynamic](https://github.com/grimoire/torch2trt_dynamic)

- [mmdetection-to-tensorrt](https://github.com/grimoire/mmdetection-to-tensorrt)

## Requirement

- TensorRT >= 8.0

## Installation

- Install TensorRT: [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)

### From sources

- clone the repo and create `build` folder

  ```bash
  git clone --depth=1 https://github.com/grimoire/amirstan_plugin.git
  cd amirstan_plugin
  git submodule update --init --progress --depth=1
  mkdir build
  cd build
  ```

- build the project with:

  - plugins only

    ```bash
    cmake -DTENSORRT_DIR=${path_to_tensorrt} ..
    make -j$(nproc)
    ```

  - or with DeepStream support

    ```bash
    cmake -DTENSORRT_DIR=${path_to_tensorrt} -DWITH_DEEPSTREAM=true -DDeepStream_DIR=${path_to_deepstream} ..
    make -j$(nproc)
    ```

- set the envoirment variable(in ~/.bashrc):

```bash
export AMIRSTAN_LIBRARY_PATH=<amirstan_plugin_root>/build/lib
```

### Using Conan

- Install [Conan](https://conan.io/):

  ```bash
  pip install conan
  ```

- Register grimoire's Conan remote:

  ```bash
  conan remote add grimoire https://grimoire.jfrog.io/artifactory/api/conan/grimoire-conan
  ```

- Add a `conanfile.txt` file to your project's root with the following content:

  ```txt
  [requires]
  amirstan_plugin/0.5.0

  [generators]
  cmake
  ```

- Additionaly, you can add a few options under the \[options\] section to configure your build:

  - tensorrt_dir: path where TensorRT is located. Default `~/SDK/TensorRT`.
  - with_deepstream: whether to compile with deepstream support. Default `False`.
  - deepstream_dir: path where deepstream is located. Default `/opt/nvidia/deepstream/deepstream`
  - cub_root_dir: Default `./third_party/cub`
  - cuda_arch: list of CUDA architectures to compile for. Default `61;62;70;72;75;80;86`

  For example, to use a custom TensorRT dir and compile for a specific CUDA architecture:

  ```txt
  [requires]
  amirstan_plugin/0.5.0

  [generators]
  cmake

  [options]
  amirstan_plugin:tensorrt_dir=/usr/include/x86_64-linux-gnu
  amirstan_plugin:cuda_arch=75
  ```

- Add the following lines to your project root's `CMakeLists.txt`:

  ```cmake
  INCLUDE(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
  CONAN_BASIC_SETUP()
  ```

- Add conan libs to the linking stage:

  ```cmake
  target_link_libraries(trt_sample PUBLIC ${CONAN_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
  ```

- Compile your project:

  ```bash
  mkdir build
  cd build
  conan install .. -s compiler.libcxx=libstdc++11 --build=missing 
  cmake .. 
  make -j$(nproc)
  ```

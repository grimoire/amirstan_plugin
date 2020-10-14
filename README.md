# Amirstan_plugin

Amirstan plugin contain some useful tensorrt plugin.
These plugins are used to support some other project such as 

https://github.com/grimoire/torch2trt_dynamic 

https://github.com/grimoire/mmdetection-to-tensorrt


## Requirement

- Tensorrt >= 7.0.0.11

## Installation

- Install tensorrt7: https://developer.nvidia.com/tensorrt

clone the repo and create build folder

```shell
git clone --depth=1 https://github.com/grimoire/amirstan_plugin.git
cd amirstan_plugin
git submodule update --init --progress --depth=1
mkdir build
cd build
```

either

- cmake base version
  ```shell
  cmake -DTENSORRT_DIR=<path_to_tensorrt> ..
  ```

- or with deepstream support
  ```shell
  cmake -DTENSORRT_DIR=<path_to_tensorrt> -DWITH_DEEPSTREAM=true -DDeepStream_DIR=<path_to_deepstream> ..
  ```

make the plugins

```shell
make -j10
```

set the envoirment variable(in ~/.bashrc):

```shell
export AMIRSTAN_LIBRARY_PATH=<amirstan_plugin_root>/build/lib
```


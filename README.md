# Amirstan_plugin

Amirstan plugin contain some useful tensorrt plugin.
These plugins are used to support some other project such as 

https://github.com/grimoire/torch2trt_dynamic 

https://github.com/grimoire/mmdetection-to-tensorrt


## Requirement

- Tensorrt >= 7.0.0.11
- cub >= 1.8.0

## Installation

- Install cub: https://nvlabs.github.io/cub/
- Install tensorrt7: https://developer.nvidia.com/tensorrt

```shell
git clone https://github.com/grimoire/amirstan_plugin.git
cd amirstan_plugin
mkdir build
cd build
cmake -DCUB_ROOT_DIR=<path_to_cub> -DTENSORRT_DIR=<path_to_tensorrt> ..
make -j10
```

set the envoirment variable(in ~/.bashrc):

```shell
export AMIRSTAN_LIBRARY_PATH=<amirstan_plugin_root>/build/lib
```


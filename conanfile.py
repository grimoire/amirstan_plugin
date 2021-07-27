from conans import ConanFile, CMake


class AmirstanPluginConan(ConanFile):
    name = "amirstan_plugin"
    version = "0.4.1"
    license = "MIT"

    url = "https://github.com/grimoire/amirstan_plugin.git"
    repo_url = "https://github.com/grimoire/amirstan_plugin.git"
    description = "Amirstan plugin contain some useful tensorrt plugin."
    topics = ("tensorrt", "mmdetection")
    settings = "os", "compiler", "build_type", "arch"
    options = { 
        "shared": [True, False],
        "tensorrt_dir": "ANY",
        "with_deepstream": [True, False],
        "deepstream_dir": "ANY",
        "cub_root_dir": "ANY",
        "cuda_arch": "ANY"
    }
    default_options = { 
        "shared": True,
        "tensorrt_dir": None,
        "with_deepstream": False,
        "deepstream_dir": None,
        "cub_root_dir": None,
        "cuda_arch": "61;62;70;72;75;80;86"
    }
    generators = "cmake"
    exports_sources = "src*", "include*", "lib*", "CMakeLists.txt", "cmake*"

    def configure(self):
        self.settings.compiler.libcxx = "libstdc++11"

    def build(self):
        cmake = CMake(self)

        cmake.definitions["WITH_DEEPSTREAM"] = self.options.with_deepstream
        cmake.definitions['GPU_ARCHS'] = self.options.cuda_arch

        if self.options.tensorrt_dir is not None:
            cmake.definitions["TENSORRT_DIR"] = self.options.tensorrt_dir

        if self.options.deepstream_dir is not None and self.options.with_deepstream:
            cmake.definitions["DeepStream_DIR"] = self.options.deepstream_dir
        
        if self.options.cub_root_dir is not None:
            cmake.definitions["CUB_ROOT_DIR"] = self.options.cub_root_dir

        cmake.configure(source_folder=".")
        cmake.build()

    def package(self):
        self.copy("*.h", dst="include/plugin", src="include/plugin")
        self.copy("*.h", dst="include/amir_cuda_util", src="include/amir_cuda_util")
        self.copy("*.lib", dst="lib", keep_path=False)
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.dylib*", dst="lib", keep_path=False)
        self.copy("*.so*", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)
    
    def package_info(self):
        self.cpp_info.libs = ["amirstan_plugin"]

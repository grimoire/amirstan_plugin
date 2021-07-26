from conans import ConanFile, CMake


class AmirstanConan(ConanFile):
    name = "amirstan"
    version = "0.4.1"
    license = "MIT"

    url = "https://github.com/grimoire/amirstan_plugin.git"
    repo_url = "https://github.com/grimoire/amirstan_plugin.git"
    description = "Amirstan plugin contain some useful tensorrt plugin."
    topics = ("tensorrt", "mmdetection")
    settings = "os", "compiler", "build_type", "arch"
    options = { 
        "shared": [True, False],
        "cuda_arch": ["61;62;70;72;75;80;86","61","62","70","72","75","80","86"]
    }
    default_options = { "shared": True,
                        "cuda_arch": "61;62;70;72;75;80;86"}
    generators = "cmake"
    exports_sources = "src*", "include*", "lib*", "CMakeLists.txt", "cmake*"

    def configure(self):
        self.settings.compiler.libcxx = "libstdc++11"

    def build(self):
        cmake = CMake(self)
        cmake.definitions['GPU_ARCHS'] = self.options.cuda_arch
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

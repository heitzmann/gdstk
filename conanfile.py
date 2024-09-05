from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


class gdstkRecipe(ConanFile):
    name = "gdstk"
    version = "0.9.55"
    package_type = "library"

    # Optional metadata
    license = "Boost"
    author = "gdstk colab (conanfile by vinci4dai)"
    url = "https://github.com/Vinci4d/gdstk"
    description = "oasis and gds file parsing"
    topics = ("gds", "oasis")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*", "include/*", "docs/cpp/*", "external/*"

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("qhull/8.0.1")
        self.requires("zlib/1.3.1")
        
    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["gdstk", "clipper"]


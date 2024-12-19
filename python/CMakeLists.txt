include(GNUInstallDirs)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

python_add_library(_gdstk MODULE WITH_SOABI "gdstk_module.cpp")

target_compile_features(_gdstk PRIVATE cxx_std_11)

target_link_libraries(_gdstk PRIVATE gdstk Python::NumPy)

install(TARGETS _gdstk
    COMPONENT _gdstk
    DESTINATION ${SKBUILD_PROJECT_NAME})


set(TARGET llama-kernel-bench)
add_executable(${TARGET}
    kernel-bench.cpp
    kernel-registry.cpp
    bench-harness.cpp
    kernels/kernels-standard.cpp
    kernels/kernels-custom.cpp
)
target_link_libraries(${TARGET} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# Inference benchmark — hooks into ggml's compute path for kernel swapping
set(TARGET2 llama-inference-bench)
add_executable(${TARGET2}
    inference-bench.cpp
    kernels/kernels-standard.cpp
    kernels/kernels-custom.cpp
)
target_link_libraries(${TARGET2} PRIVATE common llama ggml ggml-cpu ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET2} PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu
)
target_compile_features(${TARGET2} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET2} RUNTIME)
endif()

# Backend-level benchmark: registers a sample custom ggml backend and
# compares it directly against the standard CPU backend
set(TARGET3 llama-backend-bench)
add_executable(${TARGET3}
    backend-bench.cpp
    custom-backend.cpp
)
target_link_libraries(${TARGET3} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET3} PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
)
target_compile_features(${TARGET3} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET3} RUNTIME)
endif()

# Optional SVE plug-point source. Enabled only if the compiler accepts
# -march=armv8-a+sve. The TU itself has an __ARM_FEATURE_SVE guard so
# dropping the flag silently falls back to an empty namespace -- this
# block only controls whether the file is added to the build graph and
# whether the SVE march flag reaches its compile line.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=armv8-a+sve" KB_HAVE_SVE_COMPILER)
if (KB_HAVE_SVE_COMPILER)
    target_sources(${TARGET}  PRIVATE kernels/kernels-sve.cpp)
    target_sources(${TARGET2} PRIVATE kernels/kernels-sve.cpp)
    set_source_files_properties(kernels/kernels-sve.cpp
        PROPERTIES COMPILE_OPTIONS "-march=armv8-a+sve")
    target_compile_definitions(${TARGET}  PRIVATE KB_HAVE_SVE=1)
    target_compile_definitions(${TARGET2} PRIVATE KB_HAVE_SVE=1)
endif()

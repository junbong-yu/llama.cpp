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

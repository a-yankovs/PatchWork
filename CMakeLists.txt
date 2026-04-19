cmake_minimum_required(VERSION 3.18)
project(tinyvla_debugger VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build type defaults
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
add_compile_options(
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Debug>:-g -O0>
)

# AMD ROCm / HIP optional support
option(TINYVLA_USE_ROCM "Enable AMD ROCm GPU acceleration" OFF)
if(TINYVLA_USE_ROCM)
    find_package(hip QUIET)
    if(hip_FOUND)
        message(STATUS "AMD ROCm HIP found — GPU acceleration enabled")
        add_compile_definitions(TINYVLA_ROCM_ENABLED)
    else()
        message(WARNING "ROCm requested but HIP not found — building CPU-only")
    endif()
endif()

add_subdirectory(cpp)

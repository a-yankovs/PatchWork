# cpp/CMakeLists.txt — C++ motion stack + pybind11 module

# ---- Find pybind11 ----------------------------------------------------------
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Install: pip install pybind11")
    endif()
endif()

# ---- Core C++ library (no Python dependency) --------------------------------
add_library(tinyvla_core STATIC
    src/common.cpp
    src/kinematics.cpp
    src/trajectory.cpp
    src/motion_planner.cpp
    src/controller.cpp
    src/robot_interface.cpp
)

target_include_directories(tinyvla_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(tinyvla_core PUBLIC cxx_std_17)

# ---- pybind11 extension module ----------------------------------------------
pybind11_add_module(tinyvla_cpp
    bindings/pybind_module.cpp
    bindings/bind_kinematics.cpp
    bindings/bind_motion.cpp
    bindings/bind_robot.cpp
)

target_link_libraries(tinyvla_cpp PRIVATE tinyvla_core)

target_include_directories(tinyvla_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Install the .so next to the Python package so it's importable
install(TARGETS tinyvla_cpp
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/python/tinyvla_debugger
)

# ---- Optional C++ unit tests ------------------------------------------------
option(TINYVLA_BUILD_TESTS "Build C++ unit tests" OFF)
if(TINYVLA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests/cpp)
endif()

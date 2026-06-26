cmake_minimum_required(VERSION 3.20)
project(diffeq_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Complier optimizaer flags, can use -01, -02, -03 for more optimization but less stability. 
set(CMAKE_CXX_FLAGS_RELEASE "-O1 -march=native -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O1 -march=native -DNDEBUG")

set(PYTHON_VENV_PATH "${CMAKE_SOURCE_DIR}/.venv")
set(Python3_ROOT_DIR "${PYTHON_VENV_PATH}")
set(Python3_FIND_VIRTUALENV ONLY)
set(Python3_FIND_STRATEGY LOCATION)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

message(STATUS "🔹 Using Python: ${Python3_EXECUTABLE}")
message(STATUS "🔹 Python include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "🔹 NumPy include dir: ${Python3_NumPy_INCLUDE_DIRS}")
message(STATUS "🔹 CMake build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "🔹 CXX release flags: ${CMAKE_CXX_FLAGS_RELEASE}")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include_directories(include)

add_executable(diffeq_demo
    src/main.cpp
    src/solver.cpp
    src/Steppers/euler_stepper.cpp
)

add_executable(1d_heat_example
    examples/heat_1d.cpp
    src/solver.cpp
    src/pde/heat_equation.cpp
    src/Steppers/rk4_stepper.cpp
)

target_include_directories(diffeq_demo PRIVATE include)
target_include_directories(1d_heat_example PRIVATE include)

pybind11_add_module(deq
    src/bindings.cpp
    src/solver.cpp
    src/Steppers/euler_stepper.cpp
    src/Steppers/midpoint_stepper.cpp
    src/Steppers/rk3_stepper.cpp
    src/Steppers/rk4_stepper.cpp
    src/Steppers/rk5_stepper.cpp
    src/pde/heat_equation.cpp
)

target_include_directories(deq PRIVATE
    include
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(deq PRIVATE Python3::Python)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import sysconfig; print(sysconfig.get_path('platlib'))"
    OUTPUT_VARIABLE PYTHON_SITEARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "🔹 Python site-packages (platlib): ${PYTHON_SITEARCH}")

install(TARGETS deq
    LIBRARY DESTINATION ${PYTHON_SITEARCH}
)
cmake_minimum_required(VERSION 3.14)
project(SemiImplicitFV VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build unit tests" OFF)
option(ENABLE_OPENMP "Enable OpenMP parallelization" OFF)
option(ENABLE_MPI "Enable MPI parallelization" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
endif()

# OpenMP
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# MPI
if(ENABLE_MPI)
    find_package(MPI REQUIRED)
endif()

# Library sources
set(LIB_SOURCES
    src/SolutionState.cpp
    src/RectilinearMesh.cpp
    src/Reconstruction.cpp
    src/LFSolver.cpp
    src/RusanovSolver.cpp
    src/HLLCSolver.cpp
    src/IGR.cpp
    src/SemiImplicitSolver.cpp
    src/ExplicitSolver.cpp
    src/RKTimeStepping.cpp
    src/PressureLaplacian.cpp
    src/GaussSeidelPressureSolver.cpp
    src/JacobiPressureSolver.cpp
    src/StiffenedGasEOS.cpp
    src/IdealGasEOS.cpp
    src/VTKWriter.cpp
    src/MPIContext.cpp
    src/HaloExchange.cpp
    src/Runtime.cpp
    src/VTKSession.cpp
)

# Create library
add_library(SemiImplicitFV ${LIB_SOURCES})
target_include_directories(SemiImplicitFV PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(ENABLE_OPENMP)
    target_link_libraries(SemiImplicitFV PUBLIC OpenMP::OpenMP_CXX)
endif()

if(ENABLE_MPI)
    target_link_libraries(SemiImplicitFV PUBLIC MPI::MPI_CXX)
    target_compile_definitions(SemiImplicitFV PUBLIC ENABLE_MPI)
endif()

# Examples — auto-discover every examples/<case>/*.cpp
if(BUILD_EXAMPLES)
    file(GLOB EXAMPLE_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/examples
         ${CMAKE_CURRENT_SOURCE_DIR}/examples/*)
    foreach(CASE_DIR ${EXAMPLE_DIRS})
        if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/${CASE_DIR})
            file(GLOB CASE_SOURCES examples/${CASE_DIR}/*.cpp)
            if(CASE_SOURCES)
                add_executable(${CASE_DIR} ${CASE_SOURCES})
                target_link_libraries(${CASE_DIR} PRIVATE SemiImplicitFV)
            endif()
        endif()
    endforeach()
endif()

# Install
install(TARGETS SemiImplicitFV
    EXPORT SemiImplicitFVTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

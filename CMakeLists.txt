cmake_minimum_required(VERSION 3.14)
project(SemiImplicitFV VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build unit tests" OFF)
option(ENABLE_OPENMP "Enable OpenMP parallelization" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native)
    endif()
endif()

# OpenMP
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# Library sources
set(LIB_SOURCES
    src/RectilinearMesh.cpp
    src/RiemannSolver.cpp
    src/IGR.cpp
    src/SemiImplicitSolver.cpp
    src/EquationOfState.cpp
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

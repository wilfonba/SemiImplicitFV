cmake_minimum_required(VERSION 3.14)
project(SemiImplicitFV VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_DRIVER "Build generic sifv driver executable" ON)
option(ENABLE_PETSC "Enable PETSc pressure solver (KSP+GAMG)" OFF)

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

# MPI
find_package(MPI REQUIRED)

# PETSc (optional, built from source via ExternalProject)
if(ENABLE_PETSC)
    include(ExternalProject)

    set(PETSC_INSTALL_DIR ${CMAKE_BINARY_DIR}/petsc-install)

    # Find MPI compiler wrappers for PETSc configure
    find_program(MPI_CC_WRAPPER NAMES mpicc HINTS ENV PATH)
    find_program(MPI_CXX_WRAPPER NAMES mpicxx mpic++ mpiCC HINTS ENV PATH)
    if(NOT MPI_CC_WRAPPER)
        set(MPI_CC_WRAPPER "mpicc")
    endif()
    if(NOT MPI_CXX_WRAPPER)
        set(MPI_CXX_WRAPPER "mpicxx")
    endif()

    ExternalProject_Add(petsc_external
        GIT_REPOSITORY https://gitlab.com/petsc/petsc.git
        GIT_TAG v3.22.2
        GIT_SHALLOW TRUE
        PREFIX ${CMAKE_BINARY_DIR}/petsc-prefix
        CONFIGURE_COMMAND ${CMAKE_BINARY_DIR}/petsc-prefix/src/petsc_external/configure
            --prefix=${PETSC_INSTALL_DIR}
            --with-cc=${MPI_CC_WRAPPER}
            --with-cxx=${MPI_CXX_WRAPPER}
            --with-fc=0
            --with-x=0
            --with-debugging=0
            --with-shared-libraries=0
            --download-f2cblaslapack=yes
            "COPTFLAGS=-O3"
            "CXXOPTFLAGS=-O3"
        BUILD_COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS
            make PETSC_DIR=<SOURCE_DIR> PETSC_ARCH=arch-linux-c-opt all
        INSTALL_COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS
            make PETSC_DIR=<SOURCE_DIR> PETSC_ARCH=arch-linux-c-opt install
        BUILD_IN_SOURCE TRUE
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_INSTALL TRUE
    )

    # Create the install directory so CMake doesn't complain about non-existent paths
    # at configure time (PETSc is built later during the build step)
    file(MAKE_DIRECTORY ${PETSC_INSTALL_DIR}/include)

    # Create imported target for PETSc
    add_library(PETSc::PETSc STATIC IMPORTED)
    set_target_properties(PETSc::PETSc PROPERTIES
        IMPORTED_LOCATION ${PETSC_INSTALL_DIR}/lib/libpetsc.a
        INTERFACE_INCLUDE_DIRECTORIES ${PETSC_INSTALL_DIR}/include
    )

    # PETSc static lib dependencies
    find_library(DL_LIBRARY dl)
    find_library(M_LIBRARY m)
    find_library(RT_LIBRARY rt)
    set(PETSC_EXTRA_LIBS "")
    if(DL_LIBRARY)
        list(APPEND PETSC_EXTRA_LIBS ${DL_LIBRARY})
    endif()
    if(M_LIBRARY)
        list(APPEND PETSC_EXTRA_LIBS ${M_LIBRARY})
    endif()
    if(RT_LIBRARY)
        list(APPEND PETSC_EXTRA_LIBS ${RT_LIBRARY})
    endif()

    # f2cblaslapack is downloaded by PETSc configure
    set_target_properties(PETSc::PETSc PROPERTIES
        INTERFACE_LINK_LIBRARIES
        "${PETSC_INSTALL_DIR}/lib/libf2clapack.a;${PETSC_INSTALL_DIR}/lib/libf2cblas.a;${PETSC_EXTRA_LIBS};MPI::MPI_CXX"
    )
endif()

# Library sources
set(LIB_SOURCES
    src/Checkpoint.cpp
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
    src/MixtureEOS.cpp
    src/ViscousFlux.cpp
    src/SurfaceTension.cpp
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

target_link_libraries(SemiImplicitFV PUBLIC MPI::MPI_CXX)

# PETSc integration
if(ENABLE_PETSC)
    target_sources(SemiImplicitFV PRIVATE src/PETScPressureSolver.cpp)
    target_compile_definitions(SemiImplicitFV PUBLIC SIFV_HAS_PETSC)
    target_link_libraries(SemiImplicitFV PUBLIC PETSc::PETSc)
    target_include_directories(SemiImplicitFV PUBLIC ${PETSC_INSTALL_DIR}/include)
    add_dependencies(SemiImplicitFV petsc_external)
endif()

# ---- JSON Driver (sifv) ----
if(BUILD_DRIVER)
    include(FetchContent)

    # nlohmann/json (header-only)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    set(JSON_Install OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(nlohmann_json)

    # exprtk (header-only math expression parser)
    FetchContent_Declare(
        exprtk
        GIT_REPOSITORY https://github.com/ArashPartow/exprtk.git
        GIT_TAG master
    )
    FetchContent_Populate(exprtk)

    # Input library (keeps core library dependency-free)
    add_library(SemiImplicitFV_input
        src/InputParser.cpp
        src/ICPatch.cpp
        src/ExpressionEvaluator.cpp
    )
    target_link_libraries(SemiImplicitFV_input PUBLIC SemiImplicitFV nlohmann_json::nlohmann_json)
    target_include_directories(SemiImplicitFV_input PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_include_directories(SemiImplicitFV_input PRIVATE ${exprtk_SOURCE_DIR})

    # Generic driver executable
    add_executable(sifv driver/main.cpp)
    target_link_libraries(sifv PRIVATE SemiImplicitFV_input)
endif()

# Compiled cases — auto-discover every cases/<name>/*.cpp
if(BUILD_EXAMPLES)
    file(GLOB CASE_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/cases
         ${CMAKE_CURRENT_SOURCE_DIR}/cases/*)
    foreach(CASE_DIR ${CASE_DIRS})
        if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cases/${CASE_DIR})
            file(GLOB CASE_SOURCES cases/${CASE_DIR}/*.cpp)
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

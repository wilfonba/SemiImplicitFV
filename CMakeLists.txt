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
option(ENABLE_NVTX "Enable NVTX profiling ranges for Nsight Systems" OFF)
option(ENABLE_GPU_OFFLOAD "Enable OpenMP target offload to NVIDIA GPUs (requires NVHPC)" OFF)
set(GPU_OFFLOAD_CC "" CACHE STRING "Optional target compute capability (e.g. cc80 for A100, cc90 for H100). Empty = NVHPC default (multi-cc).")

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    # OpenMP `target` / `declare target` pragmas are intentional; silence them
    # when building without an offload-capable OpenMP mode.
    add_compile_options(-Wno-unknown-pragmas)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
    # nvc++ defaults to -O0; we have to ask for optimization explicitly.
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
        # -fast: enables -O2, -Munroll=c:1, -Mlre, -Mautoinline, -Mvect=simd,
        #        -Mflushz, -Mcache_align, -Mpre — nvc++'s standard release
        #        bundle.  -Mfprelaxed lets the GPU back-end use rsqrt/rcp etc.
        add_compile_options(-fast -Mfprelaxed)
        add_link_options(-fast)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g)
    endif()
endif()

# GPU offload (NVHPC) --------------------------------------------------------
# Uses OpenMP target directives.  Flags are scoped to the core
# SemiImplicitFV library (applied later via target_compile_options) rather
# than globally, so that the exprtk-backed input library — whose template
# pathology exhausts nvc++'s frontend under -mp=gpu — can build with the
# default (host-only) codegen.
if(ENABLE_GPU_OFFLOAD)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
        message(FATAL_ERROR
            "ENABLE_GPU_OFFLOAD=ON requires the NVHPC C++ compiler (nvc++). "
            "Current compiler: ${CMAKE_CXX_COMPILER_ID}.")
    endif()
    set(_gpu_target "gpu")
    # -mp=gpu                : OpenMP target offload to GPU.
    # -gpu=lineinfo          : line-number debug for nsys/ncu source attrib.
    # -gpu=fastmath          : let the device math library use approximate
    #                          intrinsics (rsqrt, rcp, sin/cos units).  Same
    #                          spirit as nvcc -use_fast_math.
    # -gpu=unified           : place automatic vars in unified-memory pools
    #                          when needed (NVHPC 24.x+).  Cheap to leave on.
    # -Mloop-count=...       : not used here; we rely on `omp target teams
    #                          distribute parallel for collapse(3)` for shape.
    # -Minfo=accel,opt,mp    : print kernel codegen + register/spill info to
    #                          stderr at compile time.  Invaluable for
    #                          spotting register spills and parallelism loss.
    set(SIFV_GPU_FLAGS "-mp=gpu;-gpu=lineinfo;-gpu=fastmath;-Minfo=accel,mp")
    set(SIFV_GPU_LINK_FLAGS "-mp=gpu")
    if(GPU_OFFLOAD_CC)
        set(_gpu_target "gpu=${GPU_OFFLOAD_CC}")
        list(APPEND SIFV_GPU_FLAGS "-gpu=${GPU_OFFLOAD_CC}")
        list(APPEND SIFV_GPU_LINK_FLAGS "-gpu=${GPU_OFFLOAD_CC}")
    endif()
    message(STATUS "GPU offload: enabled (${_gpu_target})")
endif()

# MPI
find_package(MPI REQUIRED)

# PETSc (optional, built from source via ExternalProject)
if(ENABLE_PETSC)
    set(PETSC_INSTALL_DIR ${CMAKE_BINARY_DIR}/petsc-install)

    # PETSc static lib dependencies (needed whether building or reusing)
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

    # Skip the slow ExternalProject if PETSc is already installed
    if(EXISTS ${PETSC_INSTALL_DIR}/lib/libpetsc.a)
        message(STATUS "PETSc: using existing install at ${PETSC_INSTALL_DIR}")
    else()
        message(STATUS "PETSc: building from source (first time)...")
        include(ExternalProject)

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

        set(PETSC_NEEDS_BUILD TRUE)
    endif()

    # Create imported target for PETSc
    add_library(PETSc::PETSc STATIC IMPORTED)
    set_target_properties(PETSc::PETSc PROPERTIES
        IMPORTED_LOCATION ${PETSC_INSTALL_DIR}/lib/libpetsc.a
        INTERFACE_INCLUDE_DIRECTORIES ${PETSC_INSTALL_DIR}/include
    )

    # f2cblaslapack is downloaded by PETSc configure
    set_target_properties(PETSc::PETSc PROPERTIES
        INTERFACE_LINK_LIBRARIES
        "${PETSC_INSTALL_DIR}/lib/libf2clapack.a;${PETSC_INSTALL_DIR}/lib/libf2cblas.a;${PETSC_EXTRA_LIBS};MPI::MPI_CXX"
    )
endif()

# NVTX profiling (optional, requires CUDA Toolkit for nvToolsExt)
if(ENABLE_NVTX)
    find_package(CUDAToolkit REQUIRED)
endif()

# Library sources
set(LIB_SOURCES
    src/Checkpoint.cpp
    src/SimulationConfig.cpp
    src/EquationOfState.cpp
    src/SolutionState.cpp
    src/RectilinearMesh.cpp
    src/Reconstruction.cpp
    src/RiemannSolver.cpp
    src/IGR.cpp
    src/SemiImplicitSolver.cpp
    src/ExplicitSolver.cpp
    src/RKTimeStepping.cpp
    src/PressureLaplacian.cpp
    src/PressureSolver.cpp
    src/MixtureEOS.cpp
    src/ViscousFlux.cpp
    src/SurfaceTension.cpp
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

# Apply GPU offload flags.  Compile flags are PRIVATE — only translation
# units in SemiImplicitFV see -mp=gpu, which keeps exprtk-heavy files in
# the input library out of the offload codegen path.  Link flags are
# PUBLIC so downstream binaries (sifv driver, tests) link the OpenMP
# runtime too.
if(ENABLE_GPU_OFFLOAD)
    target_compile_options(SemiImplicitFV PRIVATE ${SIFV_GPU_FLAGS})
    target_link_options(SemiImplicitFV PUBLIC ${SIFV_GPU_LINK_FLAGS})
endif()

# PETSc integration
if(ENABLE_PETSC)
    target_sources(SemiImplicitFV PRIVATE src/PETScPressureSolver.cpp)
    target_compile_definitions(SemiImplicitFV PUBLIC SIFV_HAS_PETSC)
    target_link_libraries(SemiImplicitFV PUBLIC PETSc::PETSc)
    target_include_directories(SemiImplicitFV PUBLIC ${PETSC_INSTALL_DIR}/include)
    if(PETSC_NEEDS_BUILD)
        add_dependencies(SemiImplicitFV petsc_external)
    endif()
endif()

# NVTX integration (NVTX3 is header-only in modern CUDA toolkits)
if(ENABLE_NVTX)
    target_compile_definitions(SemiImplicitFV PUBLIC SIFV_HAS_NVTX)
    target_include_directories(SemiImplicitFV PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    # CUDA 12+ removed the nvToolsExt shared library; NVTX3 is header-only.
    # Link the library only if it exists (CUDA <=11).
    if(TARGET CUDA::nvToolsExt)
        target_link_libraries(SemiImplicitFV PUBLIC CUDA::nvToolsExt)
    endif()
endif()

# ---- Input parsing library (needed by driver and tests) ----
if(BUILD_DRIVER OR BUILD_TESTS)
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

    if(ENABLE_GPU_OFFLOAD)
        # nvc++'s frontend (nvcpfe) cannot compile exprtk — the template
        # pathology exhausts its memory and the process is killed.  Build
        # the input library with the host compiler (g++/clang++) instead
        # and link the resulting objects in alongside the nvc++-built core.
        find_program(SIFV_HOST_CXX NAMES g++ c++ clang++ REQUIRED
            DOC "Host C++ compiler used for exprtk-backed input library")
        message(STATUS "Input library host compiler: ${SIFV_HOST_CXX}")

        set(_sifv_input_srcs
            ${CMAKE_CURRENT_SOURCE_DIR}/src/InputParser.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ICPatch.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ExpressionEvaluator.cpp
        )
        set(_sifv_input_objdir ${CMAKE_CURRENT_BINARY_DIR}/input_host_objs)
        file(MAKE_DIRECTORY ${_sifv_input_objdir})
        set(_sifv_input_objs "")
        foreach(_src ${_sifv_input_srcs})
            get_filename_component(_name ${_src} NAME_WE)
            set(_obj ${_sifv_input_objdir}/${_name}.o)
            add_custom_command(
                OUTPUT ${_obj}
                COMMAND ${SIFV_HOST_CXX} -std=c++17 -O3 -fPIC
                    -I${CMAKE_CURRENT_SOURCE_DIR}/include
                    -I${nlohmann_json_SOURCE_DIR}/include
                    -I${exprtk_SOURCE_DIR}
                    -c ${_src} -o ${_obj}
                DEPENDS ${_src}
                VERBATIM
                COMMENT "Building CXX object (host compiler) ${_name}.cpp.o"
            )
            list(APPEND _sifv_input_objs ${_obj})
        endforeach()

        add_library(SemiImplicitFV_input STATIC ${_sifv_input_objs})
        set_target_properties(SemiImplicitFV_input PROPERTIES LINKER_LANGUAGE CXX)
        target_link_libraries(SemiImplicitFV_input PUBLIC SemiImplicitFV nlohmann_json::nlohmann_json)
        target_include_directories(SemiImplicitFV_input PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )
    else()
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
    endif()
endif()

# ---- JSON Driver (sifv) ----
if(BUILD_DRIVER)
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

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
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

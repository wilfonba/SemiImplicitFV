include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# MPI wrapper for test discovery and execution.
# CROSSCOMPILING_EMULATOR is used by gtest_discover_tests to prefix each test
# command with mpirun, so every individual GoogleTest case runs under MPI.
set(MPI_WRAPPER "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "1")
set(MPI_WRAPPER_NP4 "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "4")
set(MPI_TEST_ENV "OMPI_MCA_rmaps_base_oversubscribe=1")

# ---- Shared MPI main ----
add_library(test_main OBJECT test_main.cpp)
target_link_libraries(test_main PUBLIC gtest MPI::MPI_CXX SemiImplicitFV)

# ---- Unit tests ----
add_executable(test_unit
    unit/test_eos.cpp
    unit/test_riemann_solver.cpp
    unit/test_reconstruction.cpp
    unit/test_mixture_eos.cpp
    unit/test_igr.cpp
    unit/test_mthinc.cpp
)
target_link_libraries(test_unit PRIVATE test_main gtest SemiImplicitFV MPI::MPI_CXX)
set_target_properties(test_unit PROPERTIES CROSSCOMPILING_EMULATOR "${MPI_WRAPPER}")
gtest_discover_tests(test_unit
    PROPERTIES
        LABELS "unit"
        ENVIRONMENT "${MPI_TEST_ENV}"
)

# ---- Integration tests ----
add_executable(test_integration
    integration/test_boundary_conditions.cpp
    integration/test_state_conversion.cpp
    integration/test_explicit_rhs.cpp
    integration/test_pressure_solver.cpp
    integration/test_mthinc.cpp
)
target_link_libraries(test_integration PRIVATE test_main gtest SemiImplicitFV MPI::MPI_CXX)
set_target_properties(test_integration PROPERTIES CROSSCOMPILING_EMULATOR "${MPI_WRAPPER}")
gtest_discover_tests(test_integration
    PROPERTIES
        LABELS "integration"
        ENVIRONMENT "${MPI_TEST_ENV}"
)

# ---- Regression tests ----
# Copy test case files and references to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/regression/cases
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/regression)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/regression/references
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/regression)

add_executable(test_regression
    regression/test_regression.cpp
)
target_link_libraries(test_regression PRIVATE test_main gtest SemiImplicitFV_input MPI::MPI_CXX)
target_compile_definitions(test_regression PRIVATE
    TEST_CASES_DIR="${CMAKE_CURRENT_BINARY_DIR}/regression/cases"
    TEST_REFS_DIR="${CMAKE_CURRENT_BINARY_DIR}/regression/references"
    SOURCE_REFS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/regression/references"
)

# np=1: register each regression case individually (names must match TEST() macros)
# We can't use gtest_discover_tests for both np=1 and np=4 with different mpirun args,
# so we register all regression tests manually for consistent behavior.
set(REGRESSION_CASES
    SodShocktube1D
    AdvectionSI1D
    GasGasShocktube1D
    IsentropicVortex2D
    ChannelFlow2D
    TaylorGreenVortex3D
    TaylorGreenVortexSI3D
)

foreach(case ${REGRESSION_CASES})
    add_test(NAME "np1/Regression.${case}"
        COMMAND ${MPI_WRAPPER}
                $<TARGET_FILE:test_regression>
                "--gtest_filter=Regression.${case}")
    set_tests_properties("np1/Regression.${case}" PROPERTIES
        LABELS "regression;regression_np1"
        ENVIRONMENT "${MPI_TEST_ENV}"
    )
endforeach()

# np=4: same cases with 4 MPI ranks
foreach(case ${REGRESSION_CASES})
    add_test(NAME "np4/Regression.${case}"
        COMMAND ${MPI_WRAPPER_NP4}
                $<TARGET_FILE:test_regression>
                "--gtest_filter=Regression.${case}")
    set_tests_properties("np4/Regression.${case}" PROPERTIES
        LABELS "regression;regression_np4"
        ENVIRONMENT "${MPI_TEST_ENV}"
    )
endforeach()

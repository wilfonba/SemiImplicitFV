#include <gtest/gtest.h>
#include <mpi.h>
#ifdef SIFV_HAS_PETSC
#include <petsc.h>
#endif

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
#ifdef SIFV_HAS_PETSC
    PetscInitialize(&argc, &argv, NULL, NULL);
#endif
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
#ifdef SIFV_HAS_PETSC
    PetscFinalize();
#endif
    MPI_Finalize();
    return result;
}

#include <gtest/gtest.h>
#include <mpi.h>
#ifdef SIFV_HAS_PETSC
#include <petsc.h>
#endif
#include <omp.h>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
#ifdef SIFV_HAS_PETSC
    PetscInitialize(&argc, &argv, NULL, NULL);
#endif
    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    int ndev = omp_get_num_devices();
    if (ndev > 0) {
        int localRank = -1;
        const char* envs[] = {
            "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK",
            "SLURM_LOCALID", "MPI_LOCALRANKID", "PMI_LOCAL_RANK", NULL };
        for (int i = 0; envs[i]; ++i) {
            const char* s = std::getenv(envs[i]);
            if (s && *s) { localRank = std::atoi(s); break; }
        }
        if (localRank < 0) localRank = worldRank;
        omp_set_default_device(localRank % ndev);
    }
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
#ifdef SIFV_HAS_PETSC
    PetscFinalize();
#endif
    MPI_Finalize();
    return result;
}

#include "VTKSession.hpp"
#include "Runtime.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"

#include <mpi.h>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

void vtk_session_init(VTKSession* s,
                      Runtime* rt,
                      const char* baseName,
                      const RectilinearMesh* mesh,
                      const SimulationConfig* config,
                      const char* dir,
                      VTKFormat format)
{
    std::strncpy(s->baseName, baseName, sizeof(s->baseName) - 1);
    s->baseName[sizeof(s->baseName) - 1] = '\0';
    std::strncpy(s->dir, dir ? dir : "VTK", sizeof(s->dir) - 1);
    s->dir[sizeof(s->dir) - 1] = '\0';
    s->format = format;
    s->fileNum = 0;
    s->rt = rt;
    s->mesh = mesh;
    s->config = config;

    std::memcpy(s->localExtent, rt->mpiCtx->localExtent, 6 * sizeof(int));

    if (rt->rank == 0) {
        std::string pvdPath = std::string(s->dir) + "/" + s->baseName + ".pvd";
        vtk_write_pvd(pvdPath.c_str(), 'w', 0.0, NULL);
    }
}

void vtk_session_write(VTKSession* s,
                       const SolutionState* state,
                       double time)
{
    if (!s) return;
    int rank = s->rt->rank;
    int nprocs = s->rt->size;

    std::string snapshotDir = "snapshot_" + std::to_string(s->fileNum);

    std::string vtrFile = std::string(s->baseName) + "_" + std::to_string(s->fileNum)
                        + "_r" + std::to_string(rank) + ".vtr";
    std::string vtrPath = std::string(s->dir) + "/" + snapshotDir + "/" + vtrFile;
    vtk_write_vtr(vtrPath.c_str(), s->mesh, state, s->config,
                  s->localExtent, rank, s->format);

    /* Gather all extents on rank 0 */
    std::vector<int> allExtBuf(nprocs * 6);
    MPI_Gather(s->localExtent, 6, MPI_INT,
               allExtBuf.data(), 6, MPI_INT, 0, s->rt->mpiCtx->cartComm);

    if (rank == 0) {
        std::vector<int> allExtents(nprocs * 6);
        std::vector<std::string> allFileStrs(nprocs);
        std::vector<const char*> allFiles(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            for (int d = 0; d < 6; ++d)
                allExtents[r * 6 + d] = allExtBuf[r * 6 + d];
            allFileStrs[r] = std::string(s->baseName) + "_" + std::to_string(s->fileNum)
                           + "_r" + std::to_string(r) + ".vtr";
            allFiles[r] = allFileStrs[r].c_str();
        }
        std::string pvtrFile = std::string(s->baseName) + "_" + std::to_string(s->fileNum) + ".pvtr";
        std::string pvtrPath = std::string(s->dir) + "/" + snapshotDir + "/" + pvtrFile;
        vtk_write_pvtr(pvtrPath.c_str(),
                       s->rt->globalNx, s->rt->globalNy, s->rt->globalNz,
                       allExtents.data(), allFiles.data(), nprocs,
                       s->config, s->format);

        std::string pvdPath = std::string(s->dir) + "/" + s->baseName + ".pvd";
        std::string pvdDataFile = snapshotDir + "/" + pvtrFile;
        vtk_write_pvd(pvdPath.c_str(), 'a', time, pvdDataFile.c_str());
    }

    s->fileNum++;
}

void vtk_session_finalize(VTKSession* s)
{
    if (!s) return;
    if (s->rt->rank == 0) {
        std::string pvdPath = std::string(s->dir) + "/" + s->baseName + ".pvd";
        vtk_write_pvd(pvdPath.c_str(), 'c', 0.0, NULL);
    }
}

#include "VTKSession.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"

#include <mpi.h>

#include <vector>

namespace SemiImplicitFV {

VTKSession::VTKSession(Runtime& rt, const std::string& baseName,
                       const RectilinearMesh& mesh, const std::string& dir)
    : rt_(rt), mesh_(mesh), baseName_(baseName), dir_(dir)
{
    localExtent_ = rt_.mpiContext().localExtent();

    if (rt_.isRoot()) {
        VTKWriter::writePVD(baseName_ + ".pvd", "w");
    }
}

void VTKSession::write(const SolutionState& state, double time) {
    int rank = rt_.rank();
    int nprocs = rt_.size();

    std::string vtrFile = baseName_ + "_" + std::to_string(fileNum_)
                        + "_r" + std::to_string(rank) + ".vtr";
    VTKWriter::writeVTR(dir_ + "/" + vtrFile, mesh_, state, localExtent_, rank);

    // Gather all extents on rank 0
    std::vector<int> allExtBuf(nprocs * 6);
    MPI_Gather(localExtent_.data(), 6, MPI_INT,
               allExtBuf.data(), 6, MPI_INT, 0, rt_.mpiContext().comm());

    if (rt_.isRoot()) {
        std::vector<std::array<int,6>> allExtents(nprocs);
        std::vector<std::string> allFiles(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            for (int d = 0; d < 6; ++d)
                allExtents[r][d] = allExtBuf[r * 6 + d];
            allFiles[r] = baseName_ + "_" + std::to_string(fileNum_)
                        + "_r" + std::to_string(r) + ".vtr";
        }
        std::string pvtrFile = baseName_ + "_" + std::to_string(fileNum_) + ".pvtr";
        int nPhases = static_cast<int>(state.alphaRho.size());
        VTKWriter::writePVTR(dir_ + "/" + pvtrFile,
                             rt_.globalNx(), rt_.globalNy(), rt_.globalNz(),
                             allExtents, allFiles, nPhases);
        VTKWriter::writePVD(baseName_ + ".pvd", "a", time, dir_ + "/" + pvtrFile);
    }

    fileNum_++;
}

void VTKSession::finalize() {
    if (rt_.isRoot()) {
        VTKWriter::writePVD(baseName_ + ".pvd", "close");
    }
}

} // namespace SemiImplicitFV

#ifndef VTK_WRITER_HPP
#define VTK_WRITER_HPP

#include <string>
#include <vector>
#include <array>

namespace SemiImplicitFV {

class RectilinearMesh;
class SolutionState;

/// VTK XML RectilinearGrid writer for ParaView visualization.
///
/// Supports serial .vtr files, parallel .pvtr meta-files,
/// and .pvd time-series collection files.
class VTKWriter {
public:
    /// Write a single .vtr piece file.
    /// pieceExtent: {i0, i1, j0, j1, k0, k1} local extent within global grid.
    /// Default (empty/zero) = full grid (serial mode).
    /// rank: MPI rank for "Rank" cell data field (-1 = omit).
    static void writeVTR(const std::string& filename,
                         const RectilinearMesh& mesh,
                         const SolutionState& state,
                         const std::array<int,6>& pieceExtent = {},
                         int rank = -1);

    /// Write .pvtr parallel meta-file referencing piece files.
    /// Only rank 0 calls this in MPI.
    /// @param globalNx/Ny/Nz  Global cell counts for WholeExtent.
    static void writePVTR(const std::string& filename,
                          int globalNx, int globalNy, int globalNz,
                          const std::vector<std::array<int,6>>& pieceExtents,
                          const std::vector<std::string>& pieceFiles,
                          int nPhases = 0);

    /// Three-phase .pvd time-series file:
    ///   mode="w"     -- write header
    ///   mode="a"     -- append timestep entry
    ///   mode="close" -- write closing tags
    static void writePVD(const std::string& filename,
                         const std::string& mode,
                         double time = 0.0,
                         const std::string& dataFile = "");
};

} // namespace SemiImplicitFV

#endif // VTK_WRITER_HPP

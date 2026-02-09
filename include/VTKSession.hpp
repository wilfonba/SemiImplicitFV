#ifndef VTK_SESSION_HPP
#define VTK_SESSION_HPP

#include "VTKWriter.hpp"
#include "Runtime.hpp"
#include <string>
#include <array>

namespace SemiImplicitFV {

class RectilinearMesh;
class SolutionState;

/// Encapsulates the entire VTK output lifecycle: PVD open/append/close,
/// per-rank VTR writing, and MPI gather + PVTR meta-file generation.
class VTKSession {
public:
    VTKSession(Runtime& rt, const std::string& baseName,
               const RectilinearMesh& mesh, const std::string& dir = "VTK");

    /// Write current state at the given time.  Collective in MPI mode.
    void write(const SolutionState& state, double time);

    /// Close the PVD time-series file.
    void finalize();

private:
    Runtime& rt_;
    const RectilinearMesh& mesh_;
    std::string baseName_;
    std::string dir_;
    int fileNum_ = 0;

#ifdef ENABLE_MPI
    std::array<int,6> localExtent_ = {};
#endif
};

} // namespace SemiImplicitFV

#endif // VTK_SESSION_HPP

#ifndef MPI_CONTEXT_HPP
#define MPI_CONTEXT_HPP

#include <mpi.h>
#include <array>
#include <vector>

namespace SemiImplicitFV {

/// Cartesian MPI domain decomposition context.
///
/// Wraps MPI_Cart_create for structured grid decomposition, providing
/// neighbor lookup, physical-boundary queries, and local grid extents.
class MPIContext {
public:
    /// Factory: create a Cartesian topology from global grid parameters.
    /// @param globalNx/Ny/Nz  Number of *cells* in each direction (global).
    /// @param xNodes/yNodes/zNodes  Global node coordinate arrays.
    /// @param dim  Spatial dimensionality (1, 2, or 3).
    /// @param periods  Periodic flags per direction (x, y, z).
    /// @param procsPerDir  Desired process counts per direction (0 = auto).
    static MPIContext create(int globalNx, int globalNy, int globalNz,
                             const std::vector<double>& xNodes,
                             const std::vector<double>& yNodes,
                             const std::vector<double>& zNodes,
                             int dim,
                             const std::array<int,3>& periods = {0,0,0},
                             const std::array<int,3>& procsPerDir = {0,0,0});

    ~MPIContext();

    // Prevent accidental copies
    MPIContext(const MPIContext&) = delete;
    MPIContext& operator=(const MPIContext&) = delete;
    MPIContext(MPIContext&& other) noexcept;
    MPIContext& operator=(MPIContext&& other) noexcept;

    /// MPI rank in the Cartesian communicator.
    int rank() const { return rank_; }
    /// Total number of processes.
    int size() const { return size_; }
    /// Cartesian communicator.
    MPI_Comm comm() const { return cartComm_; }

    /// Process grid dimensions.
    const std::array<int,3>& dims() const { return dims_; }
    /// This rank's Cartesian coordinates.
    const std::array<int,3>& coords() const { return coords_; }

    /// Face identifiers (matching RectilinearMesh).
    static constexpr int XLow  = 0;
    static constexpr int XHigh = 1;
    static constexpr int YLow  = 2;
    static constexpr int YHigh = 3;
    static constexpr int ZLow  = 4;
    static constexpr int ZHigh = 5;

    /// Neighbor rank for given face (MPI_PROC_NULL if physical boundary).
    int neighbor(int face) const { return neighbors_[face]; }

    /// True if this rank owns a physical (non-MPI) boundary on the given face.
    bool isPhysicalBoundary(int face) const {
        return neighbors_[face] == MPI_PROC_NULL;
    }

    /// Local node coordinate arrays (sliced from global arrays).
    const std::vector<double>& localXNodes() const { return localXNodes_; }
    const std::vector<double>& localYNodes() const { return localYNodes_; }
    const std::vector<double>& localZNodes() const { return localZNodes_; }

    /// Local cell counts.
    int localNx() const { return localNx_; }
    int localNy() const { return localNy_; }
    int localNz() const { return localNz_; }

    /// Global extent of this rank's piece (cell indices, for VTK output).
    /// {i0, i1, j0, j1, k0, k1} where i1 = i0 + localNx, etc.
    const std::array<int,6>& localExtent() const { return localExtent_; }

private:
    MPIContext() = default;

    MPI_Comm cartComm_ = MPI_COMM_NULL;
    int rank_ = 0;
    int size_ = 1;
    std::array<int,3> dims_ = {1,1,1};
    std::array<int,3> coords_ = {0,0,0};
    std::array<int,6> neighbors_ = {MPI_PROC_NULL, MPI_PROC_NULL,
                                     MPI_PROC_NULL, MPI_PROC_NULL,
                                     MPI_PROC_NULL, MPI_PROC_NULL};

    std::vector<double> localXNodes_;
    std::vector<double> localYNodes_;
    std::vector<double> localZNodes_;
    int localNx_ = 0, localNy_ = 0, localNz_ = 0;
    std::array<int,6> localExtent_ = {0,0,0,0,0,0};

    /// Slice a global node array into local nodes for a given coordinate.
    static std::vector<double> sliceNodes(const std::vector<double>& globalNodes,
                                           int globalN, int nProcs, int coord);
};

} // namespace SemiImplicitFV

#endif // MPI_CONTEXT_HPP

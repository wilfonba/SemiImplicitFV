#ifndef RECTILINEAR_MESH_HPP
#define RECTILINEAR_MESH_HPP

#include "SolutionState.hpp"
#include <vector>
#include <array>
#include <cstddef>

namespace SemiImplicitFV {

enum class BoundaryCondition {
    Symmetry,  // symmetry
    Outflow,     // Zero gradient (extrapolation from nearest interior)
    Periodic,    // Wrap around to opposite boundary
    SlipWall,    // Wall: copy tangential velocity, reflect normal velocity, copy scalars
    NoSlipWall  // Wall: zero velocity, copy scalars
};

/// Rectilinear mesh supporting 1D, 2D, and 3D grids without AMR.
///
/// Grid geometry is defined by 1D node-coordinate arrays along each axis.
/// For lower-dimensional problems, unused dimensions have a single cell
/// (ny=1 for 1D, nz=1 for 1D/2D) with unit width. Ghost cells are only
/// added in active dimensions.
///
/// Indexing is x-fastest (i varies fastest).
class RectilinearMesh {
public:
    /// Construct from physical node coordinates.
    /// For 1D: pass only xNodes (yNodes/zNodes default to {0,1}).
    /// For 2D: pass xNodes and yNodes (zNodes defaults to {0,1}).
    RectilinearMesh(int dim,
                    const std::vector<double>& xNodes,
                    const std::vector<double>& yNodes = {0.0, 1.0},
                    const std::vector<double>& zNodes = {0.0, 1.0},
                    int nGhost = 2);

    /// Convenience factory for uniform grids.
    static RectilinearMesh createUniform(int dim,
                                         int nx, double xMin, double xMax,
                                         int ny = 1, double yMin = 0.0, double yMax = 1.0,
                                         int nz = 1, double zMin = 0.0, double zMax = 1.0,
                                         int nGhost = 2);

    // --- Grid topology ---

    int dim() const { return dim_; }
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    int nGhost() const { return nGhost_; }

    /// Ghost cell count in each direction (0 for inactive dimensions).
    int ngx() const { return ngx_; }
    int ngy() const { return ngy_; }
    int ngz() const { return ngz_; }

    /// Total cells including ghosts per direction.
    int nxTotal() const { return nx_ + 2 * ngx_; }
    int nyTotal() const { return ny_ + 2 * ngy_; }
    int nzTotal() const { return nz_ + 2 * ngz_; }

    /// Total number of cells (including all ghost cells).
    std::size_t totalCells() const;

    /// Flat array index from cell indices (i,j,k).
    /// Physical cells: i in [0,nx), j in [0,ny), k in [0,nz).
    /// Ghost cells extend into negative indices and past nx/ny/nz.
    std::size_t index(int i, int j, int k) const {
        return static_cast<std::size_t>(
            (i + ngx_) + nxTotal() * ((j + ngy_) + nyTotal() * (k + ngz_)));
    }

    // --- Geometry (computed from node arrays, not stored per-cell) ---

    double dx(int i) const { return xNodesExt_[i + ngx_ + 1] - xNodesExt_[i + ngx_]; }
    double dy(int j) const { return yNodesExt_[j + ngy_ + 1] - yNodesExt_[j + ngy_]; }
    double dz(int k) const { return zNodesExt_[k + ngz_ + 1] - zNodesExt_[k + ngz_]; }

    double cellVolume(int i, int j, int k) const {
        return dx(i) * dy(j) * dz(k);
    }

    double cellCentroidX(int i) const {
        return 0.5 * (xNodesExt_[i + ngx_] + xNodesExt_[i + ngx_ + 1]);
    }
    double cellCentroidY(int j) const {
        return 0.5 * (yNodesExt_[j + ngy_] + yNodesExt_[j + ngy_ + 1]);
    }
    double cellCentroidZ(int k) const {
        return 0.5 * (zNodesExt_[k + ngz_] + zNodesExt_[k + ngz_ + 1]);
    }

    /// Face area normal to x-axis (between cells (i,j,k) and (i+1,j,k)).
    double faceAreaX(int j, int k) const { return dy(j) * dz(k); }
    /// Face area normal to y-axis (between cells (i,j,k) and (i,j+1,k)).
    double faceAreaY(int i, int k) const { return dx(i) * dz(k); }
    /// Face area normal to z-axis (between cells (i,j,k) and (i,j,k+1)).
    double faceAreaZ(int i, int j) const { return dx(i) * dy(j); }

    // --- Boundary conditions ---

    /// Face identifiers for setBoundaryCondition().
    static constexpr int XLow  = 0;
    static constexpr int XHigh = 1;
    static constexpr int YLow  = 2;
    static constexpr int YHigh = 3;
    static constexpr int ZLow  = 4;
    static constexpr int ZHigh = 5;

    void setBoundaryCondition(int face, BoundaryCondition bc);
    BoundaryCondition boundaryCondition(int face) const { return bc_[face]; }

    /// Fill ghost cells for all active dimensions based on current BCs.
    /// Uses an onion-peel ordering so edge/corner ghosts are filled correctly.
    void applyBoundaryConditions(SolutionState& state) const;

    /// Fill ghost cells for a single scalar field.
    void fillScalarGhosts(std::vector<double>& field) const;

private:
    int dim_;
    int nx_, ny_, nz_;
    int nGhost_;
    int ngx_, ngy_, ngz_;

    // Extended node arrays (physical + ghost regions).
    // Size: nTotal + 1 entries per direction (one more node than cells).
    std::vector<double> xNodesExt_;
    std::vector<double> yNodesExt_;
    std::vector<double> zNodesExt_;

    std::array<BoundaryCondition, 6> bc_;

    /// Build an extended node array from physical nodes by mirroring cell
    /// widths into the ghost region on each side.
    static std::vector<double> buildExtendedNodes(
        const std::vector<double>& physNodes, int nCells, int ng);

    // Ghost-fill helpers for each direction.
    void fillGhostX(SolutionState& state) const;
    void fillGhostY(SolutionState& state) const;
    void fillGhostZ(SolutionState& state) const;
};

} // namespace SemiImplicitFV

#endif // RECTILINEAR_MESH_HPP

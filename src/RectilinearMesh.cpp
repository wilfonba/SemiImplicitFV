#include "RectilinearMesh.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace SemiImplicitFV {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

RectilinearMesh::RectilinearMesh(int dim,
                                 const std::vector<double>& xNodes,
                                 const std::vector<double>& yNodes,
                                 const std::vector<double>& zNodes,
                                 int nGhost)
    : dim_(dim), nGhost_(nGhost)
{
    if (dim < 1 || dim > 3) {
        throw std::invalid_argument("RectilinearMesh: dim must be 1, 2, or 3");
    }
    if (nGhost < 0) {
        throw std::invalid_argument("RectilinearMesh: nGhost must be non-negative");
    }
    if (xNodes.size() < 2) {
        throw std::invalid_argument("RectilinearMesh: xNodes must have at least 2 entries");
    }
    if (dim >= 2 && yNodes.size() < 2) {
        throw std::invalid_argument("RectilinearMesh: yNodes must have at least 2 entries for 2D/3D");
    }
    if (dim >= 3 && zNodes.size() < 2) {
        throw std::invalid_argument("RectilinearMesh: zNodes must have at least 2 entries for 3D");
    }

    nx_ = static_cast<int>(xNodes.size()) - 1;
    ny_ = static_cast<int>(yNodes.size()) - 1;
    nz_ = static_cast<int>(zNodes.size()) - 1;

    // Ghost cells only in active dimensions.
    ngx_ = nGhost_;
    ngy_ = (dim_ >= 2) ? nGhost_ : 0;
    ngz_ = (dim_ >= 3) ? nGhost_ : 0;

    // Build extended node arrays (physical + ghost region on each side).
    xNodesExt_ = buildExtendedNodes(xNodes, nx_, ngx_);
    yNodesExt_ = buildExtendedNodes(yNodes, ny_, ngy_);
    zNodesExt_ = buildExtendedNodes(zNodes, nz_, ngz_);

    // Default all boundaries to outflow.
    bc_.fill(BoundaryCondition::Outflow);
}

RectilinearMesh RectilinearMesh::createUniform(
    int dim,
    int nx, double xMin, double xMax,
    int ny, double yMin, double yMax,
    int nz, double zMin, double zMax,
    int nGhost)
{
    auto linspace = [](int n, double lo, double hi) {
        std::vector<double> nodes(n + 1);
        double h = (hi - lo) / n;
        for (int i = 0; i <= n; ++i) {
            nodes[i] = lo + i * h;
        }
        return nodes;
    };

    return RectilinearMesh(dim,
                           linspace(nx, xMin, xMax),
                           linspace(ny, yMin, yMax),
                           linspace(nz, zMin, zMax),
                           nGhost);
}

// ---------------------------------------------------------------------------
// Topology helpers
// ---------------------------------------------------------------------------

std::size_t RectilinearMesh::totalCells() const {
    return static_cast<std::size_t>(nxTotal()) * nyTotal() * nzTotal();
}

// ---------------------------------------------------------------------------
// Extended node construction
// ---------------------------------------------------------------------------

std::vector<double> RectilinearMesh::buildExtendedNodes(
    const std::vector<double>& physNodes, int nCells, int ng)
{
    int nExt = nCells + 2 * ng + 1;
    std::vector<double> ext(nExt);

    for (int i = 0; i <= nCells; ++i) {
        ext[ng + i] = physNodes[i];
    }

    for (int g = 1; g <= ng; ++g) {
        int mirror = std::min(g - 1, nCells - 1);
        double width = physNodes[mirror + 1] - physNodes[mirror];
        ext[ng - g] = ext[ng - g + 1] - width;
    }

    for (int g = 1; g <= ng; ++g) {
        int mirror = std::max(nCells - g, 0);
        double width = physNodes[mirror + 1] - physNodes[mirror];
        ext[ng + nCells + g] = ext[ng + nCells + g - 1] + width;
    }

    return ext;
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

void RectilinearMesh::setBoundaryCondition(int face, BoundaryCondition bc) {
    if (face < 0 || face > 5) {
        throw std::out_of_range("RectilinearMesh::setBoundaryCondition: face must be 0-5");
    }
    bc_[face] = bc;
}

void RectilinearMesh::applyBoundaryConditions(SolutionState& state) const {
    fillGhostX(state);
    if (dim_ >= 2) fillGhostY(state);
    if (dim_ >= 3) fillGhostZ(state);
}

// ---------------------------------------------------------------------------
// Ghost fill: x-direction
// ---------------------------------------------------------------------------

void RectilinearMesh::fillGhostX(SolutionState& state) const {
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int g = 1; g <= ngx_; ++g) {
                std::size_t ghost = index(-g, j, k);
                std::size_t src;
                double sU(1.0), sV(1.0), sW(1.0);

                switch (bc_[XLow]) {
                case BoundaryCondition::Symmetry:
                    src = index(g - 1, j, k);
                    break;
                case BoundaryCondition::Periodic:
                    src = index(nx_ - g, j, k);
                    break;
                case BoundaryCondition::SlipWall:
                    src = index(g - 1, j, k);
                    sU = -1.0;
                    break;
                case BoundaryCondition::NoSlipWall:
                    src = index(g - 1, j, k);
                    sU = -1.0; sV = -1.0; sW = -1.0;
                    break;
                case BoundaryCondition::Outflow:
                default:
                    src = index(0, j, k);
                    break;
                }
                state.copyCell(ghost, src, sU, sV, sW);
            }

            for (int g = 1; g <= ngx_; ++g) {
                std::size_t ghost = index(nx_ - 1 + g, j, k);
                std::size_t src;
                double sU(1.0), sV(1.0), sW(1.0);

                switch (bc_[XHigh]) {
                case BoundaryCondition::Symmetry:
                    src = index(nx_ - g, j, k);
                    break;
                case BoundaryCondition::Periodic:
                    src = index(g - 1, j, k);
                    break;
                case BoundaryCondition::SlipWall:
                    src = index(nx_ - g, j, k);
                    sU = -1.0;
                    break;
                case BoundaryCondition::NoSlipWall:
                    src = index(nx_ - g, j, k);
                    sU = -1.0; sV = -1.0; sW = -1.0;
                    break;
                case BoundaryCondition::Outflow:
                default:
                    src = index(nx_ - 1, j, k);
                    break;
                }
                state.copyCell(ghost, src, sU, sV, sW);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ghost fill: y-direction
// ---------------------------------------------------------------------------

void RectilinearMesh::fillGhostY(SolutionState& state) const {
    int iLo = -ngx_;
    int iHi = nx_ + ngx_;

    for (int k = 0; k < nz_; ++k) {
        for (int i = iLo; i < iHi; ++i) {
            for (int g = 1; g <= ngy_; ++g) {
                std::size_t ghost = index(i, -g, k);
                std::size_t src;
                double sU(1.0), sV(1.0), sW(1.0);

                switch (bc_[YLow]) {
                case BoundaryCondition::Symmetry:
                    src = index(i, g - 1, k);
                    break;
                case BoundaryCondition::Periodic:
                    src = index(i, ny_ - g, k);
                    break;
                case BoundaryCondition::SlipWall:
                    src = index(i, g - 1, k);
                    sV = -1.0;
                    break;
                case BoundaryCondition::NoSlipWall:
                    src = index(i, g - 1, k);
                    sU = -1.0; sV = -1.0; sW = -1.0;
                    break;
                case BoundaryCondition::Outflow:
                default:
                    src = index(i, 0, k);
                    break;
                }
                state.copyCell(ghost, src, sU, sV, sW);
            }

            for (int g = 1; g <= ngy_; ++g) {
                std::size_t ghost = index(i, ny_ - 1 + g, k);
                std::size_t src;
                double sU(1.0), sV(1.0), sW(1.0);

                switch (bc_[YHigh]) {
                case BoundaryCondition::Symmetry:
                    src = index(i, ny_ - g, k);
                    break;
                case BoundaryCondition::Periodic:
                    src = index(i, g - 1, k);
                    break;
                case BoundaryCondition::SlipWall:
                    src = index(i, ny_ - g, k);
                    sV = -1.0;
                    break;
                case BoundaryCondition::NoSlipWall:
                    src = index(i, ny_ - g, k);
                    sU = -1.0; sV = -1.0; sW = -1.0;
                    break;
                case BoundaryCondition::Outflow:
                default:
                    src = index(i, ny_ - 1, k);
                    break;
                }
                state.copyCell(ghost, src, sU, sV, sW);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ghost fill: z-direction
// ---------------------------------------------------------------------------

void RectilinearMesh::fillGhostZ(SolutionState& state) const {
    int iLo = -ngx_;
    int iHi = nx_ + ngx_;
    int jLo = -ngy_;
    int jHi = ny_ + ngy_;

    for (int j = jLo; j < jHi; ++j) {
        for (int i = iLo; i < iHi; ++i) {
            for (int g = 1; g <= ngz_; ++g) {
                std::size_t ghost = index(i, j, -g);
                std::size_t src;
                double sU(1.0), sV(1.0), sW(1.0);

                switch (bc_[ZLow]) {
                case BoundaryCondition::Symmetry:
                    src = index(i, j, g - 1);
                    break;
                case BoundaryCondition::Periodic:
                    src = index(i, j, nz_ - g);
                    break;
                case BoundaryCondition::SlipWall:
                    src = index(i, j, g - 1);
                    sW = -1.0;
                    break;
                case BoundaryCondition::NoSlipWall:
                    src = index(i, j, g - 1);
                    sU = -1.0; sV = -1.0; sW = -1.0;
                    break;
                case BoundaryCondition::Outflow:
                default:
                    src = index(i, j, 0);
                    break;
                }
                state.copyCell(ghost, src, sU, sV, sW);
            }

            for (int g = 1; g <= ngz_; ++g) {
                std::size_t ghost = index(i, j, nz_ - 1 + g);
                std::size_t src;
                double sU(1.0), sV(1.0), sW(1.0);

                switch (bc_[ZHigh]) {
                case BoundaryCondition::Symmetry:
                    src = index(i, j, nz_ - g);
                    break;
                case BoundaryCondition::Periodic:
                    src = index(i, j, g - 1);
                    break;
                case BoundaryCondition::SlipWall:
                    src = index(i, j, nz_ - g);
                    sW = -1.0;
                    break;
                case BoundaryCondition::NoSlipWall:
                    src = index(i, j, nz_ - g);
                    sU = -1.0; sV = -1.0; sW = -1.0;
                    break;
                case BoundaryCondition::Outflow:
                default:
                    src = index(i, j, nz_ - 1);
                    break;
                }
                state.copyCell(ghost, src, sU, sV, sW);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar ghost fill
// ---------------------------------------------------------------------------

void RectilinearMesh::fillScalarGhosts(std::vector<double>& field) const {
    // X-direction (physical j,k only)
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int g = 1; g <= ngx_; ++g) {
                std::size_t ghost = index(-g, j, k);
                if (bc_[XLow] == BoundaryCondition::Periodic) {
                    field[ghost] = field[index(nx_ - g, j, k)];
                } else {
                    field[ghost] = field[index(0, j, k)];
                }
                ghost = index(nx_ - 1 + g, j, k);
                if (bc_[XHigh] == BoundaryCondition::Periodic) {
                    field[ghost] = field[index(g - 1, j, k)];
                } else {
                    field[ghost] = field[index(nx_ - 1, j, k)];
                }
            }
        }
    }

    if (dim_ >= 2) {
        int iLo = -ngx_;
        int iHi = nx_ + ngx_;
        for (int k = 0; k < nz_; ++k) {
            for (int i = iLo; i < iHi; ++i) {
                for (int g = 1; g <= ngy_; ++g) {
                    std::size_t ghost = index(i, -g, k);
                    if (bc_[YLow] == BoundaryCondition::Periodic) {
                        field[ghost] = field[index(i, ny_ - g, k)];
                    } else {
                        field[ghost] = field[index(i, 0, k)];
                    }
                    ghost = index(i, ny_ - 1 + g, k);
                    if (bc_[YHigh] == BoundaryCondition::Periodic) {
                        field[ghost] = field[index(i, g - 1, k)];
                    } else {
                        field[ghost] = field[index(i, ny_ - 1, k)];
                    }
                }
            }
        }
    }

    if (dim_ >= 3) {
        int iLo = -ngx_;
        int iHi = nx_ + ngx_;
        int jLo = -ngy_;
        int jHi = ny_ + ngy_;
        for (int j = jLo; j < jHi; ++j) {
            for (int i = iLo; i < iHi; ++i) {
                for (int g = 1; g <= ngz_; ++g) {
                    std::size_t ghost = index(i, j, -g);
                    if (bc_[ZLow] == BoundaryCondition::Periodic) {
                        field[ghost] = field[index(i, j, nz_ - g)];
                    } else {
                        field[ghost] = field[index(i, j, 0)];
                    }
                    ghost = index(i, j, nz_ - 1 + g);
                    if (bc_[ZHigh] == BoundaryCondition::Periodic) {
                        field[ghost] = field[index(i, j, g - 1)];
                    } else {
                        field[ghost] = field[index(i, j, nz_ - 1)];
                    }
                }
            }
        }
    }
}

} // namespace SemiImplicitFV

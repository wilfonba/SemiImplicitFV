#include "ImmersedBoundary.hpp"
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {

// ---- IBCircle ----

IBCircle::IBCircle(double cx, double cy, double r)
    : cx_(cx), cy_(cy), r_(r) {}

double IBCircle::signedDistance(double x, double y, double /*z*/) const {
    double dx = x - cx_;
    double dy = y - cy_;
    return std::sqrt(dx * dx + dy * dy) - r_;
}

std::array<double,3> IBCircle::closestPoint(double x, double y, double /*z*/) const {
    double dx = x - cx_;
    double dy = y - cy_;
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist < 1e-14) {
        return {cx_ + r_, cy_, 0.0};
    }
    double s = r_ / dist;
    return {cx_ + dx * s, cy_ + dy * s, 0.0};
}

std::array<double,3> IBCircle::outwardNormal(double x, double y, double /*z*/) const {
    double dx = x - cx_;
    double dy = y - cy_;
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist < 1e-14) {
        return {1.0, 0.0, 0.0};
    }
    return {dx / dist, dy / dist, 0.0};
}

// ---- IBRectangle ----

IBRectangle::IBRectangle(double cx, double cy, double hw, double hh)
    : cx_(cx), cy_(cy), hw_(hw), hh_(hh) {}

double IBRectangle::signedDistance(double x, double y, double /*z*/) const {
    double qx = std::abs(x - cx_) - hw_;
    double qy = std::abs(y - cy_) - hh_;
    double outside = std::sqrt(std::max(qx, 0.0) * std::max(qx, 0.0) +
                               std::max(qy, 0.0) * std::max(qy, 0.0));
    double inside = std::min(std::max(qx, qy), 0.0);
    return outside + inside;
}

std::array<double,3> IBRectangle::closestPoint(double x, double y, double /*z*/) const {
    double dx = x - cx_;
    double dy = y - cy_;

    double clampX = std::clamp(dx, -hw_, hw_);
    double clampY = std::clamp(dy, -hh_, hh_);

    // If the point is inside the rectangle, project to the nearest face
    if (std::abs(dx) <= hw_ && std::abs(dy) <= hh_) {
        double distToX = hw_ - std::abs(dx);
        double distToY = hh_ - std::abs(dy);
        if (distToX < distToY) {
            clampX = (dx >= 0) ? hw_ : -hw_;
        } else {
            clampY = (dy >= 0) ? hh_ : -hh_;
        }
    }

    return {cx_ + clampX, cy_ + clampY, 0.0};
}

std::array<double,3> IBRectangle::outwardNormal(double x, double y, double z) const {
    auto cp = closestPoint(x, y, z);
    double dx = cp[0] - cx_;
    double dy = cp[1] - cy_;

    bool onX = std::abs(std::abs(dx) - hw_) < 1e-12;
    bool onY = std::abs(std::abs(dy) - hh_) < 1e-12;

    if (onX && !onY) {
        return {(dx >= 0) ? 1.0 : -1.0, 0.0, 0.0};
    } else if (onY && !onX) {
        return {0.0, (dy >= 0) ? 1.0 : -1.0, 0.0};
    } else {
        // Corner: diagonal normal
        double nx = (dx >= 0) ? 1.0 : -1.0;
        double ny = (dy >= 0) ? 1.0 : -1.0;
        double len = std::sqrt(2.0);
        return {nx / len, ny / len, 0.0};
    }
}

// ---- IBCylinder ----

IBCylinder::IBCylinder(double c0, double c1, double r, int axis)
    : c0_(c0), c1_(c1), r_(r), axis_(axis) {}

double IBCylinder::signedDistance(double x, double y, double z) const {
    double p0, p1;
    if (axis_ == 0)      { p0 = y; p1 = z; }
    else if (axis_ == 1) { p0 = x; p1 = z; }
    else                 { p0 = x; p1 = y; }

    double d0 = p0 - c0_;
    double d1 = p1 - c1_;
    return std::sqrt(d0 * d0 + d1 * d1) - r_;
}

std::array<double,3> IBCylinder::closestPoint(double x, double y, double z) const {
    double p0, p1;
    if (axis_ == 0)      { p0 = y; p1 = z; }
    else if (axis_ == 1) { p0 = x; p1 = z; }
    else                 { p0 = x; p1 = y; }

    double d0 = p0 - c0_;
    double d1 = p1 - c1_;
    double dist = std::sqrt(d0 * d0 + d1 * d1);

    double cp0, cp1;
    if (dist < 1e-14) {
        cp0 = c0_ + r_;
        cp1 = c1_;
    } else {
        double s = r_ / dist;
        cp0 = c0_ + d0 * s;
        cp1 = c1_ + d1 * s;
    }

    if (axis_ == 0)      return {x,   cp0, cp1};
    else if (axis_ == 1) return {cp0, y,   cp1};
    else                 return {cp0, cp1, z};
}

std::array<double,3> IBCylinder::outwardNormal(double x, double y, double z) const {
    double p0, p1;
    if (axis_ == 0)      { p0 = y; p1 = z; }
    else if (axis_ == 1) { p0 = x; p1 = z; }
    else                 { p0 = x; p1 = y; }

    double d0 = p0 - c0_;
    double d1 = p1 - c1_;
    double dist = std::sqrt(d0 * d0 + d1 * d1);

    double n0, n1;
    if (dist < 1e-14) {
        n0 = 1.0; n1 = 0.0;
    } else {
        n0 = d0 / dist;
        n1 = d1 / dist;
    }

    if (axis_ == 0)      return {0.0, n0,  n1};
    else if (axis_ == 1) return {n0,  0.0, n1};
    else                 return {n0,  n1,  0.0};
}

// ---- IBRectangularPrism ----

IBRectangularPrism::IBRectangularPrism(double cx, double cy, double cz,
                                       double hw, double hh, double hd)
    : cx_(cx), cy_(cy), cz_(cz), hw_(hw), hh_(hh), hd_(hd) {}

double IBRectangularPrism::signedDistance(double x, double y, double z) const {
    double qx = std::abs(x - cx_) - hw_;
    double qy = std::abs(y - cy_) - hh_;
    double qz = std::abs(z - cz_) - hd_;
    double outside = std::sqrt(std::max(qx, 0.0) * std::max(qx, 0.0) +
                               std::max(qy, 0.0) * std::max(qy, 0.0) +
                               std::max(qz, 0.0) * std::max(qz, 0.0));
    double inside = std::min(std::max({qx, qy, qz}), 0.0);
    return outside + inside;
}

std::array<double,3> IBRectangularPrism::closestPoint(double x, double y, double z) const {
    double dx = x - cx_;
    double dy = y - cy_;
    double dz = z - cz_;

    double clampX = std::clamp(dx, -hw_, hw_);
    double clampY = std::clamp(dy, -hh_, hh_);
    double clampZ = std::clamp(dz, -hd_, hd_);

    if (std::abs(dx) <= hw_ && std::abs(dy) <= hh_ && std::abs(dz) <= hd_) {
        double distX = hw_ - std::abs(dx);
        double distY = hh_ - std::abs(dy);
        double distZ = hd_ - std::abs(dz);
        if (distX <= distY && distX <= distZ) {
            clampX = (dx >= 0) ? hw_ : -hw_;
        } else if (distY <= distX && distY <= distZ) {
            clampY = (dy >= 0) ? hh_ : -hh_;
        } else {
            clampZ = (dz >= 0) ? hd_ : -hd_;
        }
    }

    return {cx_ + clampX, cy_ + clampY, cz_ + clampZ};
}

std::array<double,3> IBRectangularPrism::outwardNormal(double x, double y, double z) const {
    auto cp = closestPoint(x, y, z);
    double dx = cp[0] - cx_;
    double dy = cp[1] - cy_;
    double dz = cp[2] - cz_;

    double nx = 0.0, ny = 0.0, nz = 0.0;
    if (std::abs(std::abs(dx) - hw_) < 1e-12) nx = (dx >= 0) ? 1.0 : -1.0;
    if (std::abs(std::abs(dy) - hh_) < 1e-12) ny = (dy >= 0) ? 1.0 : -1.0;
    if (std::abs(std::abs(dz) - hd_) < 1e-12) nz = (dz >= 0) ? 1.0 : -1.0;

    double len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len < 1e-14) return {1.0, 0.0, 0.0};
    return {nx / len, ny / len, nz / len};
}

// ---- ImmersedBoundaryMethod ----

void ImmersedBoundaryMethod::addBody(std::shared_ptr<IBBody> body) {
    bodies_.push_back(std::move(body));
}

int ImmersedBoundaryMethod::findCell(const RectilinearMesh& mesh,
                                     double coord, int dir) {
    int n;
    if (dir == 0)      n = mesh.nx();
    else if (dir == 1) n = mesh.ny();
    else               n = mesh.nz();

    if (n <= 1) return 0;

    auto centroid = [&](int i) -> double {
        if (dir == 0)      return mesh.cellCentroidX(i);
        else if (dir == 1) return mesh.cellCentroidY(i);
        else               return mesh.cellCentroidZ(i);
    };

    if (coord <= centroid(0)) return 0;
    if (coord >= centroid(n - 1)) return std::max(0, n - 2);

    for (int i = 0; i < n - 1; ++i) {
        if (coord >= centroid(i) && coord < centroid(i + 1))
            return i;
    }

    return std::max(0, n - 2);
}

void ImmersedBoundaryMethod::buildInterpStencil(
    const RectilinearMesh& mesh,
    double xIP, double yIP, double zIP,
    int dim,
    std::vector<std::size_t>& cells,
    std::vector<double>& weights) const
{
    int i0 = findCell(mesh, xIP, 0);
    int j0 = (dim >= 2) ? findCell(mesh, yIP, 1) : 0;
    int k0 = (dim >= 3) ? findCell(mesh, zIP, 2) : 0;

    double cx0 = mesh.cellCentroidX(i0);
    double cx1 = mesh.cellCentroidX(std::min(i0 + 1, mesh.nx() - 1));
    double dxc = cx1 - cx0;
    double tx = (dxc > 1e-14) ? std::clamp((xIP - cx0) / dxc, 0.0, 1.0) : 0.0;

    if (dim == 1) {
        int i1 = std::min(i0 + 1, mesh.nx() - 1);
        cells  = {mesh.index(i0, 0, 0), mesh.index(i1, 0, 0)};
        weights = {1.0 - tx, tx};
    } else if (dim == 2) {
        int i1 = std::min(i0 + 1, mesh.nx() - 1);
        int j1 = std::min(j0 + 1, mesh.ny() - 1);

        double cy0 = mesh.cellCentroidY(j0);
        double cy1 = mesh.cellCentroidY(j1);
        double dyc = cy1 - cy0;
        double ty = (dyc > 1e-14) ? std::clamp((yIP - cy0) / dyc, 0.0, 1.0) : 0.0;

        cells = {
            mesh.index(i0, j0, 0),
            mesh.index(i1, j0, 0),
            mesh.index(i0, j1, 0),
            mesh.index(i1, j1, 0)
        };
        weights = {
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            (1.0 - tx) * ty,
            tx * ty
        };
    } else {
        int i1 = std::min(i0 + 1, mesh.nx() - 1);
        int j1 = std::min(j0 + 1, mesh.ny() - 1);
        int k1 = std::min(k0 + 1, mesh.nz() - 1);

        double cy0 = mesh.cellCentroidY(j0);
        double cy1 = mesh.cellCentroidY(j1);
        double dyc = cy1 - cy0;
        double ty = (dyc > 1e-14) ? std::clamp((yIP - cy0) / dyc, 0.0, 1.0) : 0.0;

        double cz0 = mesh.cellCentroidZ(k0);
        double cz1 = mesh.cellCentroidZ(k1);
        double dzc = cz1 - cz0;
        double tz = (dzc > 1e-14) ? std::clamp((zIP - cz0) / dzc, 0.0, 1.0) : 0.0;

        cells = {
            mesh.index(i0, j0, k0),
            mesh.index(i1, j0, k0),
            mesh.index(i0, j1, k0),
            mesh.index(i1, j1, k0),
            mesh.index(i0, j0, k1),
            mesh.index(i1, j0, k1),
            mesh.index(i0, j1, k1),
            mesh.index(i1, j1, k1)
        };
        weights = {
            (1.0 - tx) * (1.0 - ty) * (1.0 - tz),
            tx * (1.0 - ty) * (1.0 - tz),
            (1.0 - tx) * ty * (1.0 - tz),
            tx * ty * (1.0 - tz),
            (1.0 - tx) * (1.0 - ty) * tz,
            tx * (1.0 - ty) * tz,
            (1.0 - tx) * ty * tz,
            tx * ty * tz
        };
    }

    // Validate: if any stencil cell is not Fluid, fall back to nearest fluid cell
    bool allFluid = true;
    for (auto c : cells) {
        if (cellType_[c] != CellType::Fluid) {
            allFluid = false;
            break;
        }
    }

    if (!allFluid) {
        std::size_t fluidCell = cells[0];
        for (auto c : cells) {
            if (cellType_[c] == CellType::Fluid) {
                fluidCell = c;
                break;
            }
        }
        cells = {fluidCell};
        weights = {1.0};
    }
}

void ImmersedBoundaryMethod::classifyCells(const RectilinearMesh& mesh) {
    std::size_t nTotal = mesh.totalCells();
    cellType_.assign(nTotal, CellType::Fluid);

    // Track which body each cell is inside (-1 = none)
    std::vector<int> bodyOwner(nTotal, -1);
    int dim = mesh.dim();

    // First pass: mark cells inside bodies
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                double cx = mesh.cellCentroidX(i);
                double cy = mesh.cellCentroidY(j);
                double cz = mesh.cellCentroidZ(k);

                for (int b = 0; b < static_cast<int>(bodies_.size()); ++b) {
                    if (bodies_[b]->signedDistance(cx, cy, cz) < 0) {
                        cellType_[idx] = CellType::Dead;
                        bodyOwner[idx] = b;
                        break;
                    }
                }
            }
        }
    }

    // Second pass: solid cells adjacent to fluid become Ghost
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                if (cellType_[idx] != CellType::Dead) continue;

                bool hasFluidNeighbor = false;
                if (i > 0 && cellType_[mesh.index(i - 1, j, k)] == CellType::Fluid)
                    hasFluidNeighbor = true;
                if (i < mesh.nx() - 1 && cellType_[mesh.index(i + 1, j, k)] == CellType::Fluid)
                    hasFluidNeighbor = true;
                if (dim >= 2) {
                    if (j > 0 && cellType_[mesh.index(i, j - 1, k)] == CellType::Fluid)
                        hasFluidNeighbor = true;
                    if (j < mesh.ny() - 1 && cellType_[mesh.index(i, j + 1, k)] == CellType::Fluid)
                        hasFluidNeighbor = true;
                }
                if (dim >= 3) {
                    if (k > 0 && cellType_[mesh.index(i, j, k - 1)] == CellType::Fluid)
                        hasFluidNeighbor = true;
                    if (k < mesh.nz() - 1 && cellType_[mesh.index(i, j, k + 1)] == CellType::Fluid)
                        hasFluidNeighbor = true;
                }

                if (hasFluidNeighbor)
                    cellType_[idx] = CellType::Ghost;
            }
        }
    }

    // Build GhostCellInfo for each Ghost cell
    ghostCells_.clear();
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                if (cellType_[idx] != CellType::Ghost) continue;

                int bIdx = bodyOwner[idx];
                const auto& body = bodies_[bIdx];

                double xG = mesh.cellCentroidX(i);
                double yG = mesh.cellCentroidY(j);
                double zG = mesh.cellCentroidZ(k);

                auto cp = body->closestPoint(xG, yG, zG);
                auto n  = body->outwardNormal(xG, yG, zG);

                // Image point: reflection of ghost cell through the surface
                double xIP = 2.0 * cp[0] - xG;
                double yIP = 2.0 * cp[1] - yG;
                double zIP = 2.0 * cp[2] - zG;

                GhostCellInfo info;
                info.cellIdx = idx;
                info.bodyIdx = bIdx;
                info.normal  = n;

                buildInterpStencil(mesh, xIP, yIP, zIP, dim,
                                   info.interpCells, info.interpWeights);

                ghostCells_.push_back(std::move(info));
            }
        }
    }
}

void ImmersedBoundaryMethod::applyGhostCells(
    const RectilinearMesh& /*mesh*/, SolutionState& state, int dim) const
{
    const int nAlpha    = static_cast<int>(state.alpha.size());
    const int nAlphaRho = static_cast<int>(state.alphaRho.size());

    for (const auto& gc : ghostCells_) {
        std::size_t idx = gc.cellIdx;
        const auto& cells = gc.interpCells;
        const auto& w     = gc.interpWeights;

        // Interpolate primitives at image point
        double rho_IP  = 0.0, velU_IP = 0.0, velV_IP = 0.0, velW_IP = 0.0;
        double pres_IP = 0.0, sigma_IP = 0.0;
        double rhoE_IP = 0.0;

        for (std::size_t s = 0; s < cells.size(); ++s) {
            std::size_t c = cells[s];
            rho_IP  += w[s] * state.rho[c];
            velU_IP += w[s] * state.velU[c];
            if (dim >= 2) velV_IP += w[s] * state.velV[c];
            if (dim >= 3) velW_IP += w[s] * state.velW[c];
            pres_IP  += w[s] * state.pres[c];
            sigma_IP += w[s] * state.sigma[c];
            rhoE_IP  += w[s] * state.rhoE[c];
        }

        // Scalars: copy from image point
        state.rho[idx]   = rho_IP;
        state.pres[idx]  = pres_IP;
        state.sigma[idx] = sigma_IP;
        state.rhoE[idx]  = rhoE_IP;

        // Velocity: apply wall BC
        double gU, gV, gW;
        IBBody::WallType wt = bodies_[gc.bodyIdx]->wallType();

        if (wt == IBBody::WallType::NoSlip) {
            // Linear interpolation gives v_wall = (v_ghost + v_IP)/2 = 0
            gU = -velU_IP;
            gV = -velV_IP;
            gW = -velW_IP;
        } else {
            // Slip: reflect normal component, keep tangential
            double dot = velU_IP * gc.normal[0];
            if (dim >= 2) dot += velV_IP * gc.normal[1];
            if (dim >= 3) dot += velW_IP * gc.normal[2];

            gU = velU_IP - 2.0 * dot * gc.normal[0];
            gV = velV_IP - 2.0 * dot * gc.normal[1];
            gW = velW_IP - 2.0 * dot * gc.normal[2];
        }

        state.velU[idx] = gU;
        if (dim >= 2) state.velV[idx] = gV;
        if (dim >= 3) state.velW[idx] = gW;

        // Conservative momentum (consistent with primitives)
        state.rhoU[idx] = rho_IP * gU;
        if (dim >= 2) state.rhoV[idx] = rho_IP * gV;
        if (dim >= 3) state.rhoW[idx] = rho_IP * gW;

        // Multi-phase fields
        for (int ph = 0; ph < nAlphaRho; ++ph) {
            double val = 0.0;
            for (std::size_t s = 0; s < cells.size(); ++s)
                val += w[s] * state.alphaRho[ph][cells[s]];
            state.alphaRho[ph][idx] = val;
        }
        for (int ph = 0; ph < nAlpha; ++ph) {
            double val = 0.0;
            for (std::size_t s = 0; s < cells.size(); ++s)
                val += w[s] * state.alpha[ph][cells[s]];
            state.alpha[ph][idx] = val;
        }
    }
}

void ImmersedBoundaryMethod::applyScalarGhostCells(
    const RectilinearMesh& /*mesh*/, std::vector<double>& field) const
{
    for (const auto& gc : ghostCells_) {
        double val = 0.0;
        for (std::size_t s = 0; s < gc.interpCells.size(); ++s)
            val += gc.interpWeights[s] * field[gc.interpCells[s]];
        field[gc.cellIdx] = val;
    }
}

} // namespace SemiImplicitFV

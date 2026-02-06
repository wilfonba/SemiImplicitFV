#include "SemiImplicitSolver.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace SemiImplicitFV {

// =============================================================================
// Pressure Laplacian helper (shared by Jacobi and Gauss-Seidel)
// =============================================================================

namespace {

// Compute the Laplacian coefficients and off-diagonal sum for a single cell.
// Returns the diagonal coefficient (sum of all neighbor coefficients).
// offDiag is accumulated with coeff_n * pressure[neighbor].
double pressureLaplacian(
    const RectilinearMesh& mesh,
    const std::vector<double>& pressure,
    int i, int j, int k,
    double& offDiag)
{
    offDiag = 0.0;
    double diagCoeff = 0.0;
    std::size_t idx = mesh.index(i, j, k);

    // X-direction
    {
        std::size_t xm = mesh.index(i - 1, j, k);
        std::size_t xp = mesh.index(i + 1, j, k);
        double rhoL = 0.5 * (mesh.rho[idx] + mesh.rho[xm]);
        double rhoR = 0.5 * (mesh.rho[idx] + mesh.rho[xp]);
        double dL = 0.5 * (mesh.dx(i - 1) + mesh.dx(i));
        double dR = 0.5 * (mesh.dx(i) + mesh.dx(i + 1));
        double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh.dx(i));
        double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh.dx(i));
        offDiag += cL * pressure[xm] + cR * pressure[xp];
        diagCoeff += cL + cR;
    }

    // Y-direction
    if (mesh.dim() >= 2) {
        std::size_t ym = mesh.index(i, j - 1, k);
        std::size_t yp = mesh.index(i, j + 1, k);
        double rhoL = 0.5 * (mesh.rho[idx] + mesh.rho[ym]);
        double rhoR = 0.5 * (mesh.rho[idx] + mesh.rho[yp]);
        double dL = 0.5 * (mesh.dy(j - 1) + mesh.dy(j));
        double dR = 0.5 * (mesh.dy(j) + mesh.dy(j + 1));
        double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh.dy(j));
        double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh.dy(j));
        offDiag += cL * pressure[ym] + cR * pressure[yp];
        diagCoeff += cL + cR;
    }

    // Z-direction
    if (mesh.dim() >= 3) {
        std::size_t zm = mesh.index(i, j, k - 1);
        std::size_t zp = mesh.index(i, j, k + 1);
        double rhoL = 0.5 * (mesh.rho[idx] + mesh.rho[zm]);
        double rhoR = 0.5 * (mesh.rho[idx] + mesh.rho[zp]);
        double dL = 0.5 * (mesh.dz(k - 1) + mesh.dz(k));
        double dR = 0.5 * (mesh.dz(k) + mesh.dz(k + 1));
        double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh.dz(k));
        double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh.dz(k));
        offDiag += cL * pressure[zm] + cR * pressure[zp];
        diagCoeff += cL + cR;
    }

    return diagCoeff;
}

} // anonymous namespace

// =============================================================================
// Jacobi Pressure Solver
// =============================================================================

int JacobiPressureSolver::solve(
    RectilinearMesh& mesh,
    const std::vector<double>& rhoc2,
    const std::vector<double>& rhs,
    std::vector<double>& pressure,
    double dt,
    double tolerance,
    int maxIter
) {
    std::vector<double> pNew(pressure.size(), 0.0);
    double dt2 = dt * dt;

    for (int iter = 0; iter < maxIter; ++iter) {
        mesh.fillScalarGhosts(pressure);
        double maxResidual = 0.0;

        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idx = mesh.index(i, j, k);
                    double coeff = rhoc2[idx] * dt2;

                    double offDiag;
                    double diagL = pressureLaplacian(mesh, pressure, i, j, k, offDiag);

                    double denom = 1.0 + coeff * diagL;
                    pNew[idx] = (rhs[idx] + coeff * offDiag) / denom;

                    double residual = std::abs(pNew[idx] - pressure[idx]);
                    maxResidual = std::max(maxResidual, residual);
                }
            }
        }

        // Copy new values into pressure (physical cells only)
        for (int k = 0; k < mesh.nz(); ++k)
            for (int j = 0; j < mesh.ny(); ++j)
                for (int i = 0; i < mesh.nx(); ++i)
                    pressure[mesh.index(i, j, k)] = pNew[mesh.index(i, j, k)];

        if (maxResidual < tolerance) {
            return iter + 1;
        }
    }

    return maxIter;
}

// =============================================================================
// Gauss-Seidel Pressure Solver
// =============================================================================

int GaussSeidelPressureSolver::solve(
    RectilinearMesh& mesh,
    const std::vector<double>& rhoc2,
    const std::vector<double>& rhs,
    std::vector<double>& pressure,
    double dt,
    double tolerance,
    int maxIter
) {
    double dt2 = dt * dt;

    for (int iter = 0; iter < maxIter; ++iter) {
        mesh.fillScalarGhosts(pressure);
        double maxResidual = 0.0;

        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idx = mesh.index(i, j, k);
                    double coeff = rhoc2[idx] * dt2;

                    double offDiag;
                    double diagL = pressureLaplacian(mesh, pressure, i, j, k, offDiag);

                    double denom = 1.0 + coeff * diagL;
                    double pOld = pressure[idx];
                    pressure[idx] = (rhs[idx] + coeff * offDiag) / denom;

                    double residual = std::abs(pressure[idx] - pOld);
                    maxResidual = std::max(maxResidual, residual);
                }
            }
        }

        if (maxResidual < tolerance) {
            return iter + 1;
        }
    }

    return maxIter;
}

// =============================================================================
// Semi-Implicit Solver
// =============================================================================

SemiImplicitSolver::SemiImplicitSolver(
    std::shared_ptr<RiemannSolver> riemannSolver,
    std::shared_ptr<PressureSolver> pressureSolver,
    std::shared_ptr<EquationOfStateBase> eos,
    std::shared_ptr<IGRSolver> igrSolver,
    const SemiImplicitParams& params
)
    : riemannSolver_(std::move(riemannSolver))
    , pressureSolver_(std::move(pressureSolver))
    , eos_(std::move(eos))
    , igrSolver_(std::move(igrSolver))
    , params_(params)
    , lastPressureIters_(0)
{}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

PrimitiveState SemiImplicitSolver::gatherPrimitive(
    const RectilinearMesh& mesh, std::size_t idx)
{
    PrimitiveState W;
    W.rho = mesh.rho[idx];
    W.u[0] = mesh.velU[idx];
    W.u[1] = mesh.velV[idx];
    W.u[2] = mesh.velW[idx];
    W.p = mesh.pres[idx];
    W.T = mesh.temp[idx];
    W.sigma = mesh.sigma[idx];
    return W;
}

void SemiImplicitSolver::ensureStorage(const RectilinearMesh& mesh) {
    std::size_t n = mesh.totalCells();
    if (rhoStar_.size() == n) return;

    rhoStar_.resize(n);
    rhoUStar_.resize(n);
    rhoVStar_.resize(n);
    rhoWStar_.resize(n);
    rhoEStar_.resize(n);
    pAdvected_.resize(n);
    rhoc2_.resize(n);
    pressureRhs_.resize(n);
    pressure_.resize(n);
    divUstar_.resize(n);
    if (igrSolver_) {
        gradU_.resize(n);
    }
}

void SemiImplicitSolver::writeStarToMesh(RectilinearMesh& mesh) {
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                mesh.rho[idx]  = rhoStar_[idx];
                mesh.rhoU[idx] = rhoUStar_[idx];
                mesh.rhoV[idx] = rhoVStar_[idx];
                mesh.rhoW[idx] = rhoWStar_[idx];
                mesh.rhoE[idx] = rhoEStar_[idx];

                // Convert to primitives
                ConservativeState U;
                U.rho = rhoStar_[idx];
                U.rhoU = {rhoUStar_[idx], rhoVStar_[idx], rhoWStar_[idx]};
                U.rhoE = rhoEStar_[idx];
                PrimitiveState W = eos_->toPrimitive(U);
                mesh.velU[idx] = W.u[0];
                mesh.velV[idx] = W.u[1];
                mesh.velW[idx] = W.u[2];
                mesh.pres[idx] = W.p;
                mesh.temp[idx] = W.T;
                // Don't overwrite sigma — preserved for IGR warm start
            }
        }
    }
    mesh.applyBoundaryConditions();
}

// ---------------------------------------------------------------------------
// Main step
// ---------------------------------------------------------------------------

double SemiImplicitSolver::step(RectilinearMesh& mesh, double targetDt) {
    double dt = computeTimeStep(mesh);
    if (targetDt > 0) {
        dt = std::min(dt, targetDt);
    }
    dt = std::clamp(dt, params_.minDt, params_.maxDt);

    ensureStorage(mesh);

    // Fill ghost cells for current state
    mesh.applyBoundaryConditions();

    // Step 1: Explicit advection (pressure-free Riemann solver)
    advectionStep(mesh, dt);

    // Step 2: Advect pressure using u^n, p^n (mesh still holds original state)
    advectPressure(mesh, dt);

    // Write star state to mesh and convert to primitives for steps 3-5
    writeStarToMesh(mesh);

    // Step 3: IGR — compute entropic pressure Σ (optional)
    if (params_.useIGR && igrSolver_) {
        solveIGR(mesh);
    }

    // Step 4: Solve pressure equation
    solvePressure(mesh, dt);

    // Step 5: Correct momentum and energy with pressure gradient
    correctionStep(mesh, dt);

    // Final update: convert corrected conservatives to primitives
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                ConservativeState U;
                U.rho = mesh.rho[idx];
                U.rhoU = {mesh.rhoU[idx], mesh.rhoV[idx], mesh.rhoW[idx]};
                U.rhoE = mesh.rhoE[idx];
                PrimitiveState W = eos_->toPrimitive(U);
                mesh.velU[idx] = W.u[0];
                mesh.velV[idx] = W.u[1];
                mesh.velW[idx] = W.u[2];
                mesh.pres[idx] = W.p;
                mesh.temp[idx] = W.T;
            }
        }
    }

    return dt;
}

// ---------------------------------------------------------------------------
// Time step computation
// ---------------------------------------------------------------------------

double SemiImplicitSolver::computeTimeStep(const RectilinearMesh& mesh) const {
    double maxSpeed = 0.0;
    double minDx = std::numeric_limits<double>::max();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                double dxMin = mesh.dx(i);
                if (mesh.dim() >= 2) dxMin = std::min(dxMin, mesh.dy(j));
                if (mesh.dim() >= 3) dxMin = std::min(dxMin, mesh.dz(k));
                minDx = std::min(minDx, dxMin);

                std::size_t idx = mesh.index(i, j, k);
                double u = std::sqrt(
                    mesh.velU[idx] * mesh.velU[idx] +
                    mesh.velV[idx] * mesh.velV[idx] +
                    mesh.velW[idx] * mesh.velW[idx]);
                maxSpeed = std::max(maxSpeed, u);
            }
        }
    }

    if (maxSpeed < 1e-14) return params_.maxDt;
    return params_.cfl * minDx / maxSpeed;
}

// ---------------------------------------------------------------------------
// Step 1: Advection (pressure-free fluxes)
// ---------------------------------------------------------------------------

void SemiImplicitSolver::advectionStep(RectilinearMesh& mesh, double dt) {
    // Initialize star state from current conservatives
    std::copy(mesh.rho.begin(),  mesh.rho.end(),  rhoStar_.begin());
    std::copy(mesh.rhoU.begin(), mesh.rhoU.end(), rhoUStar_.begin());
    std::copy(mesh.rhoV.begin(), mesh.rhoV.end(), rhoVStar_.begin());
    std::copy(mesh.rhoW.begin(), mesh.rhoW.end(), rhoWStar_.begin());
    std::copy(mesh.rhoE.begin(), mesh.rhoE.end(), rhoEStar_.begin());

    // --- X-direction fluxes ---
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i <= mesh.nx(); ++i) {
                // Face between cells (i-1,j,k) and (i,j,k)
                std::size_t idxL = mesh.index(i - 1, j, k);
                std::size_t idxR = mesh.index(i, j, k);

                PrimitiveState left  = gatherPrimitive(mesh, idxL);
                PrimitiveState right = gatherPrimitive(mesh, idxR);

                RiemannFlux flux = riemannSolver_->computeFlux(
                    left, right, {1.0, 0.0, 0.0});

                double area = mesh.faceAreaX(j, k);

                // Update left cell if physical
                if (i >= 1) {
                    double coeff = dt * area / mesh.cellVolume(i - 1, j, k);
                    rhoStar_[idxL]  -= coeff * flux.massFlux;
                    rhoUStar_[idxL] -= coeff * flux.momentumFlux[0];
                    rhoVStar_[idxL] -= coeff * flux.momentumFlux[1];
                    rhoWStar_[idxL] -= coeff * flux.momentumFlux[2];
                    rhoEStar_[idxL] -= coeff * flux.energyFlux;
                }

                // Update right cell if physical
                if (i < mesh.nx()) {
                    double coeff = dt * area / mesh.cellVolume(i, j, k);
                    rhoStar_[idxR]  += coeff * flux.massFlux;
                    rhoUStar_[idxR] += coeff * flux.momentumFlux[0];
                    rhoVStar_[idxR] += coeff * flux.momentumFlux[1];
                    rhoWStar_[idxR] += coeff * flux.momentumFlux[2];
                    rhoEStar_[idxR] += coeff * flux.energyFlux;
                }
            }
        }
    }

    // --- Y-direction fluxes ---
    if (mesh.dim() >= 2) {
        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j <= mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idxL = mesh.index(i, j - 1, k);
                    std::size_t idxR = mesh.index(i, j, k);

                    PrimitiveState left  = gatherPrimitive(mesh, idxL);
                    PrimitiveState right = gatherPrimitive(mesh, idxR);

                    RiemannFlux flux = riemannSolver_->computeFlux(
                        left, right, {0.0, 1.0, 0.0});

                    double area = mesh.faceAreaY(i, k);

                    if (j >= 1) {
                        double coeff = dt * area / mesh.cellVolume(i, j - 1, k);
                        rhoStar_[idxL]  -= coeff * flux.massFlux;
                        rhoUStar_[idxL] -= coeff * flux.momentumFlux[0];
                        rhoVStar_[idxL] -= coeff * flux.momentumFlux[1];
                        rhoWStar_[idxL] -= coeff * flux.momentumFlux[2];
                        rhoEStar_[idxL] -= coeff * flux.energyFlux;
                    }

                    if (j < mesh.ny()) {
                        double coeff = dt * area / mesh.cellVolume(i, j, k);
                        rhoStar_[idxR]  += coeff * flux.massFlux;
                        rhoUStar_[idxR] += coeff * flux.momentumFlux[0];
                        rhoVStar_[idxR] += coeff * flux.momentumFlux[1];
                        rhoWStar_[idxR] += coeff * flux.momentumFlux[2];
                        rhoEStar_[idxR] += coeff * flux.energyFlux;
                    }
                }
            }
        }
    }

    // --- Z-direction fluxes ---
    if (mesh.dim() >= 3) {
        for (int k = 0; k <= mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idxL = mesh.index(i, j, k - 1);
                    std::size_t idxR = mesh.index(i, j, k);

                    PrimitiveState left  = gatherPrimitive(mesh, idxL);
                    PrimitiveState right = gatherPrimitive(mesh, idxR);

                    RiemannFlux flux = riemannSolver_->computeFlux(
                        left, right, {0.0, 0.0, 1.0});

                    double area = mesh.faceAreaZ(i, j);

                    if (k >= 1) {
                        double coeff = dt * area / mesh.cellVolume(i, j, k - 1);
                        rhoStar_[idxL]  -= coeff * flux.massFlux;
                        rhoUStar_[idxL] -= coeff * flux.momentumFlux[0];
                        rhoVStar_[idxL] -= coeff * flux.momentumFlux[1];
                        rhoWStar_[idxL] -= coeff * flux.momentumFlux[2];
                        rhoEStar_[idxL] -= coeff * flux.energyFlux;
                    }

                    if (k < mesh.nz()) {
                        double coeff = dt * area / mesh.cellVolume(i, j, k);
                        rhoStar_[idxR]  += coeff * flux.massFlux;
                        rhoUStar_[idxR] += coeff * flux.momentumFlux[0];
                        rhoVStar_[idxR] += coeff * flux.momentumFlux[1];
                        rhoWStar_[idxR] += coeff * flux.momentumFlux[2];
                        rhoEStar_[idxR] += coeff * flux.energyFlux;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step 2: Pressure advection  p^a = p^n - Δt (u^n · ∇p^n)
// ---------------------------------------------------------------------------

void SemiImplicitSolver::advectPressure(const RectilinearMesh& mesh, double dt) {
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                double p = mesh.pres[idx];
                double advection = 0.0;

                // X-direction (upwind)
                {
                    double u = mesh.velU[idx];
                    double pxm = mesh.pres[mesh.index(i - 1, j, k)];
                    double pxp = mesh.pres[mesh.index(i + 1, j, k)];
                    if (u > 0) {
                        advection += u * (p - pxm) / mesh.dx(i);
                    } else {
                        advection += u * (pxp - p) / mesh.dx(i);
                    }
                }

                // Y-direction
                if (mesh.dim() >= 2) {
                    double v = mesh.velV[idx];
                    double pym = mesh.pres[mesh.index(i, j - 1, k)];
                    double pyp = mesh.pres[mesh.index(i, j + 1, k)];
                    if (v > 0) {
                        advection += v * (p - pym) / mesh.dy(j);
                    } else {
                        advection += v * (pyp - p) / mesh.dy(j);
                    }
                }

                // Z-direction
                if (mesh.dim() >= 3) {
                    double w = mesh.velW[idx];
                    double pzm = mesh.pres[mesh.index(i, j, k - 1)];
                    double pzp = mesh.pres[mesh.index(i, j, k + 1)];
                    if (w > 0) {
                        advection += w * (p - pzm) / mesh.dz(k);
                    } else {
                        advection += w * (pzp - p) / mesh.dz(k);
                    }
                }

                pAdvected_[idx] = p - dt * advection;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step 3: IGR solve for entropic pressure Σ
// ---------------------------------------------------------------------------

void SemiImplicitSolver::solveIGR(RectilinearMesh& mesh) {
    if (!igrSolver_) return;

    computeVelocityGradients(mesh);

    int nNeighbors = 2 * mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                double dxi = mesh.dx(i);

                double alpha = igrSolver_->computeAlpha(dxi);
                double rhs = igrSolver_->computeIGRRhs(gradU_[idx], alpha);

                double rhoCell = mesh.rho[idx];

                // Gather neighbor Σ/ρ
                std::array<double, 6> neighborSigmaRho = {0, 0, 0, 0, 0, 0};

                // X neighbors (always active)
                std::size_t xm = mesh.index(i - 1, j, k);
                std::size_t xp = mesh.index(i + 1, j, k);
                neighborSigmaRho[0] = mesh.sigma[xm] / std::max(mesh.rho[xm], 1e-14);
                neighborSigmaRho[1] = mesh.sigma[xp] / std::max(mesh.rho[xp], 1e-14);

                if (mesh.dim() >= 2) {
                    std::size_t ym = mesh.index(i, j - 1, k);
                    std::size_t yp = mesh.index(i, j + 1, k);
                    neighborSigmaRho[2] = mesh.sigma[ym] / std::max(mesh.rho[ym], 1e-14);
                    neighborSigmaRho[3] = mesh.sigma[yp] / std::max(mesh.rho[yp], 1e-14);
                }

                if (mesh.dim() >= 3) {
                    std::size_t zm = mesh.index(i, j, k - 1);
                    std::size_t zp = mesh.index(i, j, k + 1);
                    neighborSigmaRho[4] = mesh.sigma[zm] / std::max(mesh.rho[zm], 1e-14);
                    neighborSigmaRho[5] = mesh.sigma[zp] / std::max(mesh.rho[zp], 1e-14);
                }

                mesh.sigma[idx] = igrSolver_->solveEntropicPressure(
                    rhs, rhoCell, alpha, dxi,
                    mesh.sigma[idx],     // warm start
                    neighborSigmaRho,
                    nNeighbors);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step 4: Pressure solve
// ---------------------------------------------------------------------------

void SemiImplicitSolver::solvePressure(RectilinearMesh& mesh, double dt) {
    // Compute ρc² at each cell
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                PrimitiveState W = gatherPrimitive(mesh, idx);
                double c = eos_->soundSpeed(W);
                rhoc2_[idx] = mesh.rho[idx] * c * c;
            }
        }
    }

    // Compute divergence of predicted velocity u*
    computeDivergence(mesh, divUstar_);

    // Build RHS: p^a - ρc²Δt ∇·u*
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                pressureRhs_[idx] = pAdvected_[idx] - rhoc2_[idx] * dt * divUstar_[idx];
            }
        }
    }

    // Initialize pressure with current values
    for (int k = 0; k < mesh.nz(); ++k)
        for (int j = 0; j < mesh.ny(); ++j)
            for (int i = 0; i < mesh.nx(); ++i)
                pressure_[mesh.index(i, j, k)] = mesh.pres[mesh.index(i, j, k)];

    // Solve
    lastPressureIters_ = pressureSolver_->solve(
        mesh, rhoc2_, pressureRhs_, pressure_,
        dt, params_.pressureTol, params_.maxPressureIters);

    // Ensure ghost cells of pressure are filled for the correction step
    mesh.fillScalarGhosts(pressure_);

    // Also fill sigma ghost cells (may have been updated by IGR)
    mesh.fillScalarGhosts(mesh.sigma);

    // Store new pressure in mesh
    for (int k = 0; k < mesh.nz(); ++k)
        for (int j = 0; j < mesh.ny(); ++j)
            for (int i = 0; i < mesh.nx(); ++i)
                mesh.pres[mesh.index(i, j, k)] = pressure_[mesh.index(i, j, k)];
}

// ---------------------------------------------------------------------------
// Step 5: Correction
// ---------------------------------------------------------------------------

void SemiImplicitSolver::correctionStep(RectilinearMesh& mesh, double dt) {
    // Correct momentum: (ρu)^{n+1} = (ρu)* - Δt ∇(p + Σ)
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                // X-direction pressure gradient
                {
                    std::size_t xm = mesh.index(i - 1, j, k);
                    std::size_t xp = mesh.index(i + 1, j, k);
                    double pTotL = 0.5 * ((pressure_[xm] + mesh.sigma[xm]) +
                                           (pressure_[idx] + mesh.sigma[idx]));
                    double pTotR = 0.5 * ((pressure_[idx] + mesh.sigma[idx]) +
                                           (pressure_[xp] + mesh.sigma[xp]));
                    mesh.rhoU[idx] = rhoUStar_[idx] - dt * (pTotR - pTotL) / mesh.dx(i);
                }

                // Y-direction
                if (mesh.dim() >= 2) {
                    std::size_t ym = mesh.index(i, j - 1, k);
                    std::size_t yp = mesh.index(i, j + 1, k);
                    double pTotL = 0.5 * ((pressure_[ym] + mesh.sigma[ym]) +
                                           (pressure_[idx] + mesh.sigma[idx]));
                    double pTotR = 0.5 * ((pressure_[idx] + mesh.sigma[idx]) +
                                           (pressure_[yp] + mesh.sigma[yp]));
                    mesh.rhoV[idx] = rhoVStar_[idx] - dt * (pTotR - pTotL) / mesh.dy(j);
                }

                // Z-direction
                if (mesh.dim() >= 3) {
                    std::size_t zm = mesh.index(i, j, k - 1);
                    std::size_t zp = mesh.index(i, j, k + 1);
                    double pTotL = 0.5 * ((pressure_[zm] + mesh.sigma[zm]) +
                                           (pressure_[idx] + mesh.sigma[idx]));
                    double pTotR = 0.5 * ((pressure_[idx] + mesh.sigma[idx]) +
                                           (pressure_[zp] + mesh.sigma[zp]));
                    mesh.rhoW[idx] = rhoWStar_[idx] - dt * (pTotR - pTotL) / mesh.dz(k);
                }
            }
        }
    }

    // Convert to primitives to get u^{n+1}, then apply BCs
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                ConservativeState U;
                U.rho = mesh.rho[idx];
                U.rhoU = {mesh.rhoU[idx], mesh.rhoV[idx], mesh.rhoW[idx]};
                U.rhoE = mesh.rhoE[idx];  // still star energy
                PrimitiveState W = eos_->toPrimitive(U);
                mesh.velU[idx] = W.u[0];
                mesh.velV[idx] = W.u[1];
                mesh.velW[idx] = W.u[2];
            }
        }
    }
    mesh.applyBoundaryConditions();

    // Energy correction: E^{n+1} = E* - Δt ∇·((p+Σ) u^{n+1})
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                double energyCorr = 0.0;

                // X-direction: d((p+Σ)*u)/dx
                {
                    std::size_t xm = mesh.index(i - 1, j, k);
                    std::size_t xp = mesh.index(i + 1, j, k);
                    double pTotFaceL = 0.5 * ((pressure_[xm] + mesh.sigma[xm]) +
                                               (pressure_[idx] + mesh.sigma[idx]));
                    double pTotFaceR = 0.5 * ((pressure_[idx] + mesh.sigma[idx]) +
                                               (pressure_[xp] + mesh.sigma[xp]));
                    double uFaceL = 0.5 * (mesh.velU[xm] + mesh.velU[idx]);
                    double uFaceR = 0.5 * (mesh.velU[idx] + mesh.velU[xp]);
                    energyCorr += (pTotFaceR * uFaceR - pTotFaceL * uFaceL) / mesh.dx(i);
                }

                if (mesh.dim() >= 2) {
                    std::size_t ym = mesh.index(i, j - 1, k);
                    std::size_t yp = mesh.index(i, j + 1, k);
                    double pTotFaceL = 0.5 * ((pressure_[ym] + mesh.sigma[ym]) +
                                               (pressure_[idx] + mesh.sigma[idx]));
                    double pTotFaceR = 0.5 * ((pressure_[idx] + mesh.sigma[idx]) +
                                               (pressure_[yp] + mesh.sigma[yp]));
                    double vFaceL = 0.5 * (mesh.velV[ym] + mesh.velV[idx]);
                    double vFaceR = 0.5 * (mesh.velV[idx] + mesh.velV[yp]);
                    energyCorr += (pTotFaceR * vFaceR - pTotFaceL * vFaceL) / mesh.dy(j);
                }

                if (mesh.dim() >= 3) {
                    std::size_t zm = mesh.index(i, j, k - 1);
                    std::size_t zp = mesh.index(i, j, k + 1);
                    double pTotFaceL = 0.5 * ((pressure_[zm] + mesh.sigma[zm]) +
                                               (pressure_[idx] + mesh.sigma[idx]));
                    double pTotFaceR = 0.5 * ((pressure_[idx] + mesh.sigma[idx]) +
                                               (pressure_[zp] + mesh.sigma[zp]));
                    double wFaceL = 0.5 * (mesh.velW[zm] + mesh.velW[idx]);
                    double wFaceR = 0.5 * (mesh.velW[idx] + mesh.velW[zp]);
                    energyCorr += (pTotFaceR * wFaceR - pTotFaceL * wFaceL) / mesh.dz(k);
                }

                mesh.rhoE[idx] = rhoEStar_[idx] - dt * energyCorr;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Divergence of velocity
// ---------------------------------------------------------------------------

void SemiImplicitSolver::computeDivergence(
    const RectilinearMesh& mesh, std::vector<double>& divU)
{
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                double div = 0.0;

                // X: (u_{i+1/2} - u_{i-1/2}) / dx
                {
                    std::size_t xm = mesh.index(i - 1, j, k);
                    std::size_t xp = mesh.index(i + 1, j, k);
                    double uFaceL = 0.5 * (mesh.velU[xm] + mesh.velU[idx]);
                    double uFaceR = 0.5 * (mesh.velU[idx] + mesh.velU[xp]);
                    div += (uFaceR - uFaceL) / mesh.dx(i);
                }

                if (mesh.dim() >= 2) {
                    std::size_t ym = mesh.index(i, j - 1, k);
                    std::size_t yp = mesh.index(i, j + 1, k);
                    double vFaceL = 0.5 * (mesh.velV[ym] + mesh.velV[idx]);
                    double vFaceR = 0.5 * (mesh.velV[idx] + mesh.velV[yp]);
                    div += (vFaceR - vFaceL) / mesh.dy(j);
                }

                if (mesh.dim() >= 3) {
                    std::size_t zm = mesh.index(i, j, k - 1);
                    std::size_t zp = mesh.index(i, j, k + 1);
                    double wFaceL = 0.5 * (mesh.velW[zm] + mesh.velW[idx]);
                    double wFaceR = 0.5 * (mesh.velW[idx] + mesh.velW[zp]);
                    div += (wFaceR - wFaceL) / mesh.dz(k);
                }

                divU[idx] = div;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Velocity gradients for IGR
// ---------------------------------------------------------------------------

void SemiImplicitSolver::computeVelocityGradients(const RectilinearMesh& mesh) {
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                // X neighbors
                std::size_t xm = mesh.index(i - 1, j, k);
                std::size_t xp = mesh.index(i + 1, j, k);
                std::array<double, 3> u_xm = {mesh.velU[xm], mesh.velV[xm], mesh.velW[xm]};
                std::array<double, 3> u_xp = {mesh.velU[xp], mesh.velV[xp], mesh.velW[xp]};

                // Y neighbors
                std::array<double, 3> u_ym, u_yp;
                double dyj;
                if (mesh.dim() >= 2) {
                    std::size_t ym = mesh.index(i, j - 1, k);
                    std::size_t yp = mesh.index(i, j + 1, k);
                    u_ym = {mesh.velU[ym], mesh.velV[ym], mesh.velW[ym]};
                    u_yp = {mesh.velU[yp], mesh.velV[yp], mesh.velW[yp]};
                    dyj = mesh.dy(j);
                } else {
                    // Inactive dimension: zero gradient
                    u_ym = u_yp = {mesh.velU[idx], mesh.velV[idx], mesh.velW[idx]};
                    dyj = 1.0;
                }

                // Z neighbors
                std::array<double, 3> u_zm, u_zp;
                double dzk;
                if (mesh.dim() >= 3) {
                    std::size_t zm = mesh.index(i, j, k - 1);
                    std::size_t zp = mesh.index(i, j, k + 1);
                    u_zm = {mesh.velU[zm], mesh.velV[zm], mesh.velW[zm]};
                    u_zp = {mesh.velU[zp], mesh.velV[zp], mesh.velW[zp]};
                    dzk = mesh.dz(k);
                } else {
                    u_zm = u_zp = {mesh.velU[idx], mesh.velV[idx], mesh.velW[idx]};
                    dzk = 1.0;
                }

                gradU_[idx] = IGRSolver::computeVelocityGradient(
                    u_xm, u_xp, u_ym, u_yp, u_zm, u_zp,
                    mesh.dx(i), dyj, dzk);
            }
        }
    }
}

} // namespace SemiImplicitFV

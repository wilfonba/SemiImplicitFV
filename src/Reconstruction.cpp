#include "Reconstruction.hpp"
#include <iostream>
#include <cassert>

namespace SemiImplicitFV {

namespace {
    constexpr double WENO_EPS = 1.0e-6;

    using ReconFn = double(*)(const double*);

    inline void reconstructScalar(
        const double* field,
        const std::size_t* cells,
        int stencilSize,
        ReconFn leftFn,
        ReconFn rightFn,
        double& outLeft,
        double& outRight)
    {
        double vL[5], vR[5];
        for (int s = 0; s < stencilSize; ++s) {
            vL[s] = field[cells[s]];
            vR[s] = field[cells[s + 1]];
        }
        outLeft  = leftFn(vL);
        outRight = rightFn(vR);
    }
} // anonymous namespace

Reconstructor::Reconstructor(ReconstructionOrder order)
    : order_(order)
{}

int Reconstructor::requiredGhostCells() const {
    switch (order_) {
        case ReconstructionOrder::WENO1:     return 1;
        case ReconstructionOrder::WENO3:     return 2;
        case ReconstructionOrder::WENO5:     return 3;
    }
    return 1;
}

std::size_t Reconstructor::xFaceIndex(int i, int j, int k) const {
    return static_cast<std::size_t>(i + (nx_ + 1) * (j + ny_ * k));
}

std::size_t Reconstructor::yFaceIndex(int i, int j, int k) const {
    return static_cast<std::size_t>(i + nx_ * (j + (ny_ + 1) * k));
}

std::size_t Reconstructor::zFaceIndex(int i, int j, int k) const {
    return static_cast<std::size_t>(i + nx_ * (j + ny_ * k));
}

void Reconstructor::ensureStorage(const RectilinearMesh& mesh) {
    int newDim = mesh.dim();
    int newNx = mesh.nx(), newNy = mesh.ny(), newNz = mesh.nz();

    if (dim_ == newDim && nx_ == newNx && ny_ == newNy && nz_ == newNz)
        return;

    dim_ = newDim;
    nx_ = newNx;
    ny_ = newNy;
    nz_ = newNz;

    numXFaces_ = static_cast<std::size_t>(nx_ + 1) * ny_ * nz_;
    xLeft_.resize(numXFaces_);
    xRight_.resize(numXFaces_);

    if (dim_ >= 2) {
        numYFaces_ = static_cast<std::size_t>(nx_) * (ny_ + 1) * nz_;
        yLeft_.resize(numYFaces_);
        yRight_.resize(numYFaces_);
    } else {
        numYFaces_ = 0;
        yLeft_.clear();
        yRight_.clear();
    }

    if (dim_ >= 3) {
        numZFaces_ = static_cast<std::size_t>(nx_) * ny_ * (nz_ + 1);
        zLeft_.resize(numZFaces_);
        zRight_.resize(numZFaces_);
    } else {
        numZFaces_ = 0;
        zLeft_.clear();
        zRight_.clear();
    }
}

// ---- WENO3 kernels ----

double Reconstructor::weno3Left(const double* v) {
    // v[0] = v_{i-1}, v[1] = v_i, v[2] = v_{i+1}
    double p0 = -0.5 * v[0] + 1.5 * v[1];
    double p1 =  0.5 * v[1] + 0.5 * v[2];

    double b0 = (v[1] - v[0]) * (v[1] - v[0]);
    double b1 = (v[2] - v[1]) * (v[2] - v[1]);

    constexpr double d0 = 2.0 / 3.0;
    constexpr double d1 = 1.0 / 3.0;

    double a0 = d0 / ((WENO_EPS + b0) * (WENO_EPS + b0));
    double a1 = d1 / ((WENO_EPS + b1) * (WENO_EPS + b1));
    double aSum = a0 + a1;

    return (a0 * p0 + a1 * p1) / aSum;
}

double Reconstructor::weno3Right(const double* v) {
    // v[0] = v_i, v[1] = v_{i+1}, v[2] = v_{i+2}
    double p0 =  0.5 * v[0] + 0.5 * v[1];
    double p1 =  1.5 * v[1] - 0.5 * v[2];

    double b0 = (v[1] - v[0]) * (v[1] - v[0]);
    double b1 = (v[2] - v[1]) * (v[2] - v[1]);

    constexpr double d0 = 2.0 / 3.0;
    constexpr double d1 = 1.0 / 3.0;

    double a0 = d0 / ((WENO_EPS + b0) * (WENO_EPS + b0));
    double a1 = d1 / ((WENO_EPS + b1) * (WENO_EPS + b1));
    double aSum = a0 + a1;

    return (a0 * p0 + a1 * p1) / aSum;
}

// ---- WENO5 kernels (Jiang-Shu) ----

double Reconstructor::weno5Left(const double* v) {
    // v[0]=v_{i-2}, v[1]=v_{i-1}, v[2]=v_i, v[3]=v_{i+1}, v[4]=v_{i+2}
    double p0 = (1.0/3.0)*v[0] - (7.0/6.0)*v[1] + (11.0/6.0)*v[2];
    double p1 = -(1.0/6.0)*v[1] + (5.0/6.0)*v[2] + (1.0/3.0)*v[3];
    double p2 = (1.0/3.0)*v[2] + (5.0/6.0)*v[3] - (1.0/6.0)*v[4];

    double b0 = (13.0/12.0)*(v[0] - 2.0*v[1] + v[2])*(v[0] - 2.0*v[1] + v[2])
              + (1.0/4.0)*(v[0] - 4.0*v[1] + 3.0*v[2])*(v[0] - 4.0*v[1] + 3.0*v[2]);
    double b1 = (13.0/12.0)*(v[1] - 2.0*v[2] + v[3])*(v[1] - 2.0*v[2] + v[3])
              + (1.0/4.0)*(v[1] - v[3])*(v[1] - v[3]);
    double b2 = (13.0/12.0)*(v[2] - 2.0*v[3] + v[4])*(v[2] - 2.0*v[3] + v[4])
              + (1.0/4.0)*(3.0*v[2] - 4.0*v[3] + v[4])*(3.0*v[2] - 4.0*v[3] + v[4]);

    constexpr double d0 = 1.0 / 10.0;
    constexpr double d1 = 6.0 / 10.0;
    constexpr double d2 = 3.0 / 10.0;

    double a0 = d0 / ((WENO_EPS + b0) * (WENO_EPS + b0));
    double a1 = d1 / ((WENO_EPS + b1) * (WENO_EPS + b1));
    double a2 = d2 / ((WENO_EPS + b2) * (WENO_EPS + b2));
    double aSum = a0 + a1 + a2;

    return (a0 * p0 + a1 * p1 + a2 * p2) / aSum;
}

double Reconstructor::weno5Right(const double* v) {
    // v[0]=v_{i-2}, v[1]=v_{i-1}, v[2]=v_i, v[3]=v_{i+1}, v[4]=v_{i+2}
    // Mirror: reverse the stencil for right-biased reconstruction
    double p0 = (1.0/3.0)*v[4] - (7.0/6.0)*v[3] + (11.0/6.0)*v[2];
    double p1 = -(1.0/6.0)*v[3] + (5.0/6.0)*v[2] + (1.0/3.0)*v[1];
    double p2 = (1.0/3.0)*v[2] + (5.0/6.0)*v[1] - (1.0/6.0)*v[0];

    double b0 = (13.0/12.0)*(v[4] - 2.0*v[3] + v[2])*(v[4] - 2.0*v[3] + v[2])
              + (1.0/4.0)*(v[4] - 4.0*v[3] + 3.0*v[2])*(v[4] - 4.0*v[3] + 3.0*v[2]);
    double b1 = (13.0/12.0)*(v[3] - 2.0*v[2] + v[1])*(v[3] - 2.0*v[2] + v[1])
              + (1.0/4.0)*(v[3] - v[1])*(v[3] - v[1]);
    double b2 = (13.0/12.0)*(v[2] - 2.0*v[1] + v[0])*(v[2] - 2.0*v[1] + v[0])
              + (1.0/4.0)*(3.0*v[2] - 4.0*v[1] + v[0])*(3.0*v[2] - 4.0*v[1] + v[0]);

    constexpr double d0 = 1.0 / 10.0;
    constexpr double d1 = 6.0 / 10.0;
    constexpr double d2 = 3.0 / 10.0;

    double a0 = d0 / ((WENO_EPS + b0) * (WENO_EPS + b0));
    double a1 = d1 / ((WENO_EPS + b1) * (WENO_EPS + b1));
    double a2 = d2 / ((WENO_EPS + b2) * (WENO_EPS + b2));
    double aSum = a0 + a1 + a2;

    return (a0 * p0 + a1 * p1 + a2 * p2) / aSum;
}

// ---- Direction-specific reconstruction sweeps ----

void Reconstructor::reconstructX(const RectilinearMesh& mesh, const SolutionState& state) {
    const int nx = mesh.nx();
    const int ny = mesh.ny();
    const int nz = mesh.nz();

    const double* rho  = state.rho.data();
    const double* velU = state.velU.data();
    const double* pres = state.pres.data();
    const double* sig  = state.sigma.data();
    const double* velV = (dim_ >= 2) ? state.velV.data() : nullptr;
    const double* velW = (dim_ >= 3) ? state.velW.data() : nullptr;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                std::size_t fIdx = xFaceIndex(i, j, k);
                PrimitiveState& left  = xLeft_[fIdx];
                PrimitiveState& right = xRight_[fIdx];
                left = PrimitiveState{};
                right = PrimitiveState{};

                if (order_ == ReconstructionOrder::WENO1) {
                    std::size_t idxL = mesh.index(i - 1, j, k);
                    std::size_t idxR = mesh.index(i, j, k);
                    left.rho   = rho[idxL];
                    left.u[0]  = velU[idxL];
                    left.p     = pres[idxL];
                    left.sigma = sig[idxL];
                    right.rho   = rho[idxR];
                    right.u[0]  = velU[idxR];
                    right.p     = pres[idxR];
                    right.sigma = sig[idxR];
                    if (dim_ >= 2) {
                        left.u[1]  = velV[idxL];
                        right.u[1] = velV[idxR];
                    }
                    if (dim_ >= 3) {
                        left.u[2]  = velW[idxL];
                        right.u[2] = velW[idxR];
                    }
                }
                else if (order_ == ReconstructionOrder::WENO3) {
                    // 4 cells: i-2, i-1, i, i+1
                    std::size_t c[4];
                    c[0] = mesh.index(i - 2, j, k);
                    c[1] = mesh.index(i - 1, j, k);
                    c[2] = mesh.index(i,     j, k);
                    c[3] = mesh.index(i + 1, j, k);

                    reconstructScalar(rho,  c, 3, weno3Left, weno3Right, left.rho,  right.rho);
                    reconstructScalar(velU, c, 3, weno3Left, weno3Right, left.u[0], right.u[0]);
                    reconstructScalar(pres, c, 3, weno3Left, weno3Right, left.p,    right.p);
                    reconstructScalar(sig,  c, 3, weno3Left, weno3Right, left.sigma, right.sigma);
                    if (dim_ >= 2)
                        reconstructScalar(velV, c, 3, weno3Left, weno3Right, left.u[1], right.u[1]);
                    if (dim_ >= 3)
                        reconstructScalar(velW, c, 3, weno3Left, weno3Right, left.u[2], right.u[2]);
                }
                else { // WENO5
                    // 6 cells: i-3, i-2, i-1, i, i+1, i+2
                    std::size_t c[6];
                    c[0] = mesh.index(i - 3, j, k);
                    c[1] = mesh.index(i - 2, j, k);
                    c[2] = mesh.index(i - 1, j, k);
                    c[3] = mesh.index(i,     j, k);
                    c[4] = mesh.index(i + 1, j, k);
                    c[5] = mesh.index(i + 2, j, k);

                    reconstructScalar(rho,  c, 5, weno5Left, weno5Right, left.rho,  right.rho);
                    reconstructScalar(velU, c, 5, weno5Left, weno5Right, left.u[0], right.u[0]);
                    reconstructScalar(pres, c, 5, weno5Left, weno5Right, left.p,    right.p);
                    reconstructScalar(sig,  c, 5, weno5Left, weno5Right, left.sigma, right.sigma);
                    if (dim_ >= 2)
                        reconstructScalar(velV, c, 5, weno5Left, weno5Right, left.u[1], right.u[1]);
                    if (dim_ >= 3)
                        reconstructScalar(velW, c, 5, weno5Left, weno5Right, left.u[2], right.u[2]);
                }
            }
        }
    }
}

void Reconstructor::reconstructY(const RectilinearMesh& mesh, const SolutionState& state) {
    const int nx = mesh.nx();
    const int ny = mesh.ny();
    const int nz = mesh.nz();

    const double* rho  = state.rho.data();
    const double* velU = state.velU.data();
    const double* velV = state.velV.data();
    const double* pres = state.pres.data();
    const double* sig  = state.sigma.data();
    const double* velW = (dim_ >= 3) ? state.velW.data() : nullptr;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                std::size_t fIdx = yFaceIndex(i, j, k);
                PrimitiveState& left  = yLeft_[fIdx];
                PrimitiveState& right = yRight_[fIdx];
                left = PrimitiveState{};
                right = PrimitiveState{};

                if (order_ == ReconstructionOrder::WENO1) {
                    std::size_t idxL = mesh.index(i, j - 1, k);
                    std::size_t idxR = mesh.index(i, j, k);
                    left.rho   = rho[idxL];
                    left.u[0]  = velU[idxL];
                    left.u[1]  = velV[idxL];
                    left.p     = pres[idxL];
                    left.sigma = sig[idxL];
                    right.rho   = rho[idxR];
                    right.u[0]  = velU[idxR];
                    right.u[1]  = velV[idxR];
                    right.p     = pres[idxR];
                    right.sigma = sig[idxR];
                    if (dim_ >= 3) {
                        left.u[2]  = velW[idxL];
                        right.u[2] = velW[idxR];
                    }
                }
                else if (order_ == ReconstructionOrder::WENO3) {
                    std::size_t c[4];
                    c[0] = mesh.index(i, j - 2, k);
                    c[1] = mesh.index(i, j - 1, k);
                    c[2] = mesh.index(i, j,     k);
                    c[3] = mesh.index(i, j + 1, k);

                    reconstructScalar(rho,  c, 3, weno3Left, weno3Right, left.rho,  right.rho);
                    reconstructScalar(velU, c, 3, weno3Left, weno3Right, left.u[0], right.u[0]);
                    reconstructScalar(velV, c, 3, weno3Left, weno3Right, left.u[1], right.u[1]);
                    reconstructScalar(pres, c, 3, weno3Left, weno3Right, left.p,    right.p);
                    reconstructScalar(sig,  c, 3, weno3Left, weno3Right, left.sigma, right.sigma);
                    if (dim_ >= 3)
                        reconstructScalar(velW, c, 3, weno3Left, weno3Right, left.u[2], right.u[2]);
                }
                else { // WENO5
                    std::size_t c[6];
                    c[0] = mesh.index(i, j - 3, k);
                    c[1] = mesh.index(i, j - 2, k);
                    c[2] = mesh.index(i, j - 1, k);
                    c[3] = mesh.index(i, j,     k);
                    c[4] = mesh.index(i, j + 1, k);
                    c[5] = mesh.index(i, j + 2, k);

                    reconstructScalar(rho,  c, 5, weno5Left, weno5Right, left.rho,  right.rho);
                    reconstructScalar(velU, c, 5, weno5Left, weno5Right, left.u[0], right.u[0]);
                    reconstructScalar(velV, c, 5, weno5Left, weno5Right, left.u[1], right.u[1]);
                    reconstructScalar(pres, c, 5, weno5Left, weno5Right, left.p,    right.p);
                    reconstructScalar(sig,  c, 5, weno5Left, weno5Right, left.sigma, right.sigma);
                    if (dim_ >= 3)
                        reconstructScalar(velW, c, 5, weno5Left, weno5Right, left.u[2], right.u[2]);
                }
            }
        }
    }
}

void Reconstructor::reconstructZ(const RectilinearMesh& mesh, const SolutionState& state) {
    const int nx = mesh.nx();
    const int ny = mesh.ny();
    const int nz = mesh.nz();

    const double* rho  = state.rho.data();
    const double* velU = state.velU.data();
    const double* velV = state.velV.data();
    const double* velW = state.velW.data();
    const double* pres = state.pres.data();
    const double* sig  = state.sigma.data();

    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                std::size_t fIdx = zFaceIndex(i, j, k);
                PrimitiveState& left  = zLeft_[fIdx];
                PrimitiveState& right = zRight_[fIdx];
                left = PrimitiveState{};
                right = PrimitiveState{};

                if (order_ == ReconstructionOrder::WENO1) {
                    std::size_t idxL = mesh.index(i, j, k - 1);
                    std::size_t idxR = mesh.index(i, j, k);
                    left.rho   = rho[idxL];
                    left.u[0]  = velU[idxL];
                    left.u[1]  = velV[idxL];
                    left.u[2]  = velW[idxL];
                    left.p     = pres[idxL];
                    left.sigma = sig[idxL];
                    right.rho   = rho[idxR];
                    right.u[0]  = velU[idxR];
                    right.u[1]  = velV[idxR];
                    right.u[2]  = velW[idxR];
                    right.p     = pres[idxR];
                    right.sigma = sig[idxR];
                }
                else if (order_ == ReconstructionOrder::WENO3) {
                    std::size_t c[4];
                    c[0] = mesh.index(i, j, k - 2);
                    c[1] = mesh.index(i, j, k - 1);
                    c[2] = mesh.index(i, j, k);
                    c[3] = mesh.index(i, j, k + 1);

                    reconstructScalar(rho,  c, 3, weno3Left, weno3Right, left.rho,  right.rho);
                    reconstructScalar(velU, c, 3, weno3Left, weno3Right, left.u[0], right.u[0]);
                    reconstructScalar(velV, c, 3, weno3Left, weno3Right, left.u[1], right.u[1]);
                    reconstructScalar(velW, c, 3, weno3Left, weno3Right, left.u[2], right.u[2]);
                    reconstructScalar(pres, c, 3, weno3Left, weno3Right, left.p,    right.p);
                    reconstructScalar(sig,  c, 3, weno3Left, weno3Right, left.sigma, right.sigma);
                }
                else { // WENO5
                    std::size_t c[6];
                    c[0] = mesh.index(i, j, k - 3);
                    c[1] = mesh.index(i, j, k - 2);
                    c[2] = mesh.index(i, j, k - 1);
                    c[3] = mesh.index(i, j, k);
                    c[4] = mesh.index(i, j, k + 1);
                    c[5] = mesh.index(i, j, k + 2);

                    reconstructScalar(rho,  c, 5, weno5Left, weno5Right, left.rho,  right.rho);
                    reconstructScalar(velU, c, 5, weno5Left, weno5Right, left.u[0], right.u[0]);
                    reconstructScalar(velV, c, 5, weno5Left, weno5Right, left.u[1], right.u[1]);
                    reconstructScalar(velW, c, 5, weno5Left, weno5Right, left.u[2], right.u[2]);
                    reconstructScalar(pres, c, 5, weno5Left, weno5Right, left.p,    right.p);
                    reconstructScalar(sig,  c, 5, weno5Left, weno5Right, left.sigma, right.sigma);
                }
            }
        }
    }
}

void Reconstructor::reconstruct(
        [[maybe_unused]] const SimulationConfig& config,
        const RectilinearMesh& mesh,
        const SolutionState& state)
{
    assert(config.nGhost >= requiredGhostCells());
    ensureStorage(mesh);
    reconstructX(mesh, state);
    if (dim_ >= 2) reconstructY(mesh, state);
    if (dim_ >= 3) reconstructZ(mesh, state);
}

} // namespace SemiImplicitFV

/**
 * 1D Subsonic Advection (Entropy Wave)
 *
 * A Gaussian density pulse advects at uniform subsonic velocity through
 * a periodic domain.  This is a pure entropy wave: pressure and velocity
 * are uniform, so the exact solution is a rigid translation of the
 * initial density profile at the flow velocity u0.
 *
 * After one full domain traversal the pulse should return to its
 * starting position, making error measurement straightforward.
 *
 * Useful for verifying:
 *   - Advective flux accuracy and numerical dissipation
 *   - Periodic boundary condition implementation
 *   - Convergence rates (vary numCells)
 */

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "RusanovSolver.hpp"
#include "SemiImplicitSolver.hpp"
#include "GaussSeidelPressureSolver.hpp"
#include "IGR.hpp"
#include "IdealGasEOS.hpp"

#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>

using namespace SemiImplicitFV;

// ---- Problem parameters ----
static constexpr double rho0    = 1.225;      // Background density  [kg/m³]
static constexpr double p0      = 101325.0;   // Background pressure [Pa]
static constexpr double u0      = 50.0;       // Advection velocity  [m/s]  (Mach ≈ 0.15)
static constexpr double amp     = 0.01;       // Perturbation amplitude (1 %)
static constexpr double xCenter = 0.5;        // Initial pulse centre
static constexpr double sigma   = 0.05;       // Pulse width

// Gaussian density profile with periodic wrapping
double densityProfile(double x, double xc, double L) {
    double dx = x - xc;
    dx -= L * std::round(dx / L);   // shortest periodic distance
    return rho0 * (1.0 + amp * std::exp(-(dx * dx) / (sigma * sigma)));
}

void initializeProblem(const RectilinearMesh& mesh, SolutionState& state,
                       const IdealGasEOS& eos, double xc, double L) {
    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        double x = mesh.cellCentroidX(i);

        PrimitiveState W;
        W.rho   = densityProfile(x, xc, L);
        W.u     = {u0, 0.0, 0.0};
        W.p     = p0;
        W.sigma = 0.0;
        W.T     = eos.temperature(W);

        state.setPrimitiveState(idx, W);

        ConservativeState U = eos.toConservative(W);
        state.setConservativeState(idx, U);
    }
}

void writeSolution(const RectilinearMesh& mesh, const SolutionState& state,
                   const std::string& filename) {
    std::ofstream file(filename);
    file << "# x rho u p\n";
    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        file << mesh.cellCentroidX(i) << " "
             << state.rho[idx] << " "
             << state.velU[idx] << " "
             << state.pres[idx] << "\n";
    }
}

int main() {
    std::cout << "Semi-Implicit FV Solver - 1D Subsonic Advection\n";
    std::cout << "================================================\n\n";

    // ---- Setup ----
    const int    numCells = 200;
    const double length   = 1.0;
    const double endTime  = length / u0;   // one full domain traversal

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0);

    PrimitiveState ref;
    ref.rho = rho0;
    ref.u   = {u0, 0.0, 0.0};
    ref.p   = p0;
    double c0   = eos->soundSpeed(ref);
    double mach = u0 / c0;

    std::cout << "  Cells:    " << numCells << "\n";
    std::cout << "  u0:       " << u0    << " m/s\n";
    std::cout << "  c0:       " << c0    << " m/s\n";
    std::cout << "  Mach:     " << mach  << "\n";
    std::cout << "  End time: " << endTime << " s  (one domain traversal)\n\n";

    // ---- Mesh (periodic) ----
    RectilinearMesh mesh = RectilinearMesh::createUniform(1, numCells, 0.0, length);
    mesh.setBoundaryCondition(RectilinearMesh::XLow,  BoundaryCondition::Periodic);
    mesh.setBoundaryCondition(RectilinearMesh::XHigh, BoundaryCondition::Periodic);

    // Allocate solution state
    SolutionState state;
    state.allocate(mesh.totalCells(), mesh.dim());

    // ---- Initial condition ----
    initializeProblem(mesh, state, *eos, xCenter, length);
    writeSolution(mesh, state, "advection_t0.dat");

    // ---- Solver components ----
    auto riemann  = std::make_shared<RusanovSolver>(eos);
    auto pressure = std::make_shared<GaussSeidelPressureSolver>();

    IGRParams igrParams;
    igrParams.alphaCoeff    = 1.0;
    igrParams.maxIterations = 5;
    igrParams.tolerance     = 1e-10;
    auto igr = std::make_shared<IGRSolver>(igrParams);

    SemiImplicitParams params;
    params.cfl              = 0.95;
    params.maxDt            = 1e-3;
    params.maxPressureIters = 200;
    params.pressureTol      = 1e-8;
    params.useIGR           = true;

    SemiImplicitSolver solver(riemann, pressure, eos, igr, params);

    // ---- Time integration ----
    std::cout << "Running...\n\n";
    double time = 0.0;
    int    step = 0;

    while (time < endTime) {
        double dt = solver.step(mesh, state, endTime - time);
        time += dt;
        step++;

        if (step % 50 == 0 || step == 1) {
            std::cout << "  Step " << step
                      << ": t = " << time
                      << ", dt = " << dt
                      << ", pressure iters = " << solver.lastPressureIterations()
                      << "\n";
        }
    }

    std::cout << "\nDone after " << step << " steps.\n";

    // ---- Write results ----
    writeSolution(mesh, state, "advection_final.dat");

    // Exact solution: the pulse has translated by u0 * endTime = length,
    // so with periodic wrapping it should be back at xCenter.
    double exactCenter = xCenter + u0 * endTime;

    // ---- Error analysis ----
    double L1err = 0.0, Linf = 0.0;
    double dx = length / numCells;

    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        double x     = mesh.cellCentroidX(i);
        double exact  = densityProfile(x, exactCenter, length);
        double err    = std::abs(state.rho[idx] - exact);
        L1err += err * dx;
        Linf   = std::max(Linf, err);
    }

    std::cout << "\nError (density vs. exact advected profile):\n";
    std::cout << "  L1:   " << L1err << "\n";
    std::cout << "  Linf: " << Linf  << "\n";

    // Write exact solution for easy comparison / plotting
    {
        std::ofstream file("advection_exact.dat");
        file << "# x rho_exact\n";
        for (int i = 0; i < mesh.nx(); ++i) {
            double x = mesh.cellCentroidX(i);
            file << x << " " << densityProfile(x, exactCenter, length) << "\n";
        }
    }

    std::cout << "\nOutput files:\n";
    std::cout << "  advection_t0.dat     - initial condition\n";
    std::cout << "  advection_final.dat  - computed solution\n";
    std::cout << "  advection_exact.dat  - exact solution\n";

    return 0;
}

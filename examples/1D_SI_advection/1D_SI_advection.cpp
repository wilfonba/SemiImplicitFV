#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "RusanovSolver.hpp"
#include "LFSolver.hpp"
#include "SemiImplicitSolver.hpp"
#include "ExplicitSolver.hpp"
#include "VTKWriter.hpp"
#include "GaussSeidelPressureSolver.hpp"
#include "IGR.hpp"
#include "IdealGasEOS.hpp"
#include "SimulationConfig.hpp"

#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>

using namespace SemiImplicitFV;

// ---- Problem parameters ----
static constexpr double rho0    = 1.225;      // Background density  [kg/m³]
static constexpr double p0      = 101325.0;   // Background pressure [Pa]
static constexpr double u0      = 1.0;       // Advection velocity  [m/s]  (Mach ≈ 0.15)
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
    // ---- Setup ----
    const int    numCells = 1000;
    const double length   = 1.0;
    const double endTime  = length / u0;   // one full domain traversal

    // Simulation config
    SimulationConfig config;
    config.dim = 1;
    config.nGhost = 2;
    config.RKOrder = 1;
    config.useIGR = true;
    config.semiImplicit = true;

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);

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
    RectilinearMesh mesh = RectilinearMesh::createUniform(config, numCells, 0.0, length);
    mesh.setBoundaryCondition(RectilinearMesh::XLow,  BoundaryCondition::Periodic);
    mesh.setBoundaryCondition(RectilinearMesh::XHigh, BoundaryCondition::Periodic);

    // Allocate solution state
    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    IGRParams igrParams;
    igrParams.alphaCoeff = 10.0;
    igrParams.IGRIters = 5;
    auto igrSolver = std::make_shared<IGRSolver>(igrParams);
    auto riemannSolver  = std::make_shared<LFSolver>(eos, false, config);
    auto pressureSolver = std::make_shared<GaussSeidelPressureSolver>();

    SemiImplicitParams params;
    params.cfl              = 0.8;
    params.maxDt            = 1e-2;
    params.maxPressureIters = 200;
    SemiImplicitSolver solver(mesh, riemannSolver, pressureSolver, eos, igrSolver, params);

    //ExplicitParams params;
    //params.cfl = 0.5;
    //params.reconOrder = ReconstructionOrder::UPWIND3;
    //ExplicitSolver solver(mesh, riemannSolver, eos, igrSolver, params);

    initializeProblem(mesh, state, *eos, xCenter, length);

    // Initialize VTK time-series file
    VTKWriter::writePVD("VTK/1D_advection.pvd", "w");
    VTKWriter::writeVTR("VTK/1D_advection_0.vtr", mesh, state);
    VTKWriter::writePVD("VTK/1D_advection.pvd", "a", 0.0, "1D_advection_0.vtr");
    int fileNum = 1;
    const double outputInterval = endTime / 100.0;
    const int printInterval = 1;


    // ---- Time integration ----
    double time = 0.0;

    while (time < endTime) {
        double dt = solver.step(config, mesh, state, endTime - time);
        time += dt;
        config.step++;

        if (std::abs(time - fileNum * outputInterval) <= dt) {
            // Write VTK output
            std::string vtrFile = "1D_advection_" + std::to_string(fileNum) + ".vtr";
            VTKWriter::writeVTR("VTK/" + vtrFile, mesh, state);
            VTKWriter::writePVD("VTK/1D_advection.pvd", "a", time, vtrFile);
            fileNum++;
        }

        if (config.step % printInterval == 0 || config.step == 1) {
            double maxSigma = 0.0;
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, 0, 0);
                maxSigma = std::max(maxSigma, std::abs(state.sigma[idx]));
            }

            std::cout << "  Step " << config.step << ": t = " << time
                      << ", dt = " << dt
                      << ", max|sigma| = " << maxSigma << "\n";
        }

    }

    std::cout << "\nSimulation complete after " << config.step << " steps.\n";
    VTKWriter::writePVD("VTK/1D_advection.pvd", "close");

    return 0;
}

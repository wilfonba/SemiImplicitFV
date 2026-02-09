#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "RusanovSolver.hpp"
#include "LFSolver.hpp"
#include "HLLCSolver.hpp"
#include "SemiImplicitSolver.hpp"
#include "ExplicitSolver.hpp"
#include "VTKWriter.hpp"
#include "GaussSeidelPressureSolver.hpp"
#include "JacobiPressureSolver.hpp"
#include "IGR.hpp"
#include "IdealGasEOS.hpp"
#include "SimulationConfig.hpp"

#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>
#include <chrono>

using namespace SemiImplicitFV;

// ---- Problem parameters ----
static constexpr double rho0    = 1.225;      // Background density  [kg/m³]
static constexpr double p0      = 101325.0;   // Background pressure [Pa]
static constexpr double u0      = 10.0;       // Advection velocity  [m/s]  (Mach ≈ 0.15)
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

int main() {
    // ---- Setup ----
    const int    numCells = 1000;
    const double length   = 1.0;
    const double endTime  = length / u0;   // one full domain traversal

    // Simulation config
    SimulationConfig config;
    config.dim = 1;
    config.nGhost = 3;
    config.RKOrder = 1;
    config.useIGR = true;
    config.semiImplicit = true;
    config.reconOrder = ReconstructionOrder::UPWIND1;
    config.semiImplicitParams.cfl = 0.8;
    config.semiImplicitParams.maxDt = 1e-2;
    config.semiImplicitParams.pressureTol = 1e-9;
    config.semiImplicitParams.maxPressureIters = 200;
    config.igrParams.alphaCoeff = 10.0;
    config.igrParams.IGRIters = 5;
    config.validate();

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

    auto igrSolver = std::make_shared<IGRSolver>(config.igrParams);
    auto riemannSolver  = std::make_shared<LFSolver>(eos, config);
    auto pressureSolver = std::make_shared<GaussSeidelPressureSolver>();

    SemiImplicitSolver solver(mesh, riemannSolver, pressureSolver, eos, igrSolver, config);

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
    double stepTimeTotal = 0.0;

    while (time < endTime) {

        auto t0 = std::chrono::high_resolution_clock::now();
        double dt = solver.step(config, mesh, state, endTime - time);
        auto t1 = std::chrono::high_resolution_clock::now();
        stepTimeTotal += std::chrono::duration<double>(t1 - t0).count();

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
    std::cout << "Total time in solver.step(): " << stepTimeTotal << " s\n";
    VTKWriter::writePVD("VTK/1D_advection.pvd", "close");

    return 0;
}

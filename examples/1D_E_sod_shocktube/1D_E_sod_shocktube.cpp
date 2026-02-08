#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "HLLCSolver.hpp"
#include "LFSolver.hpp"
#include "ExplicitSolver.hpp"
#include "IGR.hpp"
#include "IdealGasEOS.hpp"
#include "SimulationConfig.hpp"
#include "VTKWriter.hpp"

#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>

using namespace SemiImplicitFV;

// Initialize Sod shock tube conditions (sharp IC smoothed with heat equation)
void initializeSodProblem(const RectilinearMesh& mesh, SolutionState& state,
                          const IdealGasEOS& eos, int smoothIters = 10) {
    double xMid = 0.5;

    // Left state: high pressure
    PrimitiveState left;
    left.rho = 1.0;
    left.u = {0.0, 0.0, 0.0};
    left.p = 1.0;
    left.sigma = 0.0;

    // Right state: low pressure
    PrimitiveState right;
    right.rho = 0.125;
    right.u = {0.0, 0.0, 0.0};
    right.p = 0.1;
    right.sigma = 0.0;

    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        const PrimitiveState& W = (mesh.cellCentroidX(i) < xMid) ? left : right;

        PrimitiveState Wt = W;
        Wt.T = eos.temperature(W);
        state.setPrimitiveState(idx, Wt);
        state.setConservativeState(idx, eos.toConservative(W));
    }

    // Smooth the sharp IC with heat equation iterations
    state.smoothFields(mesh, smoothIters);
}

int main() {
    const int numCells = 100;
    const double length = 1.0;
    [[maybe_unused]] const double constDt = 1e-4;
    const double endTime = 0.2;

    SimulationConfig config;
    config.dim = 1;
    config.nGhost = 4;
    config.RKOrder = 3;
    config.useIGR = true;

    RectilinearMesh mesh = RectilinearMesh::createUniform(
        config, numCells, 0.0, length);
    mesh.setBoundaryCondition(RectilinearMesh::XLow,  BoundaryCondition::Outflow);
    mesh.setBoundaryCondition(RectilinearMesh::XHigh, BoundaryCondition::Outflow);
    std::cout << "Created mesh with " << mesh.nx() << " cells.\n";

    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);
    auto riemannSolver = std::make_shared<LFSolver>(eos, true, config);

    IGRParams igrParams;
    igrParams.alphaCoeff = 10.0;       // α = αCoeff * Δx²
    igrParams.IGRIters = 5;
    auto igrSolver = std::make_shared<IGRSolver>(igrParams);
    std::cout << "Created IGR solver \n";

    ExplicitParams params;
    params.cfl = 0.1;
    //params.constDt = constDt;
    params.reconOrder = ReconstructionOrder::UPWIND3;

    ExplicitSolver solver(mesh, riemannSolver, eos, igrSolver, params);
    initializeSodProblem(mesh, state, *eos);

    // Initialize VTK time-series file
    VTKWriter::writePVD("VTK/1D_sod.pvd", "w");
    VTKWriter::writeVTR("VTK/1D_sod_0.vtr", mesh, state);
    VTKWriter::writePVD("VTK/1D_sod.pvd", "a", 0.0, "1D_sod_0.vtr");
    int fileNum = 1;
    const double outputInterval = 0.002;
    const int printInterval = 1;

    std::cout << "Running simulation to t = " << endTime << "...\n";

    double time = 0.0;

    while (time < endTime) {
        double dt = solver.step(config, mesh, state, endTime - time);
        time += dt;
        config.step++;

        if (std::abs(time - fileNum * outputInterval) <= dt) {
            // Write VTK output
            std::string vtrFile = "1D_sod_" + std::to_string(fileNum) + ".vtr";
            VTKWriter::writeVTR("VTK/" + vtrFile, mesh, state);
            VTKWriter::writePVD("VTK/1D_sod.pvd", "a", time, vtrFile);
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
    VTKWriter::writePVD("VTK/1D_sod.pvd", "close");

    return 0;
}

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "HLLCSolver.hpp"
#include "LFSolver.hpp"
#include "ExplicitSolver.hpp"
#include "IGR.hpp"
#include "IdealGasEOS.hpp"
#include "SimulationConfig.hpp"

#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>

using namespace SemiImplicitFV;

// Initialize Sod shock tube conditions
void initializeSodProblem(const RectilinearMesh& mesh, SolutionState& state, const IdealGasEOS& eos) {
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

        ConservativeState U = eos.toConservative(W);
        state.setConservativeState(idx, U);
    }
}

void writeSolution(const RectilinearMesh& mesh, const SolutionState& state, const std::string& filename) {
    std::ofstream file(filename);
    file << "# x rho u p sigma\n";

    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        file << mesh.cellCentroidX(i) << " "
             << state.rho[idx] << " "
             << state.velU[idx] << " "
             << state.pres[idx] << " "
             << state.sigma[idx] << "\n";
    }

    file.close();
}

int main() {
    const int numCells = 1000;
    const double length = 1.0;
    [[maybe_unused]] const double constDt = 1e-4;
    const double endTime = 0.2;

    SimulationConfig config;
    config.dim = 1;
    config.nGhost = 4;
    config.RKOrder = 3;

    RectilinearMesh mesh = RectilinearMesh::createUniform(
        config, numCells, 0.0, length);
    mesh.setBoundaryCondition(RectilinearMesh::XLow,  BoundaryCondition::Outflow);
    mesh.setBoundaryCondition(RectilinearMesh::XHigh, BoundaryCondition::Outflow);
    std::cout << "Created mesh with " << mesh.nx() << " cells.\n";

    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);
    auto riemannSolver = std::make_shared<HLLCSolver>(eos, true, config);

    IGRParams igrParams;
    igrParams.alphaCoeff = 1.0;       // α = αCoeff * Δx²
    igrParams.maxIterations = 5;
    igrParams.tolerance = 1e-10;
    auto igrSolver = std::make_shared<IGRSolver>(igrParams);
    std::cout << "Created IGR solver \n";

    ExplicitParams params;
    params.cfl = 0.1;
    //params.constDt = constDt;
    params.useIGR = false;
    params.reconOrder = ReconstructionOrder::WENO5;

    ExplicitSolver solver(riemannSolver, eos, igrSolver, params);
    initializeSodProblem(mesh, state, *eos);
    writeSolution(mesh, state, "sod_t0.dat");

    std::cout << "Running simulation to t = " << endTime << "...\n";
    std::cout << "Using IGR with alpha_coeff = " << igrParams.alphaCoeff << "\n\n";

    double time = 0.0;
    int step = 0;

    while (time < endTime) {
        double dt = solver.step(config, mesh, state, endTime - time);
        time += dt;
        step++;

        if (step % 20 == 0 || step == 1) {
            double maxSigma = 0.0;
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, 0, 0);
                maxSigma = std::max(maxSigma, std::abs(state.sigma[idx]));
            }

            std::cout << "  Step " << step << ": t = " << time
                      << ", dt = " << dt
                      << ", max|sigma| = " << maxSigma << "\n";
        }

    }

    std::cout << "\nSimulation complete after " << step << " steps.\n";

    // Write final solution
    writeSolution(mesh, state, "sod_final.dat");
    std::cout << "\nSolution written to sod_final.dat\n";

    return 0;
}

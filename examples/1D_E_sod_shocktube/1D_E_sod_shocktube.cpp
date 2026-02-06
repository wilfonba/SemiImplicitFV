#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "HLLCSolver.hpp"
#include "ExplicitSolver.hpp"
#include "IGR.hpp"
#include "IdealGasEOS.hpp"

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
    // Problem setup
    const int numCells = 1000;
    const double length = 1.0;
    const double endTime = 0.2;

    // Create mesh (1D uniform)
    RectilinearMesh mesh = RectilinearMesh::createUniform(
        1, numCells, 0.0, length);
    mesh.setBoundaryCondition(RectilinearMesh::XLow,  BoundaryCondition::Outflow);
    mesh.setBoundaryCondition(RectilinearMesh::XHigh, BoundaryCondition::Outflow);
    std::cout << "Created mesh with " << mesh.nx() << " cells.\n";

    // Allocate solution state
    SolutionState state;
    state.allocate(mesh.totalCells(), mesh.dim());

    // Create equation of state
    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0);
    std::cout << "Created ideal gas EOS" << "\n";

    // Create advective Riemann solver (pressure-free!)
    auto riemannSolver = std::make_shared<HLLCSolver>(eos, false);
    std::cout << "Created Riemann solver \n";

    // Create IGR solver
    IGRParams igrParams;
    igrParams.alphaCoeff = 1.0;       // α = αCoeff * Δx²
    igrParams.maxIterations = 5;
    igrParams.tolerance = 1e-10;
    auto igrSolver = std::make_shared<IGRSolver>(igrParams);
    std::cout << "Created IGR solver \n";

    // Explicit solver parameters
    ExplicitParams params;
    params.cfl = 0.8;
    params.maxDt = 1e-2;
    params.minDt = 1e-12;
    params.RKOrder = 1;
    params.useIGR = true;

    // Create explicit solver
    ExplicitSolver solver(riemannSolver, eos, igrSolver, params);
    std::cout << "Created explicit solver \n";

    // Initialize solution
    initializeSodProblem(mesh, state, *eos);
    std::cout << "Initialized sod shock tube problem \n";

    // Write initial condition
    writeSolution(mesh, state, "sod_t0.dat");
    std::cout << "Wrote initial conditions \n";

    // Compute reference CFL (what explicit would need)
    double maxC = 0.0;
    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        PrimitiveState W;
        W.rho = state.rho[idx];
        W.u = {state.velU[idx], 0.0, 0.0};
        W.p = state.pres[idx];
        maxC = std::max(maxC, eos->soundSpeed(W));
    }
    double dx = length / numCells;
    double explicitDt = 0.5 * dx / maxC;

    std::cout << "\nReference explicit dt (CFL=0.5, acoustic): " << explicitDt << "\n";
    std::cout << "Semi-implicit allows much larger dt!\n\n";

    // Time integration
    std::cout << "Running simulation to t = " << endTime << "...\n";
    std::cout << "Using IGR with alpha_coeff = " << igrParams.alphaCoeff << "\n\n";

    double time = 0.0;
    int step = 0;

    while (time < endTime) {
        double dt = solver.step(mesh, state, endTime - time);
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
                      << " (speedup: " << dt / explicitDt << "x)"
                      << ", max|sigma| = " << maxSigma << "\n";
        }
    }

    std::cout << "\nSimulation complete after " << step << " steps.\n";

    // Compare to explicit
    int explicitSteps = static_cast<int>(std::ceil(endTime / explicitDt));
    std::cout << "Explicit method would need ~" << explicitSteps << " steps.\n";
    std::cout << "Semi-implicit speedup: " << static_cast<double>(explicitSteps) / step << "x\n";

    // Write final solution
    writeSolution(mesh, state, "sod_final.dat");
    std::cout << "\nSolution written to sod_final.dat\n";

    // Summary statistics
    double maxSigma = 0.0;
    double minP = 1e10, maxP = 0.0;
    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        maxSigma = std::max(maxSigma, std::abs(state.sigma[idx]));
        minP = std::min(minP, state.pres[idx]);
        maxP = std::max(maxP, state.pres[idx]);
    }

    std::cout << "\nFinal statistics:\n";
    std::cout << "  Pressure range: [" << minP << ", " << maxP << "]\n";
    std::cout << "  Max |sigma|: " << maxSigma << "\n";

    return 0;
}

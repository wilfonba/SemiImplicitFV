/**
 * Sod Shock Tube Example
 *
 * Demonstrates the semi-implicit finite volume solver (Kwatra et al.)
 * with Information Geometric Regularization (IGR) on the classic
 * Sod shock tube problem.
 *
 * Key features:
 * - Flux splitting: advective (pressure-free) and pressure terms
 * - CFL based on material velocity only (not u±c)
 * - Implicit pressure solve avoids acoustic time step restriction
 * - IGR adds entropic pressure Σ to smooth discontinuities
 */

#include "RectilinearMesh.hpp"
#include "State.hpp"
#include "RiemannSolver.hpp"
#include "SemiImplicitSolver.hpp"
#include "IGR.hpp"
#include "EquationOfState.hpp"

#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>

using namespace SemiImplicitFV;

// Initialize Sod shock tube conditions
void initializeSodProblem(RectilinearMesh& mesh, const IdealGasEOS& eos) {
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

        // Primitive
        mesh.rho[idx]  = W.rho;
        mesh.velU[idx] = W.u[0];
        mesh.velV[idx] = W.u[1];
        mesh.velW[idx] = W.u[2];
        mesh.pres[idx] = W.p;
        mesh.temp[idx] = eos.temperature(W);
        mesh.sigma[idx] = W.sigma;

        // Conservative
        ConservativeState U = eos.toConservative(W);
        mesh.rhoU[idx] = U.rhoU[0];
        mesh.rhoV[idx] = U.rhoU[1];
        mesh.rhoW[idx] = U.rhoU[2];
        mesh.rhoE[idx] = U.rhoE;
    }
}

void writeSolution(const RectilinearMesh& mesh, const std::string& filename) {
    std::ofstream file(filename);
    file << "# x rho u p sigma\n";

    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        file << mesh.cellCentroidX(i) << " "
             << mesh.rho[idx] << " "
             << mesh.velU[idx] << " "
             << mesh.pres[idx] << " "
             << mesh.sigma[idx] << "\n";
    }

    file.close();
}

int main() {
    std::cout << "Semi-Implicit FV Solver (Kwatra et al.) with IGR\n";
    std::cout << "================================================\n";
    std::cout << "Sod Shock Tube Problem\n\n";

    // Problem setup
    const int numCells = 200;
    const double length = 1.0;
    const double endTime = 0.2;

    // Create mesh (1D uniform)
    std::cout << "Creating mesh with " << numCells << " cells...\n";
    RectilinearMesh mesh = RectilinearMesh::createUniform(
        1, numCells, 0.0, length);
    mesh.setBoundaryCondition(RectilinearMesh::XLow,  BoundaryCondition::Reflecting);
    mesh.setBoundaryCondition(RectilinearMesh::XHigh, BoundaryCondition::Reflecting);

    // Create equation of state
    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0);

    // Initialize solution
    std::cout << "Initializing Sod shock tube problem...\n";
    initializeSodProblem(mesh, *eos);

    // Create advective Riemann solver (pressure-free!)
    auto riemannSolver = std::make_shared<RusanovAdvectiveSolver>();

    // Create pressure solver
    auto pressureSolver = std::make_shared<GaussSeidelPressureSolver>();

    // Create IGR solver
    IGRParams igrParams;
    igrParams.alphaCoeff = 1.0;       // α = αCoeff * Δx²
    igrParams.maxIterations = 5;
    igrParams.tolerance = 1e-10;
    auto igrSolver = std::make_shared<IGRSolver>(igrParams);

    // Semi-implicit solver parameters
    SemiImplicitParams params;
    params.cfl = 0.8;                  // Can use larger CFL (no acoustic restriction!)
    params.maxPressureIters = 100;
    params.pressureTol = 1e-8;
    params.useIGR = true;

    // Create semi-implicit solver
    SemiImplicitSolver solver(riemannSolver, pressureSolver, eos, igrSolver, params);

    // Write initial condition
    writeSolution(mesh, "sod_t0.dat");

    // Compute reference CFL (what explicit would need)
    double maxC = 0.0;
    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        PrimitiveState W;
        W.rho = mesh.rho[idx];
        W.u = {mesh.velU[idx], mesh.velV[idx], mesh.velW[idx]};
        W.p = mesh.pres[idx];
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
        double dt = solver.step(mesh, endTime - time);
        time += dt;
        step++;

        if (step % 20 == 0 || step == 1) {
            double maxSigma = 0.0;
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, 0, 0);
                maxSigma = std::max(maxSigma, std::abs(mesh.sigma[idx]));
            }

            std::cout << "  Step " << step << ": t = " << time
                      << ", dt = " << dt
                      << " (speedup: " << dt / explicitDt << "x)"
                      << ", pressure iters = " << solver.lastPressureIterations()
                      << ", max|sigma| = " << maxSigma << "\n";
        }
    }

    std::cout << "\nSimulation complete after " << step << " steps.\n";

    // Compare to explicit
    int explicitSteps = static_cast<int>(std::ceil(endTime / explicitDt));
    std::cout << "Explicit method would need ~" << explicitSteps << " steps.\n";
    std::cout << "Semi-implicit speedup: " << static_cast<double>(explicitSteps) / step << "x\n";

    // Write final solution
    writeSolution(mesh, "sod_final.dat");
    std::cout << "\nSolution written to sod_final.dat\n";

    // Summary statistics
    double maxSigma = 0.0;
    double minP = 1e10, maxP = 0.0;
    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        maxSigma = std::max(maxSigma, std::abs(mesh.sigma[idx]));
        minP = std::min(minP, mesh.pres[idx]);
        maxP = std::max(maxP, mesh.pres[idx]);
    }

    std::cout << "\nFinal statistics:\n";
    std::cout << "  Pressure range: [" << minP << ", " << maxP << "]\n";
    std::cout << "  Max |sigma|: " << maxSigma << "\n";

    return 0;
}

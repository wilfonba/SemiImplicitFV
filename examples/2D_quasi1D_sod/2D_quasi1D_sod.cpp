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

// 2D Riemann problem Configuration 3 (Lax & Liu, 1998)
// Four constant states separated by discontinuities at (x,y) = (0.5, 0.5)
void initializeRiemannProblem(const RectilinearMesh& mesh, SolutionState& state, const IdealGasEOS& eos, int testDir) {
    double xMid = 0.5;
    double yMid = 0.5;

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

    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            std::size_t idx = mesh.index(i, j, 0);
            double x = mesh.cellCentroidX(i);
            double y = mesh.cellCentroidY(j);

            const PrimitiveState* W;
            if (testDir == 1) {
                if (x <= xMid) W = &left;
                else           W = &right;
            } else {
                if (y <= xMid) W = &left;
                else           W = &right;
            }

            PrimitiveState Wt = *W;
            Wt.T = eos.temperature(*W);
            state.setPrimitiveState(idx, Wt);

            ConservativeState U = eos.toConservative(*W);
            state.setConservativeState(idx, U);
        }
    }
}

int main() {
    const int N1 = 200;
    const int N2 = 200;
    const double length = 1.0;
    const double endTime = 0.2;
    int testDir = 1;

    SimulationConfig config;
    config.dim = 2;
    config.nGhost = 4;
    config.RKOrder = 3;
    config.useIGR = true;
    config.reconOrder = ReconstructionOrder::UPWIND5;
    config.explicitParams.cfl = 0.1;
    config.igrParams.alphaCoeff = 10.0;
    config.igrParams.IGRIters = 5;
    config.validate();

    RectilinearMesh mesh = RectilinearMesh::createUniform(
        config, N1, 0.0, length, N2, 0.0, length);
    mesh.setBoundaryCondition(RectilinearMesh::XLow,  BoundaryCondition::Outflow);
    mesh.setBoundaryCondition(RectilinearMesh::XHigh, BoundaryCondition::Outflow);
    mesh.setBoundaryCondition(RectilinearMesh::YLow,  BoundaryCondition::Outflow);
    mesh.setBoundaryCondition(RectilinearMesh::YHigh, BoundaryCondition::Outflow);
    std::cout << "Created " << mesh.nx() << "x" << mesh.ny() << " mesh.\n";

    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);
    auto riemannSolver = std::make_shared<LFSolver>(eos, config);
    auto igrSolver = std::make_shared<IGRSolver>(config.igrParams);

    ExplicitSolver solver(mesh, riemannSolver, eos, igrSolver, config);
    initializeRiemannProblem(mesh, state, *eos, testDir);
    state.smoothFields(mesh, 10);

    // Initialize VTK time-series file
    VTKWriter::writePVD("VTK/quasi1D_sod.pvd", "w");
    VTKWriter::writeVTR("VTK/quasi1D_sod_0.vtr", mesh, state);
    VTKWriter::writePVD("VTK/quasi1D_sod.pvd", "a", 0.0, "quasi1D_sod_0.vtr");
    int fileNum = 1;
    const double outputInterval = 0.002;
    const int printInterval = 20;

    std::cout << "Running simulation to t = " << endTime << "...\n\n";

    double time = 0.0;
    int step = 0;

    while (time < endTime) {
        double dt = solver.step(config, mesh, state, endTime - time);
        time += dt;
        step++;

        if (std::abs(time - fileNum * outputInterval) <= dt) {
            // Write VTK output
            std::string vtrFile = "quasi1D_sod_" + std::to_string(fileNum) + ".vtr";
            VTKWriter::writeVTR("VTK/" + vtrFile, mesh, state);
            VTKWriter::writePVD("VTK/quasi1D_sod.pvd", "a", time, vtrFile);
            fileNum++;
        }

        if (step % printInterval == 0 || step == 1) {
            double maxSigma = 0.0;
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idx = mesh.index(i, j, 0);
                    maxSigma = std::max(maxSigma, std::abs(state.sigma[idx]));
                }
            }

            std::cout << "  Step " << step << ": t = " << time
                      << ", dt = " << dt
                      << ", max|sigma| = " << maxSigma << "\n";
        }

    }

    std::cout << "\nSimulation complete after " << step << " steps.\n";
    VTKWriter::writePVD("VTK/quasi1D_sod.pvd", "close");

    return 0;
}

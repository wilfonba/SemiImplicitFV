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

// Four constant states separated by discontinuities at (x,y) = (0.75, 0.75)
void initializeRiemannProblem(const RectilinearMesh& mesh, SolutionState& state, const IdealGasEOS& eos) {
    double xMid = 0.75;
    double yMid = 0.75;

    // Quadrant 1: upper-right (x > 0.5, y > 0.5)
    PrimitiveState q1;
    q1.rho = 1.5;
    q1.u = {0.0, 0.0, 0.0};
    q1.p = 1.5;
    q1.sigma = 0.0;

    // Quadrant 2: upper-left (x < 0.5, y > 0.5)
    PrimitiveState q2;
    q2.rho = 0.5323;
    q2.u = {1.206, 0.0, 0.0};
    q2.p = 0.3;
    q2.sigma = 0.0;

    // Quadrant 3: lower-left (x < 0.5, y < 0.5)
    PrimitiveState q3;
    q3.rho = 0.138;
    q3.u = {1.206, 1.206, 0.0};
    q3.p = 0.029;
    q3.sigma = 0.0;

    // Quadrant 4: lower-right (x > 0.5, y < 0.5)
    PrimitiveState q4;
    q4.rho = 0.5323;
    q4.u = {0.0, 1.206, 0.0};
    q4.p = 0.3;
    q4.sigma = 0.0;

    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            std::size_t idx = mesh.index(i, j, 0);
            double x = mesh.cellCentroidX(i);
            double y = mesh.cellCentroidY(j);

            const PrimitiveState* W;
            if (x >= xMid && y >= yMid)      W = &q1;
            else if (x < xMid && y >= yMid)  W = &q2;
            else if (x < xMid && y < yMid)   W = &q3;
            else                             W = &q4;

            PrimitiveState Wt = *W;
            Wt.T = eos.temperature(*W);
            state.setPrimitiveState(idx, Wt);

            ConservativeState U = eos.toConservative(*W);
            state.setConservativeState(idx, U);
        }
    }
}

int main() {
    const int N = 256;
    const double length = 1.0;
    const double endTime = 0.8;
    const double outputInterval = 0.008;
    const int printInterval = 1;

    SimulationConfig config;
    config.dim = 2;
    config.nGhost = 4;
    config.RKOrder = 3;
    config.useIGR = true;

    RectilinearMesh mesh = RectilinearMesh::createUniform(
        config, N, 0.0, length, N, 0.0, length);
    mesh.setBoundaryCondition(RectilinearMesh::XLow,  BoundaryCondition::Outflow);
    mesh.setBoundaryCondition(RectilinearMesh::XHigh, BoundaryCondition::Outflow);
    mesh.setBoundaryCondition(RectilinearMesh::YLow,  BoundaryCondition::Outflow);
    mesh.setBoundaryCondition(RectilinearMesh::YHigh, BoundaryCondition::Outflow);
    std::cout << "Created " << mesh.nx() << "x" << mesh.ny() << " mesh.\n";

    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);
    auto riemannSolver = std::make_shared<LFSolver>(eos, true, config);

    IGRParams igrParams;
    igrParams.alphaCoeff = 10.0;
    igrParams.IGRIters= 5;
    auto igrSolver = std::make_shared<IGRSolver>(igrParams);

    ExplicitParams params;
    params.cfl = 0.6;
    params.reconOrder = ReconstructionOrder::UPWIND3;

    ExplicitSolver solver(mesh, riemannSolver, eos, igrSolver, params);
    initializeRiemannProblem(mesh, state, *eos);
    state.smoothFields(mesh, 10);

    // Initialize VTK time-series file
    VTKWriter::writePVD("VTK/riemann2d.pvd", "w");
    VTKWriter::writeVTR("VTK/riemann2d_0.vtr", mesh, state);
    VTKWriter::writePVD("VTK/riemann2d.pvd", "a", 0.0, "riemann2d_0.vtr");
    int fileNum = 1;

    std::cout << "Running simulation to t = " << endTime << "...\n\n";

    double time = 0.0;
    int step = 0;

    while (time < endTime) {
        double dt = solver.step(config, mesh, state, endTime - time);
        time += dt;
        step++;

        if (std::abs(time - fileNum * outputInterval) <= dt) {
            // Write VTK output
            std::string vtrFile = "riemann2d_" + std::to_string(fileNum) + ".vtr";
            VTKWriter::writeVTR("VTK/" + vtrFile, mesh, state);
            VTKWriter::writePVD("VTK/riemann2d.pvd", "a", time, vtrFile);
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

    VTKWriter::writePVD("VTK/riemann2d.pvd", "close");

    return 0;
}

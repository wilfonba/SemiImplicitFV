#include "InputParser.hpp"
#include "ICPatch.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "IdealGasEOS.hpp"
#include "StiffenedGasEOS.hpp"
#include "LFSolver.hpp"
#include "RusanovSolver.hpp"
#include "HLLCSolver.hpp"
#include "ExplicitSolver.hpp"
#include "SemiImplicitSolver.hpp"
#include "GaussSeidelPressureSolver.hpp"
#include "JacobiPressureSolver.hpp"
#ifdef SIFV_HAS_HYPRE
#include "HyprePressureSolver.hpp"
#endif
#include "IGR.hpp"
#include "SimulationConfig.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"
#include "RKTimeStepping.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <functional>

using namespace SemiImplicitFV;

static void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " <input.jsonc>\n";
    std::cerr << "\nGeneric driver for SemiImplicitFV simulations.\n";
    std::cerr << "Reads a JSON/JSONC input file and runs the simulation.\n";
}

int main(int argc, char** argv) {
    Runtime rt(argc, argv);

    // Find the input file argument (skip MPI-injected args)
    std::string inputFile;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        // Skip common MPI launcher args
        if (arg[0] != '-') {
            inputFile = arg;
            break;
        }
    }

    if (inputFile.empty()) {
        if (rt.isRoot()) printUsage(argv[0]);
        return 1;
    }

    // ---- Parse input ----
    InputData input;
    try {
        input = InputParser::parseFile(inputFile);
    } catch (const std::exception& e) {
        if (rt.isRoot()) {
            std::cerr << "Error parsing input file '" << inputFile << "':\n"
                      << "  " << e.what() << "\n";
        }
        return 1;
    }

    auto& config = input.config;

    // ---- Validate config ----
    try {
        config.validate();
    } catch (const std::exception& e) {
        if (rt.isRoot()) {
            std::cerr << "Configuration error: " << e.what() << "\n";
        }
        return 1;
    }

    // ---- Create mesh ----
    const auto& mp = input.meshParams;
    RectilinearMesh mesh = [&]() {
        if (config.dim == 1) {
            return rt.createUniformMesh(config, mp.nx, mp.xMin, mp.xMax);
        } else if (config.dim == 2) {
            return rt.createUniformMesh(config, mp.nx, mp.xMin, mp.xMax,
                                         mp.ny, mp.yMin, mp.yMax);
        } else {
            return rt.createUniformMesh(config, mp.nx, mp.xMin, mp.xMax,
                                         mp.ny, mp.yMin, mp.yMax,
                                         mp.nz, mp.zMin, mp.zMax);
        }
    }();

    // ---- Set boundary conditions ----
    for (int f = 0; f < 6; ++f) {
        rt.setBoundaryCondition(mesh, f, input.bcParams.bc[f]);
    }

    rt.print("=== SemiImplicitFV Driver ===\n");
    rt.print("  Input file: ", inputFile, "\n");
    rt.print("  Dimension:  ", config.dim, "D\n");
    rt.print("  Grid:       ", mp.nx);
    if (config.dim >= 2) rt.print(" x ", mp.ny);
    if (config.dim >= 3) rt.print(" x ", mp.nz);
    rt.print("\n");
    rt.print("  Solver:     ", config.semiImplicit ? "Semi-implicit" : "Explicit", "\n");
    rt.print("  Riemann:    ", input.riemannSolverType, "\n");
    rt.print("  End time:   ", input.timeLoopParams.endTime, "\n");
    if (config.isMultiPhase()) {
        rt.print("  Phases:     ", config.multiPhaseParams.nPhases, "\n");
    }
    rt.print("\n");

    // ---- Allocate solution state ----
    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    // ---- Create EOS ----
    std::shared_ptr<EquationOfState> eos;
    if (input.eosParams.type == "StiffenedGas") {
        eos = std::make_shared<StiffenedGasEOS>(
            input.eosParams.gamma, input.eosParams.pInf, input.eosParams.R, config);
    } else {
        eos = std::make_shared<IdealGasEOS>(
            input.eosParams.gamma, input.eosParams.R, config);
    }

    // ---- Create Riemann solver ----
    std::shared_ptr<RiemannSolver> riemannSolver;
    if (input.riemannSolverType == "LF") {
        riemannSolver = std::make_shared<LFSolver>(eos, config);
    } else if (input.riemannSolverType == "Rusanov") {
        riemannSolver = std::make_shared<RusanovSolver>(eos, config);
    } else {
        riemannSolver = std::make_shared<HLLCSolver>(eos, config);
    }

    // ---- Create IGR solver (if enabled) ----
    std::shared_ptr<IGRSolver> igrSolver;
    if (config.useIGR) {
        igrSolver = std::make_shared<IGRSolver>(config.igrParams);
    }

    // ---- Apply initial conditions ----
    applyInitialConditions(mesh, state, *eos, config, input.defaultState, input.patches);

    // ---- Build solver and step function ----
    std::function<double(double)> stepFn;
    std::unique_ptr<ExplicitSolver> explicitSolver;
    std::unique_ptr<SemiImplicitSolver> semiImplicitSolver;

    if (config.semiImplicit) {
        // Create pressure solver
        std::shared_ptr<PressureSolver> pressureSolver;
        if (input.pressureSolverType == "Jacobi") {
            pressureSolver = std::make_shared<JacobiPressureSolver>();
        } else if (input.pressureSolverType == "Hypre") {
#ifdef SIFV_HAS_HYPRE
            if (config.dim == 1) {
                if (rt.isRoot()) {
                    std::cerr << "Error: Hypre pressure solver is not supported for 1D cases.\n"
                              << "  Use GaussSeidel or Jacobi instead.\n";
                }
                return 1;
            }
            if (rt.size() > 1) {
                pressureSolver = std::make_shared<HyprePressureSolver>(
                    rt.mpiContext().comm(),
                    rt.mpiContext().localExtent(),
                    std::array<int,3>{0, 0, 0});
            } else {
                pressureSolver = std::make_shared<HyprePressureSolver>();
            }
#else
            if (rt.isRoot()) {
                std::cerr << "Error: Hypre pressure solver requested but not built.\n"
                          << "  Rebuild with: ./run_case.sh --hypre ...\n";
            }
            return 1;
#endif
        } else {
            pressureSolver = std::make_shared<GaussSeidelPressureSolver>();
        }

        semiImplicitSolver = std::make_unique<SemiImplicitSolver>(
            mesh, riemannSolver, pressureSolver, eos, igrSolver, config);
        rt.attachSolver(*semiImplicitSolver, mesh);
        stepFn = [&](double targetDt) {
            return semiImplicitSolver->step(config, mesh, state, targetDt);
        };
    } else {
        explicitSolver = std::make_unique<ExplicitSolver>(
            mesh, riemannSolver, eos, igrSolver, config);
        rt.attachSolver(*explicitSolver, mesh);
        stepFn = [&](double targetDt) {
            return explicitSolver->step(config, mesh, state, targetDt);
        };
    }

    // ---- Smoothing (after attachSolver creates HaloExchange) ----
    if (input.smoothingParams.iterations > 0) {
        rt.smoothFields(state, mesh, input.smoothingParams.iterations, config);
    }

    // ---- VTK output ----
    VTKSession vtk(rt, input.outputParams.baseName, mesh, config,
                   input.outputParams.directory);

    // ---- Run time loop ----
    TimeLoopParams tlp;
    tlp.endTime        = input.timeLoopParams.endTime;
    tlp.outputInterval = input.timeLoopParams.outputInterval;
    tlp.printInterval  = input.timeLoopParams.printInterval;
    tlp.checkNaN       = input.timeLoopParams.checkNaN;

    if (config.semiImplicit) {
        tlp.acousticDtFn = [&]() {
            return computeAcousticTimeStep(mesh, state, *eos, config,
                                           1.0, 1e30, rt.mpiContext().comm());
        };
    }

    runTimeLoop(rt, config, mesh, state, vtk, stepFn, tlp);

    return 0;
}

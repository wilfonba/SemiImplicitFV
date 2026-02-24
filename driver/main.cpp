#include "InputParser.hpp"
#include "ICPatch.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "EquationOfState.hpp"
#include "RiemannSolver.hpp"
#include "ExplicitSolver.hpp"
#include "SemiImplicitSolver.hpp"
#include "PressureSolver.hpp"
#include "SimulationConfig.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"
#include "RKTimeStepping.hpp"
#include "Checkpoint.hpp"
#include "MPIContext.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

static void printUsage(const char* progName) {
    std::fprintf(stderr, "Usage: %s <input.jsonc>\n", progName);
    std::fprintf(stderr, "\nGeneric driver for SemiImplicitFV simulations.\n");
    std::fprintf(stderr, "Reads a JSON/JSONC input file and runs the simulation.\n");
}

/* Context for the step function callback */
struct StepContext {
    ExplicitSolverWork* explSolver;
    SemiImplicitSolverWork* siSolver;
    SimulationConfig* config;
    RectilinearMesh* mesh;
    SolutionState* state;
};

static double step_callback(double targetDt, void* ctx) {
    StepContext* sc = (StepContext*)ctx;
    if (sc->explSolver)
        return explicit_step(sc->explSolver, sc->config, sc->mesh, sc->state, targetDt);
    else
        return semi_implicit_step(sc->siSolver, sc->config, sc->mesh, sc->state, targetDt);
}

/* Context for the acoustic dt callback */
struct AcousticDtContext {
    RectilinearMesh* mesh;
    SolutionState* state;
    EOSData* eos;
    SimulationConfig* config;
    MPI_Comm comm;
};

static double acoustic_dt_callback(void* ctx) {
    AcousticDtContext* ac = (AcousticDtContext*)ctx;
    return computeAcousticTimeStep_config_mpi(ac->mesh, ac->state, ac->eos,
                                              ac->config, 1.0, 1e30, ac->comm);
}

int main(int argc, char** argv) {
    Runtime rt;
    runtime_init(&rt, &argc, &argv);

    /* Find the input file argument (skip MPI-injected args) */
    const char* inputFile = NULL;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') {
            inputFile = argv[i];
            break;
        }
    }

    if (!inputFile) {
        if (rt.rank == 0) printUsage(argv[0]);
        runtime_free(&rt);
        return 1;
    }

    /* ---- Parse input ---- */
    InputData input;
    try {
        input = parse_input_file(inputFile);
    } catch (const std::exception& e) {
        if (rt.rank == 0) {
            std::fprintf(stderr, "Error parsing input file '%s':\n  %s\n",
                         inputFile, e.what());
        }
        runtime_free(&rt);
        return 1;
    }

    SimulationConfig* config = &input.config;

    /* ---- Validate config ---- */
    try {
        config_validate(config);
    } catch (const std::exception& e) {
        if (rt.rank == 0) {
            std::fprintf(stderr, "Configuration error: %s\n", e.what());
        }
        input_data_free(&input);
        runtime_free(&rt);
        return 1;
    }

    /* ---- Create mesh ---- */
    MeshParams* mp = &input.meshParams;

    /* Derive periodic flags from boundary conditions for MPI topology */
    int periods[3] = {
        (input.bc[XLOW] == BC_PERIODIC) ? 1 : 0,
        (input.bc[YLOW] == BC_PERIODIC) ? 1 : 0,
        (input.bc[ZLOW] == BC_PERIODIC) ? 1 : 0
    };

    RectilinearMesh mesh;
    if (config->dim == 1) {
        runtime_create_uniform_mesh_1d(&rt, &mesh, config,
                                       mp->nx, mp->xMin, mp->xMax, periods);
    } else if (config->dim == 2) {
        runtime_create_uniform_mesh_2d(&rt, &mesh, config,
                                       mp->nx, mp->xMin, mp->xMax,
                                       mp->ny, mp->yMin, mp->yMax, periods);
    } else {
        runtime_create_uniform_mesh_3d(&rt, &mesh, config,
                                       mp->nx, mp->xMin, mp->xMax,
                                       mp->ny, mp->yMin, mp->yMax,
                                       mp->nz, mp->zMin, mp->zMax, periods);
    }

    /* ---- Set boundary conditions ---- */
    for (int f = 0; f < 6; ++f) {
        runtime_set_bc(&rt, &mesh, f, input.bc[f]);
    }

    {
        char buf[512];
        runtime_print(&rt, "=== SemiImplicitFV Driver ===\n");
        std::snprintf(buf, sizeof(buf), "  Input file: %s\n", inputFile);
        runtime_print(&rt, buf);
        std::snprintf(buf, sizeof(buf), "  Dimension:  %dD\n", config->dim);
        runtime_print(&rt, buf);
        if (config->dim == 1) {
            std::snprintf(buf, sizeof(buf), "  Grid:       %d\n", mp->nx);
        } else if (config->dim == 2) {
            std::snprintf(buf, sizeof(buf), "  Grid:       %d x %d\n", mp->nx, mp->ny);
        } else {
            std::snprintf(buf, sizeof(buf), "  Grid:       %d x %d x %d\n", mp->nx, mp->ny, mp->nz);
        }
        runtime_print(&rt, buf);
        std::snprintf(buf, sizeof(buf), "  Solver:     %s\n",
                      config->semiImplicit ? "Semi-implicit" : "Explicit");
        runtime_print(&rt, buf);
        const char* rsNames[] = {"LF", "Rusanov", "HLLC"};
        std::snprintf(buf, sizeof(buf), "  Riemann:    %s\n", rsNames[input.riemannSolverType]);
        runtime_print(&rt, buf);
        std::snprintf(buf, sizeof(buf), "  End time:   %g\n", input.timeLoopParams.endTime);
        runtime_print(&rt, buf);
        if (config_is_multi_phase(config)) {
            std::snprintf(buf, sizeof(buf), "  Phases:     %d\n", config->multiPhaseParams.nPhases);
            runtime_print(&rt, buf);
        }
        runtime_print(&rt, "\n");
    }

    /* ---- Allocate solution state ---- */
    SolutionState state;
    solution_state_init(&state, mesh_total_cells(&mesh), config);

    /* ---- Create EOS ---- */
    EOSData eos;
    if (std::strcmp(input.eosParams.type, "StiffenedGas") == 0) {
        eos = eos_create_stiffened_gas(input.eosParams.gamma, input.eosParams.pInf,
                                       input.eosParams.R, config->dim);
    } else {
        eos = eos_create_ideal_gas(input.eosParams.gamma, input.eosParams.R, config->dim);
    }

    /* ---- Apply initial conditions or load checkpoint ---- */
    int restarting = (input.restartParams.file[0] != '\0') ? 1 : 0;
    if (restarting) {
        /* Replace {rank} placeholder with zero-padded MPI rank */
        char ckptFile[256];
        std::strncpy(ckptFile, input.restartParams.file, sizeof(ckptFile) - 1);
        ckptFile[sizeof(ckptFile) - 1] = '\0';

        char* pos = std::strstr(ckptFile, "{rank}");
        if (pos) {
            char rankStr[8];
            std::snprintf(rankStr, sizeof(rankStr), "%04d", rt.rank);
            /* Build replacement string */
            char tmp[256];
            size_t prefix_len = (size_t)(pos - ckptFile);
            std::memcpy(tmp, ckptFile, prefix_len);
            std::memcpy(tmp + prefix_len, rankStr, 4);
            std::strcpy(tmp + prefix_len + 4, pos + 6); /* 6 = strlen("{rank}") */
            std::strcpy(ckptFile, tmp);
        }
        load_checkpoint(ckptFile, &mesh, &state, config);
        state_cons_to_prim(&state, &mesh, &eos);

        char buf[512];
        std::snprintf(buf, sizeof(buf), "  Restarting from checkpoint: %s\n", ckptFile);
        runtime_print(&rt, buf);
        std::snprintf(buf, sizeof(buf), "  Restart time: %g, step: %d\n\n", config->time, config->step);
        runtime_print(&rt, buf);
    } else {
        apply_initial_conditions(&mesh, &state, &eos, config,
                                 &input.defaultState, input.patches, input.nPatches);
    }

    /* ---- Build solver and step function ---- */
    ExplicitSolverWork explSolver;
    SemiImplicitSolverWork siSolver;
    int useSemiImplicit = config->semiImplicit;

    StepContext stepCtx;
    std::memset(&stepCtx, 0, sizeof(stepCtx));
    stepCtx.config = config;
    stepCtx.mesh   = &mesh;
    stepCtx.state  = &state;

    PressureSolverData pressureSolver;

    if (useSemiImplicit) {
        /* Create pressure solver */
        if (input.pressureSolverType == PS_JACOBI) {
            pressure_solver_init(&pressureSolver, PS_JACOBI);
        } else if (input.pressureSolverType == PS_PETSC) {
#ifdef SIFV_HAS_PETSC
            if (rt.size > 1) {
                pressure_solver_init_petsc_mpi(&pressureSolver,
                                               (void*)&rt.mpiCtx->cartComm,
                                               rt.mpiCtx->localExtent,
                                               periods,
                                               rt.mpiCtx->dims);
            } else {
                pressure_solver_init_petsc(&pressureSolver);
            }
#else
            if (rt.rank == 0) {
                std::fprintf(stderr, "Error: PETSc pressure solver requested but not built.\n"
                                     "  Rebuild with: ./run_case.sh --petsc ...\n");
            }
            input_data_free(&input);
            mesh_free(&mesh);
            solution_state_free(&state);
            runtime_free(&rt);
            return 1;
#endif
        } else {
            pressure_solver_init(&pressureSolver, PS_GAUSS_SEIDEL);
        }

        semi_implicit_solver_init(&siSolver, &mesh, &eos,
                                  input.riemannSolverType,
                                  &pressureSolver, NULL, config);
        runtime_attach_semi_implicit(&rt, &siSolver, &mesh);
        stepCtx.siSolver = &siSolver;
    } else {
        explicit_solver_init(&explSolver, &mesh, &eos,
                             input.riemannSolverType, NULL, config);
        runtime_attach_explicit(&rt, &explSolver, &mesh);
        stepCtx.explSolver = &explSolver;
    }

    /* ---- Smoothing (after attach creates HaloExchange) ---- */
    if (!restarting && input.smoothingParams.iterations > 0) {
        runtime_smooth_fields_config(&rt, &state, &mesh,
                                     input.smoothingParams.iterations, config);
    }

    /* ---- VTK output ---- */
    VTKSession vtk;
    vtk_session_init(&vtk, &rt, input.outputParams.baseName, &mesh, config,
                     input.outputParams.directory, input.outputParams.format);

    /* ---- Run time loop ---- */
    TimeLoopParams tlp = time_loop_params_defaults();
    tlp.endTime        = input.timeLoopParams.endTime;
    tlp.outputInterval = input.timeLoopParams.outputInterval;
    tlp.printInterval  = input.timeLoopParams.printInterval;
    tlp.checkNaN       = input.timeLoopParams.checkNaN;
    tlp.checkpoint     = input.restartParams.checkpoint;
    tlp.startTime      = config->time;

    AcousticDtContext acousticCtx;
    if (useSemiImplicit) {
        acousticCtx.mesh   = &mesh;
        acousticCtx.state  = &state;
        acousticCtx.eos    = &eos;
        acousticCtx.config = config;
        acousticCtx.comm   = (rt.mpiCtx) ? rt.mpiCtx->cartComm : MPI_COMM_WORLD;
        tlp.acousticDtFn   = acoustic_dt_callback;
        tlp.acousticDtCtx  = &acousticCtx;
    }

    run_time_loop(&rt, config, &mesh, &state, &vtk,
                  step_callback, &stepCtx, &tlp);

    /* ---- Cleanup ---- */
    vtk_session_finalize(&vtk);
    if (useSemiImplicit) {
        semi_implicit_solver_free(&siSolver);
        pressure_solver_free(&pressureSolver);
    } else {
        explicit_solver_free(&explSolver);
    }
    solution_state_free(&state);
    mesh_free(&mesh);
    input_data_free(&input);
    runtime_free(&rt);

    return 0;
}

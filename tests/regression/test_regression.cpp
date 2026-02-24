#include <gtest/gtest.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

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
#include "PETScPressureSolver.hpp"
#include "SimulationConfig.hpp"
#include "Runtime.hpp"
#include "RKTimeStepping.hpp"
#include "MPIContext.hpp"
#include "HaloExchange.hpp"

/* ---- Step callback (same pattern as driver/main.cpp) ---- */

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

/* ---- Pointwise field data collection ---- */

struct CellData {
    int i, j, k;
    double rho, rhoU, rhoV, rhoW, rhoE;
    /* Multi-phase: up to MAX_PHASES of each */
    int nPhases;
    double alpha[MAX_PHASES];
    double alphaRho[MAX_PHASES];
};

/* Gather interior cells from this rank in natural (i,j,k) order, using global indices */
static std::vector<CellData> gatherLocalCells(
    const RectilinearMesh* mesh,
    const SolutionState* state,
    const SimulationConfig* config,
    const Runtime* rt)
{
    std::vector<CellData> cells;
    int nPhases = config->multiPhaseParams.nPhases;
    size_t tc = state->totalCells;

    /* Global offsets from localExtent (VTK extent: [xStart,xEnd,yStart,yEnd,zStart,zEnd]) */
    int gi0 = rt->mpiCtx ? rt->mpiCtx->localExtent[0] : 0;
    int gj0 = rt->mpiCtx ? rt->mpiCtx->localExtent[2] : 0;
    int gk0 = rt->mpiCtx ? rt->mpiCtx->localExtent[4] : 0;

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);
                CellData c;
                memset(&c, 0, sizeof(c));
                c.i = gi0 + i; c.j = gj0 + j; c.k = gk0 + k;
                c.rho = state->rho[idx];
                c.rhoU = state->rhoU[idx];
                c.rhoV = (config->dim >= 2) ? state->rhoV[idx] : 0.0;
                c.rhoW = (config->dim >= 3) ? state->rhoW[idx] : 0.0;
                c.rhoE = state->rhoE[idx];
                c.nPhases = nPhases;
                for (int ph = 0; ph < nPhases; ++ph) {
                    c.alpha[ph] = state->alpha[ph * tc + idx];
                    c.alphaRho[ph] = state->alphaRho[ph * tc + idx];
                }
                cells.push_back(c);
            }
        }
    }
    return cells;
}

/* Write reference file (rank 0 only, single-rank data) */
static void writeReference(const char* path, const char* caseName,
                           int steps, int dim, int nPhases,
                           const std::vector<CellData>& cells)
{
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open %s for writing\n", path);
        return;
    }

    fprintf(fp, "# case: %s  steps: %d  dims: %d  nPhases: %d  cells: %zu\n",
            caseName, steps, dim, nPhases, cells.size());

    /* Header line */
    fprintf(fp, "# i j k rho rhoU rhoV rhoW rhoE");
    for (int ph = 0; ph < nPhases; ++ph)
        fprintf(fp, " alpha%d alphaRho%d", ph, ph);
    fprintf(fp, "\n");

    for (const auto& c : cells) {
        fprintf(fp, "%d %d %d %.17g %.17g %.17g %.17g %.17g",
                c.i, c.j, c.k, c.rho, c.rhoU, c.rhoV, c.rhoW, c.rhoE);
        for (int ph = 0; ph < nPhases; ++ph)
            fprintf(fp, " %.17g %.17g", c.alpha[ph], c.alphaRho[ph]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

/* Read reference file, return cell data */
static std::vector<CellData> readReference(const char* path, int* nPhasesOut) {
    std::vector<CellData> cells;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "Cannot open reference file: %s\n", path);
        return cells;
    }

    std::string line;
    int nPhases = 0;

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') {
            /* Parse nPhases from header */
            if (line.find("nPhases:") != std::string::npos) {
                sscanf(line.c_str() + line.find("nPhases:") + 8, "%d", &nPhases);
            }
            continue;
        }

        CellData c;
        memset(&c, 0, sizeof(c));
        c.nPhases = nPhases;

        std::istringstream iss(line);
        iss >> c.i >> c.j >> c.k >> c.rho >> c.rhoU >> c.rhoV >> c.rhoW >> c.rhoE;
        for (int ph = 0; ph < nPhases; ++ph)
            iss >> c.alpha[ph] >> c.alphaRho[ph];

        cells.push_back(c);
    }

    if (nPhasesOut) *nPhasesOut = nPhases;
    return cells;
}

/* Compare two cell arrays pointwise */
static bool compareCells(const std::vector<CellData>& computed,
                         const std::vector<CellData>& reference,
                         int nPhases, double relTol, int maxErrors)
{
    if (computed.size() != reference.size()) {
        fprintf(stderr, "  Cell count mismatch: computed=%zu reference=%zu\n",
                computed.size(), reference.size());
        return false;
    }

    int errors = 0;
    auto checkField = [&](double a, double b, const char* name, int i, int j, int k) {
        double scale = std::max(std::abs(b), 1e-15);
        double relErr = std::abs(a - b) / scale;
        if (relErr > relTol && std::abs(a - b) > 1e-14) {
            if (errors < maxErrors) {
                fprintf(stderr, "  MISMATCH at (%d,%d,%d) %s: computed=%.15g reference=%.15g relErr=%.3e\n",
                        i, j, k, name, a, b, relErr);
            }
            ++errors;
        }
    };

    for (size_t c = 0; c < computed.size(); ++c) {
        const auto& comp = computed[c];
        const auto& ref = reference[c];
        checkField(comp.rho,  ref.rho,  "rho",  ref.i, ref.j, ref.k);
        checkField(comp.rhoU, ref.rhoU, "rhoU", ref.i, ref.j, ref.k);
        checkField(comp.rhoV, ref.rhoV, "rhoV", ref.i, ref.j, ref.k);
        checkField(comp.rhoW, ref.rhoW, "rhoW", ref.i, ref.j, ref.k);
        checkField(comp.rhoE, ref.rhoE, "rhoE", ref.i, ref.j, ref.k);
        for (int ph = 0; ph < nPhases; ++ph) {
            char buf[32];
            snprintf(buf, sizeof(buf), "alpha%d", ph);
            checkField(comp.alpha[ph], ref.alpha[ph], buf, ref.i, ref.j, ref.k);
            snprintf(buf, sizeof(buf), "alphaRho%d", ph);
            checkField(comp.alphaRho[ph], ref.alphaRho[ph], buf, ref.i, ref.j, ref.k);
        }
    }

    if (errors > 0) {
        fprintf(stderr, "  Total mismatches: %d (showed first %d)\n",
                errors, std::min(errors, maxErrors));
    }
    return errors == 0;
}

/* ---- Run a case for N steps and return cell data ---- */

static std::vector<CellData> runCase(const char* jsonFile,
                                     int targetSteps,
                                     int* nPhasesOut,
                                     int* dimOut)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Parse input */
    InputData input = parse_input_file(jsonFile);
    SimulationConfig* config = &input.config;
    config_validate(config);

    /* Create runtime (reuse MPI_COMM_WORLD) */
    Runtime rt;
    memset(&rt, 0, sizeof(rt));
    rt.rank = rank;
    rt.size = size;

    MeshParams* mp = &input.meshParams;
    int periods[3] = {
        (input.bc[XLOW] == BC_PERIODIC) ? 1 : 0,
        (input.bc[YLOW] == BC_PERIODIC) ? 1 : 0,
        (input.bc[ZLOW] == BC_PERIODIC) ? 1 : 0
    };

    RectilinearMesh mesh;
    if (config->dim == 1)
        runtime_create_uniform_mesh_1d(&rt, &mesh, config, mp->nx, mp->xMin, mp->xMax, periods);
    else if (config->dim == 2)
        runtime_create_uniform_mesh_2d(&rt, &mesh, config, mp->nx, mp->xMin, mp->xMax,
                                       mp->ny, mp->yMin, mp->yMax, periods);
    else
        runtime_create_uniform_mesh_3d(&rt, &mesh, config, mp->nx, mp->xMin, mp->xMax,
                                       mp->ny, mp->yMin, mp->yMax,
                                       mp->nz, mp->zMin, mp->zMax, periods);

    for (int f = 0; f < 6; ++f)
        runtime_set_bc(&rt, &mesh, f, input.bc[f]);

    SolutionState state;
    solution_state_init(&state, mesh_total_cells(&mesh), config);

    EOSData eos;
    if (strcmp(input.eosParams.type, "StiffenedGas") == 0)
        eos = eos_create_stiffened_gas(input.eosParams.gamma, input.eosParams.pInf,
                                       input.eosParams.R, config->dim);
    else
        eos = eos_create_ideal_gas(input.eosParams.gamma, input.eosParams.R, config->dim);

    apply_initial_conditions(&mesh, &state, &eos, config,
                             &input.defaultState, input.patches, input.nPatches);

    ExplicitSolverWork explSolver;
    SemiImplicitSolverWork siSolver;
    PressureSolverData pressureSolver;

    StepContext stepCtx;
    memset(&stepCtx, 0, sizeof(stepCtx));
    stepCtx.config = config;
    stepCtx.mesh = &mesh;
    stepCtx.state = &state;

    if (config->semiImplicit) {
        if (input.pressureSolverType == PS_PETSC) {
#ifdef SIFV_HAS_PETSC
            if (size > 1) {
                pressure_solver_init_petsc_mpi(&pressureSolver,
                                               (void*)&rt.mpiCtx->cartComm,
                                               rt.mpiCtx->localExtent,
                                               periods,
                                               rt.mpiCtx->dims);
            } else {
                pressure_solver_init_petsc(&pressureSolver);
            }
#else
            if (rank == 0)
                fprintf(stderr, "PETSc requested but not built; falling back to GaussSeidel\n");
            pressure_solver_init(&pressureSolver, PS_GAUSS_SEIDEL);
#endif
        } else {
            pressure_solver_init(&pressureSolver, input.pressureSolverType);
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

    if (input.smoothingParams.iterations > 0) {
        runtime_smooth_fields_config(&rt, &state, &mesh,
                                     input.smoothingParams.iterations, config);
    }

    /* Run time loop */
    TimeLoopParams tlp = time_loop_params_defaults();
    tlp.endTime = input.timeLoopParams.endTime;
    tlp.outputInterval = 1e30;  /* No VTK output */
    tlp.printInterval = 100000;  /* Suppress step printing */
    tlp.startTime = config->time;

    AcousticDtContext acousticCtx;
    if (config->semiImplicit) {
        acousticCtx.mesh = &mesh;
        acousticCtx.state = &state;
        acousticCtx.eos = &eos;
        acousticCtx.config = config;
        acousticCtx.comm = (rt.mpiCtx) ? rt.mpiCtx->cartComm : MPI_COMM_WORLD;
        tlp.acousticDtFn = acoustic_dt_callback;
        tlp.acousticDtCtx = &acousticCtx;
    }

    run_time_loop(&rt, config, &mesh, &state, NULL,
                  step_callback, &stepCtx, &tlp);

    int nPhases = config->multiPhaseParams.nPhases;
    if (nPhasesOut) *nPhasesOut = nPhases;
    if (dimOut) *dimOut = config->dim;

    /* Gather local cells (using global indices) */
    std::vector<CellData> cells = gatherLocalCells(&mesh, &state, config, &rt);

    /* Cleanup */
    if (config->semiImplicit) {
        semi_implicit_solver_free(&siSolver);
        pressure_solver_free(&pressureSolver);
    } else {
        explicit_solver_free(&explSolver);
    }
    solution_state_free(&state);
    mesh_free(&mesh);

    /* Free MPI context created by runtime_create_uniform_mesh */
    if (rt.mpiCtx) {
        mpi_context_free(rt.mpiCtx);
        free(rt.mpiCtx);
        rt.mpiCtx = NULL;
    }
    if (rt.halo) {
        halo_free(rt.halo);
        free(rt.halo);
        rt.halo = NULL;
    }

    input_data_free(&input);

    return cells;
}

/* ---- MPI gather all cells to rank 0 ---- */

static std::vector<CellData> gatherAllToRank0(const std::vector<CellData>& local) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int localCount = (int)local.size();
    std::vector<int> counts(size);
    MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Compute byte displacements */
    int byteSize = (int)sizeof(CellData);
    std::vector<int> byteCounts(size);
    std::vector<int> byteDispls(size, 0);
    int totalCount = 0;
    for (int r = 0; r < size; ++r) {
        byteCounts[r] = counts[r] * byteSize;
        if (r > 0) byteDispls[r] = byteDispls[r-1] + byteCounts[r-1];
        totalCount += counts[r];
    }

    std::vector<CellData> all;
    if (rank == 0) all.resize(totalCount);

    MPI_Gatherv(local.data(), localCount * byteSize, MPI_BYTE,
                rank == 0 ? all.data() : nullptr,
                byteCounts.data(), byteDispls.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);

    /* Sort by global (k, j, i) so multi-rank ordering matches single-rank */
    if (rank == 0) {
        std::sort(all.begin(), all.end(), [](const CellData& a, const CellData& b) {
            if (a.k != b.k) return a.k < b.k;
            if (a.j != b.j) return a.j < b.j;
            return a.i < b.i;
        });
    }

    return all;
}

/* ---- Test macro for each case ---- */

static bool isGenerateMode() {
    const char* env = getenv("GENERATE_REFERENCES");
    return env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0);
}

#define REGRESSION_TEST(TestName, caseFile, refFile, numSteps)                       \
    REGRESSION_TEST_TOL(TestName, caseFile, refFile, numSteps, 1e-8)

#define REGRESSION_TEST_TOL(TestName, caseFile, refFile, numSteps, tol)              \
TEST(Regression, TestName) {                                                         \
    int rank;                                                                        \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                            \
                                                                                     \
    std::string casePath = std::string(TEST_CASES_DIR) + "/" + caseFile;              \
    std::string refPath = std::string(TEST_REFS_DIR) + "/" + refFile;                 \
    std::string srcRefPath = std::string(SOURCE_REFS_DIR) + "/" + refFile;            \
                                                                                     \
    int nPhases = 0, dim = 0;                                                        \
    std::vector<CellData> local = runCase(casePath.c_str(), numSteps,                \
                                          &nPhases, &dim);                           \
    std::vector<CellData> all = gatherAllToRank0(local);                             \
                                                                                     \
    if (rank == 0) {                                                                 \
        if (isGenerateMode()) {                                                      \
            writeReference(srcRefPath.c_str(), #TestName, numSteps, dim,             \
                           nPhases, all);                                            \
            printf("  Generated reference: %s (%zu cells)\n",                        \
                   srcRefPath.c_str(), all.size());                                  \
        } else {                                                                     \
            int refPhases = 0;                                                       \
            std::vector<CellData> ref = readReference(refPath.c_str(),               \
                                                       &refPhases);                  \
            ASSERT_FALSE(ref.empty())                                                \
                << "Reference file missing or empty: " << refPath                    \
                << "\nRun with GENERATE_REFERENCES=1 to create it.";                 \
            bool ok = compareCells(all, ref, std::max(nPhases, refPhases),           \
                                   tol, 20);                                         \
            EXPECT_TRUE(ok) << "Pointwise comparison failed for " << #TestName;      \
        }                                                                            \
    }                                                                                \
                                                                                     \
    /* Broadcast pass/fail so all ranks agree */                                     \
    int passed = 1;                                                                  \
    if (rank == 0 && !isGenerateMode()) {                                            \
        /* Re-check without printing */                                              \
        int refPhases = 0;                                                           \
        std::vector<CellData> ref = readReference(refPath.c_str(), &refPhases);      \
        passed = (!ref.empty() && compareCells(all, ref,                             \
                  std::max(nPhases, refPhases), tol, 0)) ? 1 : 0;                   \
    }                                                                                \
    MPI_Bcast(&passed, 1, MPI_INT, 0, MPI_COMM_WORLD);                              \
}

/* ---- Register regression test cases ---- */

REGRESSION_TEST(SodShocktube1D,       "1D_sod_shocktube_50.jsonc",       "1D_sod_shocktube_50.dat",       50)
REGRESSION_TEST(AdvectionSI1D,        "1D_advection_SI_50.jsonc",        "1D_advection_SI_50.dat",        50)
REGRESSION_TEST_TOL(LiquidGasShocktube1D, "1D_liquid_gas_shocktube_50.jsonc","1D_liquid_gas_shocktube_50.dat",50, 1e-6)
REGRESSION_TEST(IsentropicVortex2D,   "2D_isentropic_vortex_50.jsonc",   "2D_isentropic_vortex_50.dat",   50)
REGRESSION_TEST(ChannelFlow2D,        "2D_channel_flow_50.jsonc",        "2D_channel_flow_50.dat",        50)
REGRESSION_TEST(TaylorGreenVortex3D,      "3D_taylor_green_vortex_50.jsonc",   "3D_taylor_green_vortex_50.dat",   50)
REGRESSION_TEST_TOL(TaylorGreenVortexSI3D,"3D_taylor_green_vortex_SI_50.jsonc","3D_taylor_green_vortex_SI_50.dat",50, 1e-5)

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "RusanovSolver.hpp"
#include "LFSolver.hpp"
#include "HLLCSolver.hpp"
#include "SemiImplicitSolver.hpp"
#include "ExplicitSolver.hpp"
#include "GaussSeidelPressureSolver.hpp"
#include "JacobiPressureSolver.hpp"
#include "IGR.hpp"
#include "IdealGasEOS.hpp"
#include "SimulationConfig.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"

#include "RKTimeStepping.hpp"

#include <iostream>
#include <memory>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>

using namespace SemiImplicitFV;

// ============================================================
//  Solver toggle — flip to switch between explicit and
//  semi-implicit time stepping.
// ============================================================
static constexpr bool useSemiImplicit = false;

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

int main(int argc, char** argv) {
    Runtime rt(argc, argv);

    // ---- Setup ----
    const int    numCells = 256;
    const double length   = 1.0;
    const double endTime  = length / u0;   // one full domain traversal

    // Simulation config
    SimulationConfig config;
    config.dim = 1;
    config.nGhost = 3;
    config.semiImplicit = useSemiImplicit;
    config.reconOrder = ReconstructionOrder::WENO5;

    if (useSemiImplicit) {
        config.RKOrder = 3;
        config.useIGR = false;
        config.semiImplicitParams.cfl = 0.8;
        config.semiImplicitParams.maxDt = 1e-2;
        config.semiImplicitParams.pressureTol = 1e-9;
        config.semiImplicitParams.maxPressureIters = 200;
        config.igrParams.alphaCoeff = 10.0;
        config.igrParams.IGRIters = 5;
    } else {
        config.RKOrder = 3;
        config.useIGR = false;
        config.explicitParams.cfl = 0.8;
        config.explicitParams.maxDt = 1e-2;
    }
    config.validate();

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);

    PrimitiveState ref;
    ref.rho = rho0;
    ref.u   = {u0, 0.0, 0.0};
    ref.p   = p0;
    double c0   = eos->soundSpeed(ref);
    double mach = u0 / c0;

    rt.print("=== 1D Advection ===\n");
    rt.print("  Solver:   ", useSemiImplicit ? "Semi-implicit" : "Explicit", "\n");
    rt.print("  Cells:    ", numCells, "\n");
    rt.print("  RK order: ", config.RKOrder, "\n");
    rt.print("  u0:       ", u0, " m/s\n");
    rt.print("  c0:       ", c0, " m/s\n");
    rt.print("  Mach:     ", mach, "\n");
    rt.print("  End time: ", endTime, " s  (one domain traversal)\n\n");

    // ---- Mesh setup ----
    RectilinearMesh mesh = rt.createUniformMesh(config, numCells, 0.0, length, {1, 0, 0});
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow,  BoundaryCondition::Periodic);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XHigh, BoundaryCondition::Periodic);

    // Allocate solution state
    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto riemannSolver = std::make_shared<LFSolver>(eos, config);
    auto igrSolver     = config.useIGR
                         ? std::make_shared<IGRSolver>(config.igrParams)
                         : nullptr;

    // ---- Build solver ----
    std::function<double(double)> stepFn;

    std::unique_ptr<ExplicitSolver>     explicitSolver;
    std::unique_ptr<SemiImplicitSolver> semiImplicitSolver;

    if (useSemiImplicit) {
        auto pressureSolver = std::make_shared<GaussSeidelPressureSolver>();
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

    initializeProblem(mesh, state, *eos, xCenter, length);

    // Store initial state for error computation
    std::vector<double> rhoRef  = state.rho;
    std::vector<double> velURef = state.velU;
    std::vector<double> presRef = state.pres;

    VTKSession vtk(rt, "1D_advection", mesh);
    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime = endTime, .outputInterval = endTime / 100.0, .printInterval = 1});

    // ---- Recompute primitives from final conservative state ----
    state.convertConservativeToPrimitiveVariables(mesh, eos);

    // ---- Error norms ----
    struct ErrorNorms { double L1 = 0, L2 = 0, Linf = 0; };
    auto accumulateError = [](ErrorNorms& e, double diff) {
        double a = std::abs(diff);
        e.L1   += a;
        e.L2   += diff * diff;
        e.Linf  = std::max(e.Linf, a);
    };
    auto normalizeError = [](ErrorNorms& e, int n) {
        e.L1 /= n;
        e.L2  = std::sqrt(e.L2 / n);
    };

    ErrorNorms errRho, errU, errP;
    int nCells = mesh.nx();

    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        accumulateError(errRho, state.rho[idx]  - rhoRef[idx]);
        accumulateError(errU,   state.velU[idx] - velURef[idx]);
        accumulateError(errP,   state.pres[idx] - presRef[idx]);
    }

    normalizeError(errRho, nCells);
    normalizeError(errU,   nCells);
    normalizeError(errP,   nCells);

    rt.print("\n=== Error Norms (final vs. initial, N = ", numCells, ") ===\n");
    {
        std::ostringstream hdr;
        hdr << "  " << std::left << std::setw(12) << "Variable"
            << std::setw(14) << "L1"
            << std::setw(14) << "L2"
            << std::setw(14) << "Linf" << "\n";
        hdr << "  " << std::string(52, '-') << "\n";
        rt.print(hdr.str());
    }
    auto printRow = [&](const char* name, const ErrorNorms& e) {
        std::ostringstream oss;
        oss << "  " << std::left << std::setw(12) << name
            << std::scientific << std::setprecision(6)
            << std::setw(14) << e.L1
            << std::setw(14) << e.L2
            << std::setw(14) << e.Linf
            << "\n";
        rt.print(oss.str());
    };
    printRow("Density",    errRho);
    printRow("Velocity-u", errU);
    printRow("Pressure",   errP);

    return 0;
}

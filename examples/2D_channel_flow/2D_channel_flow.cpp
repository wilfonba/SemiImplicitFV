#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "HLLCSolver.hpp"
#include "ExplicitSolver.hpp"
#include "SemiImplicitSolver.hpp"
#include "GaussSeidelPressureSolver.hpp"
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
#include <sstream>

using namespace SemiImplicitFV;

// ============================================================
//  Problem parameters
// ============================================================
static constexpr double gamma_gas = 1.4;
static constexpr double Rgas      = 1.0;
static constexpr double L         = 1.0;    // channel length (x)
static constexpr double H         = 1.0;    // channel height (y)
static constexpr double G         = 0.1;    // body force acceleration in x
static constexpr double mu        = 0.1;    // dynamic viscosity
static constexpr double rho0      = 1.0;    // initial density
static constexpr double p0        = 100.0;  // background pressure (keep Mach low)
static constexpr double endTime   = 20.0;   // ~2 viscous diffusion times (t_visc = rho*H^2/mu = 10)

// Analytical steady-state Poiseuille profile:
//   u(y) = (rho * G) / (2 * mu) * y * (H - y)
//   u_max = rho * G * H^2 / (8 * mu) = 0.125
static double analyticalU(double y) {
    return (rho0 * G) / (2.0 * mu) * y * (H - y);
}

// ============================================================
//  Error norms
// ============================================================
struct ErrorNorms { double L1 = 0, L2 = 0, Linf = 0; };

void accumulateError(ErrorNorms& e, double diff) {
    double a = std::abs(diff);
    e.L1   += a;
    e.L2   += diff * diff;
    e.Linf  = std::max(e.Linf, a);
}

void normalizeError(ErrorNorms& e, int nCells) {
    e.L1 /= nCells;
    e.L2  = std::sqrt(e.L2 / nCells);
}

void printRow(Runtime& rt, const char* name, const ErrorNorms& e) {
    std::ostringstream oss;
    oss << "  " << std::left << std::setw(12) << name
        << std::scientific << std::setprecision(6)
        << std::setw(14) << e.L1
        << std::setw(14) << e.L2
        << std::setw(14) << e.Linf
        << "\n";
    rt.print(oss.str());
}

// ============================================================
//  Main
// ============================================================
int main(int argc, char** argv)
{
    Runtime rt(argc, argv);

    // ---- Parse command-line arguments ----
    int Ny = 32;
    bool useSemiImplicit = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--semi-implicit") {
            useSemiImplicit = true;
        } else {
            Ny = std::stoi(arg);
        }
    }

    int Nx = 8;  // minimal cells in periodic x-direction for WENO5

    // ---- Configuration ----
    SimulationConfig config;
    config.dim          = 2;
    config.nGhost       = 3;
    config.useIGR       = false;
    config.semiImplicit = useSemiImplicit;
    config.RKOrder      = 3;
    config.reconOrder   = ReconstructionOrder::WENO5;

    // Body force: constant acceleration G in x-direction
    config.bodyForceParams.a[0] = G;

    // Viscosity
    config.viscousParams.mu = mu;

    // Viscous CFL limit: dt <= CFL_visc * dy^2 / (2 * nu)  where nu = mu/rho
    // Viscous fluxes are explicit in both solvers, so this constrains maxDt.
    double dy = H / Ny;
    double nu = mu / rho0;
    double viscousDtLimit = 0.4 * dy * dy / (2.0 * nu);

    if (useSemiImplicit) {
        config.semiImplicitParams.cfl              = 0.8;
        config.semiImplicitParams.maxDt            = viscousDtLimit;
        config.semiImplicitParams.pressureTol      = 1e-8;
        config.semiImplicitParams.maxPressureIters = 500;
    } else {
        config.explicitParams.cfl   = 0.8;
        config.explicitParams.maxDt = viscousDtLimit;
    }

    config.validate();

    // ---- Print setup ----
    double uMax = rho0 * G * H * H / (8.0 * mu);
    double cSound = std::sqrt(gamma_gas * p0 / rho0);
    double Mach = uMax / cSound;

    rt.print("=== 2D Channel Flow (Poiseuille) ===\n");
    rt.print("  Solver:    ", useSemiImplicit ? "Semi-implicit" : "Explicit", "\n");
    rt.print("  Grid:      ", Nx, " x ", Ny, "\n");
    rt.print("  mu:        ", mu, "\n");
    rt.print("  G:         ", G, "\n");
    rt.print("  u_max:     ", uMax, " (analytical)\n");
    rt.print("  Mach:      ", Mach, "\n");
    rt.print("  RK order:  ", config.RKOrder, "\n");
    rt.print("  Recon:     WENO5\n");
    rt.print("  End time:  ", endTime, "\n\n");

    // ---- Mesh: periodic in x, walls in y ----
    RectilinearMesh mesh = rt.createUniformMesh(
        config, Nx, 0.0, L, Ny, 0.0, H, {1, 0, 0});
    rt.setBoundaryCondition(mesh, RectilinearMesh::YLow,  BoundaryCondition::NoSlipWall);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YHigh, BoundaryCondition::NoSlipWall);

    // ---- Allocate state ----
    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos           = std::make_shared<IdealGasEOS>(gamma_gas, Rgas, config);
    auto riemannSolver = std::make_shared<HLLCSolver>(eos, config);

    // ---- Initialize: uniform quiescent state ----
    PrimitiveState W0;
    W0.rho   = rho0;
    W0.u     = {0.0, 0.0, 0.0};
    W0.p     = p0;
    W0.sigma = 0.0;
    W0.T     = eos->temperature(W0);
    ConservativeState U0 = eos->toConservative(W0);

    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            std::size_t idx = mesh.index(i, j, 0);
            state.setPrimitiveState(idx, W0);
            state.setConservativeState(idx, U0);
        }
    }

    // ---- Build solver ----
    std::function<double(double)> stepFn;

    std::unique_ptr<ExplicitSolver>     explicitSolver;
    std::unique_ptr<SemiImplicitSolver> semiImplicitSolver;

    if (useSemiImplicit) {
        auto pressureSolver = std::make_shared<GaussSeidelPressureSolver>();
        semiImplicitSolver = std::make_unique<SemiImplicitSolver>(
            mesh, riemannSolver, pressureSolver, eos, nullptr, config);
        rt.attachSolver(*semiImplicitSolver, mesh);
        stepFn = [&](double targetDt) {
            return semiImplicitSolver->step(config, mesh, state, targetDt);
        };
    } else {
        explicitSolver = std::make_unique<ExplicitSolver>(
            mesh, riemannSolver, eos, nullptr, config);
        rt.attachSolver(*explicitSolver, mesh);
        stepFn = [&](double targetDt) {
            return explicitSolver->step(config, mesh, state, targetDt);
        };
    }

    // ---- Run ----
    VTKSession vtk(rt, "2D_channel_flow", mesh);

    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime        = endTime,
                 .outputInterval = endTime / 10.0,
                 .printInterval  = 100});

    // ---- Recompute primitives from final conservative state ----
    state.convertConservativeToPrimitiveVariables(mesh, eos);

    // ---- Error norms: compare u-velocity against analytical profile ----
    // Sample a single x-column (mid-domain)
    int iMid = Nx / 2;
    ErrorNorms errU, errV;

    for (int j = 0; j < mesh.ny(); ++j) {
        std::size_t idx = mesh.index(iMid, j, 0);
        double y = mesh.cellCentroidY(j);
        double uExact = analyticalU(y);

        accumulateError(errU, state.velU[idx] - uExact);
        accumulateError(errV, state.velV[idx] - 0.0);
    }

    normalizeError(errU, mesh.ny());
    normalizeError(errV, mesh.ny());

    rt.print("\n=== Error Norms (Ny = ", Ny, ") ===\n");
    {
        std::ostringstream hdr;
        hdr << "  " << std::left << std::setw(12) << "Variable"
            << std::setw(14) << "L1"
            << std::setw(14) << "L2"
            << std::setw(14) << "Linf" << "\n";
        hdr << "  " << std::string(52, '-') << "\n";
        rt.print(hdr.str());
    }
    printRow(rt, "Velocity-u", errU);
    printRow(rt, "Velocity-v", errV);

    // ---- Profile comparison table ----
    rt.print("\n=== Velocity Profile (x = ", mesh.cellCentroidX(iMid), ") ===\n");
    {
        std::ostringstream hdr;
        hdr << "  " << std::left
            << std::setw(12) << "y"
            << std::setw(16) << "u_numerical"
            << std::setw(16) << "u_analytical"
            << std::setw(16) << "error"
            << "\n";
        hdr << "  " << std::string(58, '-') << "\n";
        rt.print(hdr.str());
    }

    // Print every few cells for a readable table
    int stride = std::max(1, mesh.ny() / 16);
    for (int j = 0; j < mesh.ny(); j += stride) {
        std::size_t idx = mesh.index(iMid, j, 0);
        double y = mesh.cellCentroidY(j);
        double uExact = analyticalU(y);
        double uNum   = state.velU[idx];

        std::ostringstream row;
        row << "  " << std::fixed << std::setprecision(4)
            << std::setw(12) << y
            << std::scientific << std::setprecision(6)
            << std::setw(16) << uNum
            << std::setw(16) << uExact
            << std::setw(16) << (uNum - uExact)
            << "\n";
        rt.print(row.str());
    }

    return 0;
}

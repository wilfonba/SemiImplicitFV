#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "HLLCSolver.hpp"
#include "ExplicitSolver.hpp"
#include "SemiImplicitSolver.hpp"
#include "GaussSeidelPressureSolver.hpp"
#include "IdealGasEOS.hpp"
#include "MixtureEOS.hpp"
#include "SimulationConfig.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"
#include "RKTimeStepping.hpp"

#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace SemiImplicitFV;

// ============================================================
//  Problem parameters (air-water at ~20 °C)
// ============================================================
static constexpr double gammaWater = 4.4;
static constexpr double pInfWater  = 6.0e8;     // Pa
static constexpr double gammaAir   = 1.4;
static constexpr double pInfAir    = 0.0;
static constexpr double rhoWater   = 1000.0;     // kg/m^3
static constexpr double rhoAir     = 1.225;      // kg/m^3
static constexpr double sigma      = 0.0728;     // N/m  (surface tension)
static constexpr double R0         = 1.0e-3;     // droplet radius (m)
static constexpr double pAtm       = 1.0e5;      // Pa   (atmospheric, outside droplet)
static constexpr double domainSize = 6.0e-3;     // [0, 6 mm]^2

// Laplace pressure jump:  dp = sigma / R  (2-D cylinder)
static constexpr double laplaceDp = sigma / R0;  // 72.8 Pa

// ============================================================
//  Initialize circular water droplet in quiescent air
// ============================================================
static void initializeDroplet(const RectilinearMesh& mesh,
                              SolutionState& state,
                              const MultiPhaseParams& mp)
{
    double xc = 0.5 * domainSize;
    double yc = 0.5 * domainSize;
    double alphaMin = mp.alphaMin;

    // Diffuse interface width: ~3 cells
    double epsilon = 3.0 * mesh.dx(0);

    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            std::size_t idx = mesh.index(i, j, 0);
            double x = mesh.cellCentroidX(i);
            double y = mesh.cellCentroidY(j);

            double dx = x - xc;
            double dy = y - yc;
            double r = std::sqrt(dx * dx + dy * dy);

            // Smooth alpha profile: water inside, air outside
            double alphaW = 0.5 * (1.0 - std::tanh((r - R0) / epsilon));
            alphaW = std::max(alphaMin, std::min(1.0 - alphaMin, alphaW));
            double alphaA = 1.0 - alphaW;

            // Pressure with Laplace jump (smoothed with same profile)
            double p = pAtm + laplaceDp * 0.5 * (1.0 - std::tanh((r - R0) / epsilon));

            // Partial densities
            state.alphaRho[0][idx] = alphaW * rhoWater;
            state.alphaRho[1][idx] = alphaA * rhoAir;
            state.alpha[0][idx]    = alphaW;

            state.rho[idx]  = state.alphaRho[0][idx] + state.alphaRho[1][idx];
            state.rhoU[idx] = 0.0;
            state.rhoV[idx] = 0.0;
            state.velU[idx] = 0.0;
            state.velV[idx] = 0.0;
            state.pres[idx] = pAtm; //p;

            // Total energy from mixture EOS (accounts for stiffened-gas pInf)
            std::vector<double> alphas = {alphaW};
            double ke = 0.0;
            state.rhoE[idx] = MixtureEOS::mixtureTotalEnergy(
                state.rho[idx], p, alphas, ke, mp);
        }
    }
}

// ============================================================
//  Main
// ============================================================
int main(int argc, char** argv)
{
    Runtime rt(argc, argv);

    // ---- Parse command-line arguments ----
    int N = 100;
    bool useSemiImplicit = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--semi-implicit") {
            useSemiImplicit = true;
        } else {
            N = std::stoi(arg);
        }
    }

    // ---- Configuration ----
    SimulationConfig config;
    config.dim          = 2;
    config.nGhost       = 3;
    config.RKOrder      = 3;
    config.reconOrder   = ReconstructionOrder::WENO5;
    config.semiImplicit = useSemiImplicit;

    // Two-phase stiffened-gas setup
    //   Phase 0 = water (gamma=4.4, pInf=6e8)
    //   Phase 1 = air   (gamma=1.4, pInf=0)
    config.multiPhaseParams.nPhases = 2;
    config.multiPhaseParams.phases  = {{gammaWater, pInfWater}, {gammaAir, pInfAir}};
    config.multiPhaseParams.alphaMin = 1e-8;

    // Surface tension
    config.surfaceTensionParams.sigma = sigma;

    // End time: several capillary relaxation times
    double tCap   = std::sqrt(rhoWater * R0 * R0 * R0 / sigma);
    double endTime = 0.2 * tCap;

    // Capillary dt limit is now enforced automatically by the solvers
    // via computeCapillaryDt.
    if (useSemiImplicit) {
        config.semiImplicitParams.cfl              = 0.5;
        config.semiImplicitParams.pressureTol      = 1e-3;
        config.semiImplicitParams.maxPressureIters = 200;
    } else {
        config.explicitParams.cfl   = 0.5;
    }

    config.validate();

    // ---- Mesh ----
    RectilinearMesh mesh = rt.createUniformMesh(
        config, N, 0.0, domainSize, N, 0.0, domainSize);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow,  BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XHigh, BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YLow,  BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YHigh, BoundaryCondition::Outflow);

    // ---- Print setup ----
    rt.print("=== 2D Laplace Pressure Jump (Water-Air) ===\n");
    rt.print("  Solver:       ", useSemiImplicit ? "Semi-implicit" : "Explicit", "\n");
    rt.print("  Grid:         ", N, " x ", N, "\n");
    rt.print("  dx:           ", domainSize / N, " m\n");
    rt.print("  Droplet R:    ", R0, " m\n");
    rt.print("  sigma:        ", sigma, " N/m\n");
    rt.print("  dp_exact:     ", laplaceDp, " Pa  (sigma/R)\n");
    rt.print("  p_atm:        ", pAtm, " Pa\n");
    rt.print("  t_cap:        ", tCap, " s\n");
    rt.print("  End time:     ", endTime, " s\n");
    rt.print("  RK order:     3\n");
    rt.print("  Recon:        WENO5\n\n");

    // ---- Allocate & initialize ----
    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    // Fallback EOS (not primary for multi-phase)
    auto eos           = std::make_shared<IdealGasEOS>(gammaAir, 287.0, config);
    auto riemannSolver = std::make_shared<HLLCSolver>(eos, config);

    initializeDroplet(mesh, state, config.multiPhaseParams);

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
    VTKSession vtk(rt, "2D_laplace_pressure_jump", mesh, config);

    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime        = endTime,
                 .outputInterval = endTime / 100.0,
                 .printInterval  = 100});

    // ---- Diagnostics: Laplace pressure jump ----
    // Pressure at center (inside droplet) and at corner (far outside)
    int iCenter = N / 2;
    int jCenter = N / 2;
    std::size_t idxCenter = mesh.index(iCenter, jCenter, 0);
    std::size_t idxCorner = mesh.index(0, 0, 0);

    double pCenter    = state.pres[idxCenter];
    double pCorner    = state.pres[idxCorner];
    double dpMeasured = pCenter - pCorner;

    // Max and L2 velocity magnitude (spurious currents)
    double maxVel = 0.0;
    double l2Vel  = 0.0;
    int nCells    = 0;
    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            std::size_t idx = mesh.index(i, j, 0);
            double vmag = std::sqrt(state.velU[idx] * state.velU[idx]
                                  + state.velV[idx] * state.velV[idx]);
            maxVel = std::max(maxVel, vmag);
            l2Vel += vmag * vmag;
            ++nCells;
        }
    }
    l2Vel = std::sqrt(l2Vel / nCells);

    double dpError    = std::abs(dpMeasured - laplaceDp);
    double dpRelError = dpError / laplaceDp;

    rt.print("\n=== Laplace Pressure Jump Results ===\n");
    rt.print("  Theoretical dp (sigma/R): ", laplaceDp, " Pa\n");
    rt.print("  Measured dp:              ", dpMeasured, " Pa\n");
    rt.print("  Absolute error:           ", dpError, " Pa\n");
    rt.print("  Relative error:           ", dpRelError * 100.0, " %\n");
    rt.print("  Max |u| (spurious curr.): ", maxVel, " m/s\n");
    rt.print("  L2  |u| (spurious curr.): ", l2Vel,  " m/s\n");

    // ---- Pressure profile through center (y = yc) ----
    rt.print("\n=== Pressure Profile (y = center) ===\n");
    {
        std::ostringstream hdr;
        hdr << "  " << std::left
            << std::setw(14) << "x (m)"
            << std::setw(14) << "r/R"
            << std::setw(16) << "p (Pa)"
            << std::setw(16) << "p_exact (Pa)"
            << std::setw(16) << "error (Pa)"
            << "\n";
        hdr << "  " << std::string(74, '-') << "\n";
        rt.print(hdr.str());
    }

    double xc = 0.5 * domainSize;
    int stride = std::max(1, N / 20);
    for (int i = 0; i < mesh.nx(); i += stride) {
        std::size_t idx = mesh.index(i, jCenter, 0);
        double x = mesh.cellCentroidX(i);
        double r = std::abs(x - xc);
        double pExact = pAtm + laplaceDp * ((r < R0) ? 1.0 : 0.0);
        double pNum   = state.pres[idx];

        std::ostringstream row;
        row << "  " << std::scientific << std::setprecision(4)
            << std::setw(14) << x
            << std::fixed << std::setprecision(4)
            << std::setw(14) << (r / R0)
            << std::scientific << std::setprecision(6)
            << std::setw(16) << pNum
            << std::setw(16) << pExact
            << std::setw(16) << (pNum - pExact)
            << "\n";
        rt.print(row.str());
    }

    return 0;
}

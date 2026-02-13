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
//  Problem parameters — Hysing et al. (2009), Test Case 2
//  Re = 35, Eo = 125
// ============================================================
static constexpr double Lx       = 1.0;        // domain width  (m)
static constexpr double Ly       = 2.0;        // domain height (m)
static constexpr double R0       = 0.25;       // bubble radius (m)
static constexpr double xc0      = 0.5;        // bubble center x
static constexpr double yc0      = 0.5;        // bubble center y

static constexpr double rhoHeavy = 1000.0;     // surrounding fluid density (kg/m^3)
static constexpr double rhoLight = 1.0;        // bubble fluid density (kg/m^3)
static constexpr double grav     = 0.98;       // gravitational acceleration (m/s^2)
static constexpr double sigmaST  = 1.96;       // surface tension coefficient (N/m)

// Dynamic viscosity: mu_heavy = 10 Pa*s, mu_light = 0.1 Pa*s
// NOTE: The code uses a single global mu, so we set it from the heavy phase.
// The light phase would need mu = 0.1 Pa*s for an exact match,
// but that requires per-phase viscosity which is not yet implemented.
static constexpr double muVisc   = 10.0;       // Pa*s (heavy phase)

// EOS: both phases ideal gas with same gamma (artificial benchmark fluids)
static constexpr double gammaGas = 1.4;
static constexpr double pInfGas  = 0.0;
static constexpr double gammaW = 6.12;
static constexpr double pInfW  = 3.43e8;

// Background pressure: elevated to keep Mach number low (~incompressible)
static constexpr double pRef     = 1.0e6;      // Pa

// Non-dimensional numbers (for reference):
//   Re = rho_heavy * sqrt(g * 2R) * 2R / mu = 1000 * sqrt(0.98*0.5) * 0.5 / 10 = 35
//   Eo = rho_heavy * g * (2R)^2 / sigma      = 1000 * 0.98 * 0.25 / 1.96       = 125

static constexpr double endTime  = 3.0;        // s

// ============================================================
//  Initialize bubble in quiescent heavy fluid
// ============================================================
static void initializeBubble(const RectilinearMesh& mesh,
                             SolutionState& state,
                             const MultiPhaseParams& mp)
{
    double alphaMin = mp.alphaMin;

    // Diffuse interface width: ~3 cells
    double epsilon = 3.0 * mesh.dx(0);

    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            std::size_t idx = mesh.index(i, j, 0);
            double x = mesh.cellCentroidX(i);
            double y = mesh.cellCentroidY(j);

            double dx = x - xc0;
            double dy = y - yc0;
            double r  = std::sqrt(dx * dx + dy * dy);

            // Smooth alpha profile: heavy fluid (phase 0) outside, light (phase 1) inside
            // alpha[0] = volume fraction of phase 0 (heavy, surrounding)
            double alphaHeavy = 0.5 * (1.0 + std::tanh((r - R0) / epsilon));
            alphaHeavy = std::max(alphaMin, std::min(1.0 - alphaMin, alphaHeavy));
            double alphaLight = 1.0 - alphaHeavy;

            // Partial densities
            state.alphaRho[0][idx] = alphaHeavy * rhoHeavy;
            state.alphaRho[1][idx] = alphaLight * rhoLight;
            state.alpha[0][idx]    = alphaHeavy;

            double rho = state.alphaRho[0][idx] + state.alphaRho[1][idx];
            state.rho[idx] = rho;

            // Velocity: zero everywhere
            state.rhoU[idx] = 0.0;
            state.rhoV[idx] = 0.0;
            state.velU[idx] = 0.0;
            state.velV[idx] = 0.0;

            // Hydrostatic pressure: p(y) = pRef + rhoHeavy * g * (Ly - y)
            double p = pRef + rhoHeavy * grav * (Ly - y);
            state.pres[idx] = p;

            // Total energy from mixture EOS
            std::vector<double> alphas = {alphaHeavy};
            double ke = 0.0;
            state.rhoE[idx] = MixtureEOS::mixtureTotalEnergy(rho, p, alphas, ke, mp);
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
    int Ny = 80;
    bool useSemiImplicit = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--semi-implicit") {
            useSemiImplicit = true;
        } else {
            Ny = std::stoi(arg);
        }
    }

    int Nx = Ny / 2;  // uniform grid: dx = dy

    // ---- Configuration ----
    SimulationConfig config;
    config.dim          = 2;
    config.nGhost       = 3;
    config.RKOrder      = 3;
    config.reconOrder   = ReconstructionOrder::WENO5;
    config.semiImplicit = useSemiImplicit;

    // Two-phase ideal gas setup
    //   Phase 0 = heavy fluid (surrounding)
    //   Phase 1 = light fluid (bubble)
    config.multiPhaseParams.nPhases  = 2;
    config.multiPhaseParams.phases   = {{gammaW, pInfW}, {gammaGas, pInfGas}};
    config.multiPhaseParams.alphaMin = 1e-8;

    // Body force: gravity in -y direction
    config.bodyForceParams.a[1] = -grav;

    // Viscosity
    config.viscousParams.mu = muVisc;

    // Surface tension
    config.surfaceTensionParams.sigma = sigmaST;

    // ---- Solver parameters ----
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
        config, Nx, 0.0, Lx, Ny, 0.0, Ly);

    // BCs: no-slip top/bottom, slip left/right
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow,  BoundaryCondition::SlipWall);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XHigh, BoundaryCondition::SlipWall);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YLow,  BoundaryCondition::NoSlipWall);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YHigh, BoundaryCondition::NoSlipWall);

    // ---- Print setup ----
    rt.print("=== 2D Rising Bubble (Hysing et al. 2009, Case 2) ===\n");
    rt.print("  Solver:       ", useSemiImplicit ? "Semi-implicit" : "Explicit", "\n");
    rt.print("  Grid:         ", Nx, " x ", Ny, "\n");
    rt.print("  dx = dy:      ", Ly / Ny, " m\n");
    rt.print("  Bubble R:     ", R0, " m, center (", xc0, ", ", yc0, ")\n");
    rt.print("  rho_heavy:    ", rhoHeavy, " kg/m^3\n");
    rt.print("  rho_light:    ", rhoLight, " kg/m^3\n");
    rt.print("  mu:           ", muVisc, " Pa*s\n");
    rt.print("  g:            ", grav, " m/s^2\n");
    rt.print("  sigma:        ", sigmaST, " N/m\n");
    rt.print("  Re:           35\n");
    rt.print("  Eo:           125\n");
    rt.print("  p_ref:        ", pRef, " Pa\n");
    rt.print("  End time:     ", endTime, " s\n");
    rt.print("  RK order:     3\n");
    rt.print("  Recon:        WENO5\n\n");

    // ---- Allocate & initialize ----
    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos           = std::make_shared<IdealGasEOS>(gammaGas, 287.0, config);
    auto riemannSolver = std::make_shared<HLLCSolver>(eos, config);

    initializeBubble(mesh, state, config.multiPhaseParams);

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
    VTKSession vtk(rt, "2D_rising_bubble", mesh);

    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime        = endTime,
                 .outputInterval = 0.03,
                 .printInterval  = 50});

    // ---- Diagnostics: bubble center of mass and rise velocity ----
    // Phase 1 (light/bubble) volume fraction = 1 - alpha[0]
    double sumAlphaV  = 0.0;
    double sumAlphaYV = 0.0;
    double sumAlphaVyV = 0.0;
    double dV = mesh.dx(0) * mesh.dy(0);

    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            std::size_t idx = mesh.index(i, j, 0);
            double y = mesh.cellCentroidY(j);
            double alphaBubble = 1.0 - state.alpha[0][idx];
            double vy = state.velV[idx];

            sumAlphaV   += alphaBubble * dV;
            sumAlphaYV  += alphaBubble * y * dV;
            sumAlphaVyV += alphaBubble * vy * dV;
        }
    }

    double yCenter = (sumAlphaV > 0.0) ? sumAlphaYV / sumAlphaV : 0.0;
    double vRise   = (sumAlphaV > 0.0) ? sumAlphaVyV / sumAlphaV : 0.0;

    rt.print("\n=== Rising Bubble Diagnostics (t = ", endTime, " s) ===\n");
    rt.print("  Bubble center of mass y_c: ", yCenter, " m\n");
    rt.print("  Bubble rise velocity  v_c: ", vRise, " m/s\n");
    rt.print("\n  Hysing et al. benchmark references (t = 3 s):\n");
    rt.print("    y_c ~ 1.138 m\n");
    rt.print("    v_c_max ~ 0.251 m/s\n");

    return 0;
}

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "HLLCSolver.hpp"
#include "ExplicitSolver.hpp"
#include "SemiImplicitSolver.hpp"
#include "GaussSeidelPressureSolver.hpp"
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
static constexpr bool useSemiImplicit = true;

// ============================================================
//  Problem parameters
// ============================================================
static constexpr double gammaGas  = 1.4;
static constexpr double Rgas   = 1.0;         // non-dimensional EOS
static constexpr double rhoInf = 1.0;         // background density
static constexpr double uInf   = 1.0;         // background x-velocity
static constexpr double vInf   = 1.0;         // background y-velocity (diagonal advection)
static constexpr double Mref   = 0.1;         // reference Mach number
// p_inf chosen so that c_inf = |V_inf| / Mref :
//   c^2 = gammaGas * p / rho  =>  p = rho * (u^2+v^2) / (gammaGas * M^2)
static constexpr double pInf   = 2.0 / (gammaGas * Mref * Mref);  // ~ 142.857
static constexpr double beta   = 5.0;         // vortex strength
static constexpr double L      = 10.0;        // domain side length
static constexpr double xc0    = 5.0;         // initial vortex centre
static constexpr double yc0    = 5.0;
static constexpr int    N      = 128;         // cells per direction
// End time = L so the vortex traverses one full period in each
// direction and returns to its initial position.
static constexpr double endTime = L;

// ============================================================
//  Isentropic vortex perturbation  (Shu 1998)
//
//  Given background state (rhoInf, uInf, vInf, pInf) and a
//  vortex centred at (xc, yc) with strength beta:
//
//    f(r)  = exp((1 - r^2) / 2)
//    du    = -beta / (2 pi) * (y - yc) * f
//    dv    =  beta / (2 pi) * (x - xc) * f
//    dT    = -(gammaGas-1) beta^2 / (8 gammaGas pi^2) * f^2
//
//  Isentropic relations then give rho, p from T.
// ============================================================
void vortexState(double x, double y, double xc, double yc,
                 const IdealGasEOS& eos, PrimitiveState& W)
{
    double TInf = pInf / (rhoInf * Rgas);

    // Shortest periodic distance to vortex centre
    double dx = x - xc;
    double dy = y - yc;
    dx -= L * std::round(dx / L);
    dy -= L * std::round(dy / L);

    double r2 = dx * dx + dy * dy;
    double f  = std::exp(0.5 * (1.0 - r2));       // exp((1-r^2)/2)
    double f2 = std::exp(1.0 - r2);               // f^2

    double du = -beta / (2.0 * M_PI) * dy * f;
    double dv =  beta / (2.0 * M_PI) * dx * f;
    double dT = -(gammaGas - 1.0) * beta * beta
                / (8.0 * gammaGas * M_PI * M_PI) * f2;

    double T   = TInf + dT;
    double rho = rhoInf * std::pow(T / TInf, 1.0 / (gammaGas - 1.0));
    double p   = pInf   * std::pow(T / TInf, gammaGas / (gammaGas - 1.0));

    W.rho   = rho;
    W.u     = {uInf + du, vInf + dv, 0.0};
    W.p     = p;
    W.sigma = 0.0;
    W.T     = eos.temperature(W);
}

void initializeVortex(const RectilinearMesh& mesh,
                      SolutionState& state,
                      const IdealGasEOS& eos)
{
    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            std::size_t idx = mesh.index(i, j, 0);
            double x = mesh.cellCentroidX(i);
            double y = mesh.cellCentroidY(j);

            PrimitiveState W;
            vortexState(x, y, xc0, yc0, eos, W);

            state.setPrimitiveState(idx, W);
            state.setConservativeState(idx, eos.toConservative(W));
        }
    }
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

    // ---- Configuration ----
    SimulationConfig config;
    config.dim       = 2;
    config.nGhost    = 3;
    config.useIGR    = false;
    config.semiImplicit = useSemiImplicit;
    config.reconOrder   = ReconstructionOrder::WENO3;

    if (useSemiImplicit) {
        config.RKOrder = 3;       // SSP-RK3
        config.semiImplicitParams.cfl            = 0.5;
        config.semiImplicitParams.maxDt          = 1.0;
        config.semiImplicitParams.pressureTol    = 1e-8;
        config.semiImplicitParams.maxPressureIters = 500;
    } else {
        config.RKOrder = 3;       // SSP-RK3
        config.explicitParams.cfl   = 0.5;
        config.explicitParams.maxDt = 1.0;
    }

    config.validate();

    // ---- Print setup ----
    double cInf = std::sqrt(gammaGas * pInf / rhoInf);
    double Mach = std::sqrt(uInf * uInf + vInf * vInf) / cInf;

    rt.print("=== 2D Isentropic Vortex Advection ===\n");
    rt.print("  Solver:    ", useSemiImplicit ? "Semi-implicit" : "Explicit", "\n");
    rt.print("  Grid:      ", N, " x ", N, "\n");
    rt.print("  Mach:      ", Mach, "\n");
    rt.print("  Beta:      ", beta, "\n");
    rt.print("  RK order:  ", config.RKOrder, "\n");
    rt.print("  Recon:     WENO5\n");
    rt.print("  End time:  ", endTime, " (one full diagonal traverse)\n\n");

    // ---- Mesh (biperiodic) ----
    RectilinearMesh mesh = rt.createUniformMesh(
        config, N, 0.0, L, N, 0.0, L, {1, 1, 0});
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow,  BoundaryCondition::Periodic);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XHigh, BoundaryCondition::Periodic);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YLow,  BoundaryCondition::Periodic);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YHigh, BoundaryCondition::Periodic);

    // ---- Allocate state ----
    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos           = std::make_shared<IdealGasEOS>(gammaGas, Rgas, config);
    auto riemannSolver = std::make_shared<HLLCSolver>(eos, config);
    auto igrSolver     = config.useIGR
                         ? std::make_shared<IGRSolver>(config.igrParams)
                         : nullptr;

    // ---- Initialize vortex ----
    initializeVortex(mesh, state, *eos);

    // Store initial state for error computation
    std::vector<double> rhoRef  = state.rho;
    std::vector<double> velURef = state.velU;
    std::vector<double> velVRef = state.velV;
    std::vector<double> presRef = state.pres;

    // ---- Build solver ----
    std::function<double(double)> stepFn;

    std::unique_ptr<ExplicitSolver>      explicitSolver;
    std::unique_ptr<SemiImplicitSolver>  semiImplicitSolver;

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

    // ---- Run ----
    VTKSession vtk(rt, "2D_isentropic_vortex", mesh, config);

    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime       = endTime,
                 .outputInterval = endTime / 10.0,
                 .printInterval  = 20});

    // ---- Recompute primitives from final conservative state ----
    state.convertConservativeToPrimitiveVariables(mesh, eos);

    // ---- Error norms ----
    ErrorNorms errRho, errU, errV, errP;
    int nCells = mesh.nx() * mesh.ny();

    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            std::size_t idx = mesh.index(i, j, 0);
            accumulateError(errRho, state.rho[idx]  - rhoRef[idx]);
            accumulateError(errU,   state.velU[idx] - velURef[idx]);
            accumulateError(errV,   state.velV[idx] - velVRef[idx]);
            accumulateError(errP,   state.pres[idx] - presRef[idx]);
        }
    }

    normalizeError(errRho, nCells);
    normalizeError(errU,   nCells);
    normalizeError(errV,   nCells);
    normalizeError(errP,   nCells);

    rt.print("\n=== Error Norms (final vs. initial, N = ", N, ") ===\n");
    {
        std::ostringstream hdr;
        hdr << "  " << std::left << std::setw(12) << "Variable"
            << std::setw(14) << "L1"
            << std::setw(14) << "L2"
            << std::setw(14) << "Linf" << "\n";
        hdr << "  " << std::string(52, '-') << "\n";
        rt.print(hdr.str());
    }
    printRow(rt, "Density",    errRho);
    printRow(rt, "Velocity-u", errU);
    printRow(rt, "Velocity-v", errV);
    printRow(rt, "Pressure",   errP);

    return 0;
}

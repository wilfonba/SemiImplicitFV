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
#include "ImmersedBoundary.hpp"

#include <iostream>
#include <memory>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>

using namespace SemiImplicitFV;

// ============================================================
//  Problem parameters
// ============================================================
static constexpr double gamma_gas = 1.4;
static constexpr double Rgas      = 1.0;

// Domain  [0, Lx] x [0, Ly]
static constexpr double Lx = 4.0;
static constexpr double Ly = 2.0;

// Circle centred at (xc, yc) with radius R
static constexpr double xc = 1.0;
static constexpr double yc = 1.0;
static constexpr double R  = 0.2;

// Free-stream conditions
static constexpr double Mach   = 0.2;
static constexpr double rhoInf = 1.0;
static constexpr double pInf   = rhoInf / (gamma_gas * Mach * Mach);  // c = 1/M
static constexpr double uInf   = 1.0;   // free-stream x-velocity (Mach * c)

static constexpr double endTime = 12.0;  // ~1.5 flow-throughs

// ============================================================
//  Main
// ============================================================
int main(int argc, char** argv)
{
    Runtime rt(argc, argv);

    // ---- Parse command-line arguments ----
    int Nx = 200;
    int Ny = 100;
    bool useSemiImplicit = false;
    bool useIGR = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--explicit")       useSemiImplicit = false;
        else if (arg == "--igr")       useIGR = true;
        else if (arg == "--slip")      {}  // handled below
        else if (arg == "--nx" && i + 1 < argc) Nx = std::stoi(argv[++i]);
        else if (arg == "--ny" && i + 1 < argc) Ny = std::stoi(argv[++i]);
    }

    bool slipWall = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--slip") slipWall = true;

    // ---- Configuration ----
    SimulationConfig config;
    config.dim          = 2;
    config.nGhost       = 3;
    config.useIGR       = useIGR;
    config.semiImplicit = useSemiImplicit;
    config.RKOrder      = 3;
    config.reconOrder   = ReconstructionOrder::WENO5;

    if (useSemiImplicit) {
        config.semiImplicitParams.cfl              = 0.8;
        config.semiImplicitParams.maxDt            = 1.0;
        config.semiImplicitParams.pressureTol      = 1e-8;
        config.semiImplicitParams.maxPressureIters = 500;
    } else {
        config.explicitParams.cfl   = 0.5;
        config.explicitParams.maxDt = 1.0;
    }

    if (useIGR) {
        config.igrParams.alphaCoeff       = 1.0;
        config.igrParams.IGRIters         = 5;
        config.igrParams.IGRWarmStartIters = 50;
    }

    config.validate();

    // ---- Print setup ----
    double cInf = std::sqrt(gamma_gas * pInf / rhoInf);

    rt.print("=== 2D Flow Over a Circle (Immersed Boundary) ===\n");
    rt.print("  Solver:    ", useSemiImplicit ? "Semi-implicit" : "Explicit", "\n");
    rt.print("  IGR:       ", useIGR ? "ON" : "OFF", "\n");
    rt.print("  Wall type: ", slipWall ? "Slip" : "No-slip", "\n");
    rt.print("  Grid:      ", Nx, " x ", Ny, "\n");
    rt.print("  Domain:    [0, ", Lx, "] x [0, ", Ly, "]\n");
    rt.print("  Circle:    centre=(", xc, ", ", yc, "), R=", R, "\n");
    rt.print("  Mach:      ", Mach, "\n");
    rt.print("  u_inf:     ", uInf, "\n");
    rt.print("  c_inf:     ", cInf, "\n");
    rt.print("  RK order:  ", config.RKOrder, "\n");
    rt.print("  End time:  ", endTime, "\n\n");

    // ---- Mesh ----
    RectilinearMesh mesh = rt.createUniformMesh(
        config, Nx, 0.0, Lx, Ny, 0.0, Ly);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow,  BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XHigh, BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YLow,  BoundaryCondition::SlipWall);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YHigh, BoundaryCondition::SlipWall);

    // ---- Immersed boundary: circle ----
    auto circle = std::make_shared<IBCircle>(xc, yc, R);
    circle->setWallType(slipWall ? IBBody::WallType::Slip
                                 : IBBody::WallType::NoSlip);

    ImmersedBoundaryMethod ibm;
    ibm.addBody(circle);
    ibm.classifyCells(mesh);

    // Count classified cells
    int nGhost = 0, nDead = 0;
    for (int k = 0; k < mesh.nz(); ++k)
        for (int j = 0; j < mesh.ny(); ++j)
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                if (ibm.cellType(idx) == CellType::Ghost) ++nGhost;
                if (ibm.cellType(idx) == CellType::Dead)  ++nDead;
            }
    rt.print("  IBM cells: ", nGhost, " ghost, ", nDead, " dead, ",
             Nx * Ny - nGhost - nDead, " fluid\n\n");

    // ---- Allocate state ----
    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos           = std::make_shared<IdealGasEOS>(gamma_gas, Rgas, config);
    auto riemannSolver = std::make_shared<HLLCSolver>(eos, config);
    auto igrSolver     = useIGR ? std::make_shared<IGRSolver>(config.igrParams)
                                : nullptr;

    // ---- Initialize: uniform free-stream everywhere ----
    PrimitiveState W0;
    W0.rho   = rhoInf;
    W0.u     = {uInf, 0.0, 0.0};
    W0.p     = pInf;
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

    // ---- Build solver and attach IBM ----
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
        rt.attachIBM(ibm, *explicitSolver);
        stepFn = [&](double targetDt) {
            return explicitSolver->step(config, mesh, state, targetDt);
        };
    }

    // ---- Run ----
    VTKSession vtk(rt, "2D_flow_over_circle", mesh, config);

    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime        = endTime,
                 .outputInterval = endTime / 200,
                 .printInterval  = 10});

    // ---- Recompute primitives from final conservative state ----
    if (useSemiImplicit)
        state.convertConservativeToPrimitiveVariables(mesh, eos);

    // ---- Print wake diagnostics ----
    // Sample velocity along the wake centreline (y = yc) behind the circle
    rt.print("\n=== Wake Centreline (y = ", yc, ") ===\n");
    {
        std::ostringstream hdr;
        hdr << "  " << std::left
            << std::setw(10) << "x"
            << std::setw(14) << "u"
            << std::setw(14) << "v"
            << std::setw(14) << "p"
            << std::setw(10) << "cellType"
            << "\n";
        hdr << "  " << std::string(60, '-') << "\n";
        rt.print(hdr.str());
    }

    int jc = static_cast<int>(yc / (Ly / Ny));
    int stride = std::max(1, Nx / 40);
    for (int i = 0; i < Nx; i += stride) {
        std::size_t idx = mesh.index(i, jc, 0);
        double xi = mesh.cellCentroidX(i);
        const char* ct = "fluid";
        if (ibm.cellType(idx) == CellType::Ghost) ct = "ghost";
        if (ibm.cellType(idx) == CellType::Dead)  ct = "dead";

        std::ostringstream row;
        row << "  " << std::fixed << std::setprecision(3)
            << std::setw(10) << xi
            << std::scientific << std::setprecision(4)
            << std::setw(14) << state.velU[idx]
            << std::setw(14) << state.velV[idx]
            << std::setw(14) << state.pres[idx]
            << "  " << ct
            << "\n";
        rt.print(row.str());
    }

    // ---- Drag/lift estimate via surface pressure integration ----
    // Approximate by summing (p - p_inf) * n * dA over ghost cells
    double Fx = 0.0, Fy = 0.0;
    double dx = Lx / Nx;
    double dy = Ly / Ny;
    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            std::size_t idx = mesh.index(i, j, 0);
            if (ibm.cellType(idx) != CellType::Ghost) continue;

            // Use the body's outward normal at this cell
            double cx_i = mesh.cellCentroidX(i);
            double cy_j = mesh.cellCentroidY(j);
            auto n = circle->outwardNormal(cx_i, cy_j, 0.0);

            double dp = state.pres[idx] - pInf;
            // Surface element ~ cell face area in 2D
            double dA = std::sqrt(dx * dx + dy * dy);  // approximate
            Fx += dp * n[0] * dA;
            Fy += dp * n[1] * dA;
        }
    }

    double qInf = 0.5 * rhoInf * uInf * uInf;
    double D = 2.0 * R;  // diameter
    double Cd = Fx / (qInf * D);
    double Cl = Fy / (qInf * D);

    rt.print("\n=== Force Coefficients (pressure only, approximate) ===\n");
    rt.print("  Cd ~ ", Cd, "\n");
    rt.print("  Cl ~ ", Cl, "\n");

    return 0;
}

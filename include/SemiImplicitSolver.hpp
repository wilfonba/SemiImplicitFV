#ifndef SEMI_IMPLICIT_SOLVER_HPP
#define SEMI_IMPLICIT_SOLVER_HPP

#include "RectilinearMesh.hpp"
#include "State.hpp"
#include "RiemannSolver.hpp"
#include "IGR.hpp"
#include "EquationOfState.hpp"
#include "PressureSolver.hpp"
#include <memory>
#include <vector>

namespace SemiImplicitFV {

// Parameters for the semi-implicit solver
struct SemiImplicitParams {
    double cfl;               // CFL number (based on material velocity)
    double maxDt;             // Maximum time step
    double minDt;             // Minimum time step
    int maxPressureIters;     // Max iterations for pressure solve
    double pressureTol;       // Pressure solve tolerance
    bool useIGR;              // Enable IGR regularization

    SemiImplicitParams()
        : cfl(0.8)            // Can use larger CFL since no acoustic restriction
        , maxDt(1e-2)
        , minDt(1e-12)
        , maxPressureIters(100)
        , pressureTol(1e-8)
        , useIGR(true)
    {}
};

// Semi-implicit time stepper for compressible flow (Kwatra et al.)
// Combined with Information Geometric Regularization (IGR)
//
// Algorithm:
// 1. Advection step: Compute (ρ)*, (ρu)*, E* using pressure-free fluxes
// 2. Pressure advection: p^a = p^n - Δt(u^n · ∇p^n)
// 3. IGR step: Solve for entropic pressure Σ
// 4. Pressure step: Solve modified Helmholtz equation for p^{n+1}
// 5. Correction: Update momentum and energy with ∇(p+Σ)
class SemiImplicitSolver {
public:
    SemiImplicitSolver(
        std::shared_ptr<RiemannSolver> riemannSolver,
        std::shared_ptr<PressureSolver> pressureSolver,
        std::shared_ptr<EquationOfState> eos,
        std::shared_ptr<IGRSolver> igrSolver = nullptr,
        const SemiImplicitParams& params = SemiImplicitParams()
    );

    ~SemiImplicitSolver() = default;

    void setParameters(const SemiImplicitParams& params) { params_ = params; }
    const SemiImplicitParams& parameters() const { return params_; }

    // Perform one time step
    double step(RectilinearMesh& mesh, double targetDt = -1.0);

    // Compute stable time step (CFL based on material velocity only)
    double computeAdvectiveTimeStep(const RectilinearMesh& mesh) const;

    // Access components
    RiemannSolver& riemannSolver() { return *riemannSolver_; }
    PressureSolver& pressureSolver() { return *pressureSolver_; }
    const EquationOfState& eos() const { return *eos_; }

    // Statistics
    int lastPressureIterations() const { return lastPressureIters_; }

private:
    std::shared_ptr<RiemannSolver> riemannSolver_;
    std::shared_ptr<PressureSolver> pressureSolver_;
    std::shared_ptr<EquationOfState> eos_;
    std::shared_ptr<IGRSolver> igrSolver_;
    SemiImplicitParams params_;

    int lastPressureIters_;

    // Internal storage (SoA, sized to mesh.totalCells())
    std::vector<double> rhoStar_;
    std::vector<double> rhoUStar_;
    std::vector<double> rhoVStar_;
    std::vector<double> rhoWStar_;
    std::vector<double> rhoEStar_;

    std::vector<double> pAdvected_;
    std::vector<double> rhoc2_;
    std::vector<double> pressureRhs_;
    std::vector<double> pressure_;
    std::vector<double> divUstar_;
    std::vector<GradientTensor> gradU_;

    // Step 1: Explicit advection (pressure-free)
    void advectionStep(RectilinearMesh& mesh, double dt);

    // Step 2: Advect pressure (for RHS of pressure equation)
    void advectPressure(const RectilinearMesh& mesh, double dt);

    // Step 3: Solve IGR for entropic pressure Σ
    void solveIGR(RectilinearMesh& mesh);

    // Step 4: Solve pressure Poisson equation
    void solvePressure(RectilinearMesh& mesh, double dt);

    // Step 5: Correct momentum and energy
    void correctionStep(RectilinearMesh& mesh, double dt);

    // Compute divergence of velocity at each cell
    void computeDivergence(const RectilinearMesh& mesh, std::vector<double>& divU);

    // Compute velocity gradients for IGR
    void computeVelocityGradients(const RectilinearMesh& mesh);

    // Gather mesh SoA fields into a PrimitiveState bundle
    static PrimitiveState gatherPrimitive(const RectilinearMesh& mesh, std::size_t idx);

    // Write star conservative state into mesh and convert to primitives
    void writeStarToMesh(RectilinearMesh& mesh);

    // Resize internal storage to match mesh
    void ensureStorage(const RectilinearMesh& mesh);
};

} // namespace SemiImplicitFV

#endif // SEMI_IMPLICIT_SOLVER_HPP


#ifndef SEMI_IMPLICIT_SOLVER_HPP
#define SEMI_IMPLICIT_SOLVER_HPP

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
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
    int RKOrder;              // Runge-Kutta order for advection step (1, 2, or 3)
    double pressureTol;       // Pressure solve tolerance
    bool useIGR;              // Enable IGR regularization

    SemiImplicitParams()
        : cfl(0.8)            // Can use larger CFL since no acoustic restriction
        , maxDt(1e-2)
        , minDt(1e-12)
        , maxPressureIters(100)
        , RKOrder(1)
        , pressureTol(1e-8)
        , useIGR(true)
    {}
};

// Semi-implicit time stepper for compressible flow (Kwatra et al.)
// Combined with Information Geometric Regularization (IGR)
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
    double step(const RectilinearMesh& mesh, SolutionState& state, double targetDt = -1.0);

    // Compute stable time step (CFL based on material velocity only)
    double computeAdvectiveTimeStep(const RectilinearMesh& mesh, const SolutionState& state) const;

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

    void advectionStep(const RectilinearMesh& mesh, const SolutionState& state, double dt);
    void advectPressure(const RectilinearMesh& mesh, const SolutionState& state, double dt);
    void solveIGR(const RectilinearMesh& mesh, SolutionState& state);
    void solvePressure(const RectilinearMesh& mesh, SolutionState& state, double dt);
    void correctionStep(const RectilinearMesh& mesh, SolutionState& state, double dt);
    void computeDivergence(const RectilinearMesh& mesh, const SolutionState& state, std::vector<double>& divU);
    void computeVelocityGradients(const RectilinearMesh& mesh, const SolutionState& state);
    void writeStarToState(const RectilinearMesh& mesh, SolutionState& state);
    void ensureStorage(const RectilinearMesh& mesh);
};

} // namespace SemiImplicitFV

#endif // SEMI_IMPLICIT_SOLVER_HPP

#ifndef SEMI_IMPLICIT_SOLVER_HPP
#define SEMI_IMPLICIT_SOLVER_HPP

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "RiemannSolver.hpp"
#include "Reconstruction.hpp"
#include "IGR.hpp"
#include "EquationOfState.hpp"
#include "PressureSolver.hpp"
#include "SimulationConfig.hpp"
#include <memory>
#include <vector>

#ifdef ENABLE_MPI
#include "HaloExchange.hpp"
#endif

namespace SemiImplicitFV {

// Semi-implicit time stepper for compressible flow (Kwatra et al.)
// Combined with Information Geometric Regularization (IGR)
class SemiImplicitSolver {
public:
    SemiImplicitSolver(
        const RectilinearMesh& mesh,
        std::shared_ptr<RiemannSolver> riemannSolver,
        std::shared_ptr<PressureSolver> pressureSolver,
        std::shared_ptr<EquationOfState> eos,
        std::shared_ptr<IGRSolver> igrSolver,
        const SimulationConfig& config
    );

    ~SemiImplicitSolver() = default;

    // Perform one time step
    double step(const SimulationConfig& config, const RectilinearMesh& mesh, SolutionState& state, double targetDt = -1.0);

#ifdef ENABLE_MPI
    void setHaloExchange(HaloExchange* halo) { halo_ = halo; }
#endif

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
    Reconstructor reconstructor_;

#ifdef ENABLE_MPI
    HaloExchange* halo_ = nullptr;
#endif

    std::vector<double> pressureRhs_;
    std::vector<double> pressure_;
    std::vector<double> divUstar_;
    std::vector<GradientTensor> gradU_;

    // RHS storage (flux divergence per conservative variable)
    std::vector<double> rhsRho_;
    std::vector<double> rhsRhoU_;
    std::vector<double> rhsRhoV_;
    std::vector<double> rhsRhoW_;
    std::vector<double> rhsRhoE_;
    std::vector<double> rhsPadvected_;

    void computeRHS(const SimulationConfig& config, const RectilinearMesh& mesh, SolutionState& state);
    void solveIGR(const SimulationConfig& config, const RectilinearMesh& mesh, SolutionState& state);
    void solvePressure(const RectilinearMesh& mesh, SolutionState& state, double dt);
    void correctionStep(const RectilinearMesh& mesh, SolutionState& state, double dt);
    void computeDivergence(const RectilinearMesh& mesh, const SolutionState& state, std::vector<double>& divU);
    void computeVelocityGradients(const SimulationConfig& config, const RectilinearMesh& mesh, const SolutionState& state);
    void writeStarToState(const RectilinearMesh& mesh, SolutionState& state);
};

} // namespace SemiImplicitFV

#endif // SEMI_IMPLICIT_SOLVER_HPP

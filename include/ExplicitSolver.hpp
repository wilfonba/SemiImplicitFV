#ifndef EXPLICIT_SOLVER_HPP
#define EXPLICIT_SOLVER_HPP

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "SimulationConfig.hpp"
#include "State.hpp"
#include "RiemannSolver.hpp"
#include "Reconstruction.hpp"
#include "IGR.hpp"
#include "EquationOfState.hpp"
#include <memory>
#include <vector>

#ifdef ENABLE_MPI
#include "HaloExchange.hpp"
#endif

namespace SemiImplicitFV {

class ExplicitSolver {
public:
    ExplicitSolver(
        const RectilinearMesh& mesh,
        std::shared_ptr<RiemannSolver> riemannSolver,
        std::shared_ptr<EquationOfState> eos,
        std::shared_ptr<IGRSolver> igrSolver,
        const SimulationConfig& config
    );

    ~ExplicitSolver() = default;

    double step(const SimulationConfig& config,
                const RectilinearMesh& mesh,
                SolutionState& state,
                double targetDt = -1.0);

#ifdef ENABLE_MPI
    void setHaloExchange(HaloExchange* halo) { halo_ = halo; }
#endif

    RiemannSolver& riemannSolver() { return *riemannSolver_; }
    const EquationOfState& eos() const { return *eos_; }
    const Reconstructor& reconstructor() const { return reconstructor_; }

private:
    std::shared_ptr<RiemannSolver> riemannSolver_;
    std::shared_ptr<EquationOfState> eos_;
    std::shared_ptr<IGRSolver> igrSolver_;
    ExplicitParams params_;
    Reconstructor reconstructor_;

#ifdef ENABLE_MPI
    HaloExchange* halo_ = nullptr;
#endif

    // RHS storage (flux divergence per conservative variable)
    std::vector<double> rhsRho_;
    std::vector<double> rhsRhoU_;
    std::vector<double> rhsRhoV_;
    std::vector<double> rhsRhoW_;
    std::vector<double> rhsRhoE_;

    std::vector<GradientTensor> gradU_;

    void computeRHS(const SimulationConfig& config,
            const RectilinearMesh& mesh, SolutionState& state);

    void solveIGR(const SimulationConfig& config, const RectilinearMesh& mesh, SolutionState& state);
    void computeVelocityGradients(const SimulationConfig& config, const RectilinearMesh& mesh, const SolutionState& state);

};

} // namespace SemiImplicitFV

#endif // EXPLICIT_SOLVER_HPP

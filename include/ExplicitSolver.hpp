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
#include "MixtureEOS.hpp"
#include <memory>
#include <vector>
#include "HaloExchange.hpp"

namespace SemiImplicitFV {

class ImmersedBoundaryMethod;

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

    void setHaloExchange(HaloExchange* halo) { halo_ = halo; }
    void setIBM(ImmersedBoundaryMethod* ibm) {
        ibm_ = ibm;
        if (igrSolver_) igrSolver_->setIBM(ibm);
    }

    RiemannSolver& riemannSolver() { return *riemannSolver_; }
    const EquationOfState& eos() const { return *eos_; }
    const Reconstructor& reconstructor() const { return reconstructor_; }

private:
    std::shared_ptr<RiemannSolver> riemannSolver_;
    std::shared_ptr<EquationOfState> eos_;
    std::shared_ptr<IGRSolver> igrSolver_;
    ExplicitParams params_;
    Reconstructor reconstructor_;

    HaloExchange* halo_ = nullptr;
    ImmersedBoundaryMethod* ibm_ = nullptr;

    // RHS storage (flux divergence per conservative variable)
    std::vector<double> rhsRho_;
    std::vector<double> rhsRhoU_;
    std::vector<double> rhsRhoV_;
    std::vector<double> rhsRhoW_;
    std::vector<double> rhsRhoE_;

    // Multi-phase RHS storage
    std::vector<std::vector<double>> rhsAlphaRho_;  // N arrays
    std::vector<std::vector<double>> rhsAlpha_;      // N arrays

    // Scratch array for velocity divergence (alpha source term)
    std::vector<double> divU_;

    std::vector<GradientTensor> gradU_;

    void computeRHS(const SimulationConfig& config,
            const RectilinearMesh& mesh, SolutionState& state);

    void solveIGR(const SimulationConfig& config, const RectilinearMesh& mesh, SolutionState& state);
    void computeVelocityGradients(const SimulationConfig& config, const RectilinearMesh& mesh, const SolutionState& state);

};

} // namespace SemiImplicitFV

#endif // EXPLICIT_SOLVER_HPP

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

namespace SemiImplicitFV {

struct ExplicitParams{
    double cfl;               // CFL number (based on material velocity)
    double constDt;           // Constant time step (if > 0, overrides CFL-based time step)
    double maxDt;             // Maximum time step
    double minDt;             // Minimum time step
    int RKOrder;              // Runge-Kutta order (1, 2, or 3)
    bool useIGR;              // Enable IGR

    ReconstructionOrder reconOrder = ReconstructionOrder::FirstOrder;

    ExplicitParams()
        : cfl(0.8)
        , constDt(-1.0)
        , maxDt(1e-4)
        , minDt(1e-12)
        , RKOrder(1)
        , useIGR(true)
    {}
};

class ExplicitSolver {
public:
    ExplicitSolver(
        std::shared_ptr<RiemannSolver> riemannSolver,
        std::shared_ptr<EquationOfState> eos,
        std::shared_ptr<IGRSolver> igrSolver = nullptr,
        const ExplicitParams& params = ExplicitParams()
    );

    ~ExplicitSolver() = default;

    double step(const SimulationConfig& config,
            const RectilinearMesh& mesh, SolutionState& state, double targetDt = -1.0);

    double computeAcousticTimeStep(const RectilinearMesh& mesh, const SolutionState& state) const;

    RiemannSolver& riemannSolver() { return *riemannSolver_; }
    const EquationOfState& eos() const { return *eos_; }
    const Reconstructor& reconstructor() const { return reconstructor_; }

private:
    std::shared_ptr<RiemannSolver> riemannSolver_;
    std::shared_ptr<EquationOfState> eos_;
    std::shared_ptr<IGRSolver> igrSolver_;
    ExplicitParams params_;
    Reconstructor reconstructor_;

    // RHS storage (flux divergence per conservative variable)
    std::vector<double> rhsRho_;
    std::vector<double> rhsRhoU_;
    std::vector<double> rhsRhoV_;
    std::vector<double> rhsRhoW_;
    std::vector<double> rhsRhoE_;

    void ensureStorage(const RectilinearMesh& mesh);
    void computeRHS(const SimulationConfig& config,
            const RectilinearMesh& mesh, SolutionState& state);
};

} // namespace SemiImplicitFV

#endif // EXPLICIT_SOLVER_HPP

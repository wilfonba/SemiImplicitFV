#ifndef EXPLICITSOLVER_HPP
#define EXPLICITSOLVER_HPP

#include "RectilinearMesh.hpp"
#include "State.hpp"
#include "RiemannSolver.hpp"
#include "IGR.hpp"
#include "EquationOfState.hpp"
#include <memory>
#include <vector>

namespace SemiImplicitFV {

struct ExplicitParams{
    double cfl;               // CFL number (based on material velocity)
    double maxDt;             // Maximum time step
    double minDt;             // Minimum time step
    bool useIGR;              // Enable IGR

    ExplicitParams()
        : cfl(0.8)
        , maxDt(1e-2)
        , minDt(1e-12)
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

    // Compute stable time step (CFL based on material velocity only)
    double computeAcousticTimeStep(const RectilinearMesh& mesh) const;

    // Access components
    RiemannSolver& riemannSolver() { return *riemannSolver_; }
    const EquationOfState& eos() const { return *eos_; }

private:
    std::shared_ptr<RiemannSolver> riemannSolver_;
    std::shared_ptr<EquationOfState> eos_;
    std::shared_ptr<IGRSolver> igrSolver_;
    ExplicitParams params_;
};


} // namespace SemiImplicitFV

#endif /* end of include guard EXPLICITSOLVER_HPP */


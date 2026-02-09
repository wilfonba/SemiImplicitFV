#ifndef SOLUTION_STATE_HPP
#define SOLUTION_STATE_HPP

#include "State.hpp"
#include "SimulationConfig.hpp"
#include <vector>
#include <memory>
#include <cstddef>

namespace SemiImplicitFV {

class RectilinearMesh;
class EquationOfState;

enum class VarSet {
    CONS,        // Conservative variables
    PRIM,        // Primitive variables
    SIGMA,       // Entropic pressure (IGR)
};

/// Struct-of-arrays storage for all per-cell field data.
///
/// Arrays are sized to totalCells (including ghost cells).  Indexing is
/// done via RectilinearMesh::index(i,j,k), which returns the flat offset.
class SolutionState {
public:
    SolutionState() = default;

    /// Allocate field arrays to the given size, zero-filled.
    /// Backup arrays for multi-stage time stepping are allocated when config.RKOrder > 1.
    void allocate(std::size_t totalCells, const SimulationConfig& config);

    /// Number of cells this state is allocated for.
    std::size_t size() const { return rho.size(); }

    /// Spatial dimensionality (1, 2, or 3).
    int dim() const { return dim_; }

    /// Copy primitive field values from src to dst, applying velocity sign multipliers.
    void copyCell_P(std::size_t dst, std::size_t src,
                            double sU, double sV, double sW);
    /// Copy conservative field values from src to dst, applying velocity sign multipliers.
    void copyCell_C(std::size_t dst, std::size_t src,
                            double sU, double sV, double sW);
    /// Copy all field values from src to dst, applying velocity sign multipliers.
    void copyCell(std::size_t dst, std::size_t src,
                            double sU, double sV, double sW);

    /// Gather a ConservativeState bundle from a flat index.
    ConservativeState getConservativeState(std::size_t idx) const;

    /// Scatter a ConservativeState bundle into a flat index.
    void setConservativeState(std::size_t idx, const ConservativeState& U);

    /// Gather a PrimitiveState bundle from a flat index.
    PrimitiveState getPrimitiveState(std::size_t idx) const;

    /// Scatter a PrimitiveState bundle into a flat index.
    void setPrimitiveState(std::size_t idx, const PrimitiveState& W);

    /// Convert from conservative to primitive variables.
    void convertConservativeToPrimitiveVariables(
        const RectilinearMesh& mesh,
        const std::shared_ptr<EquationOfState>& eos
        );

    /// Convert from primitive to conservative variables.
    void convertPrimitiveToConservativeVariables(
        const RectilinearMesh& mesh,
        const std::shared_ptr<EquationOfState>& eos
        );

    // Conservative variables
    std::vector<double> rho;   // density
    std::vector<double> rhoU;  // x-momentum
    std::vector<double> rhoV;  // y-momentum
    std::vector<double> rhoW;  // z-momentum
    std::vector<double> rhoE;  // total energy

    // Primitive variables
    std::vector<double> velU;  // x-velocity
    std::vector<double> velV;  // y-velocity
    std::vector<double> velW;  // z-velocity
    std::vector<double> pres;  // pressure
    std::vector<double> temp;  // temperature
    std::vector<double> sigma; // entropic pressure (IGR)

    // Backup conservative variables (for multi-stage time stepping)
    std::vector<double> rho0;
    std::vector<double> rhoU0;
    std::vector<double> rhoV0;
    std::vector<double> rhoW0;
    std::vector<double> rhoE0;

    // Start state for semi-implicit solve
    std::vector<double> rhoUStar;
    std::vector<double> rhoVStar;
    std::vector<double> rhoWStar;
    std::vector<double> rhoEstar;
    std::vector<double> pAdvected;
    std::vector<double> rhoc2;      // rho * c^2 for implicit pressure solve
    std::vector<double> divUStar;   // divergence of velocity field at start of time step

    // Auxiliary variable
    std::vector<double> aux;

    /// Save conservative variables for a single cell to backup storage.
    void saveConservativeCell(std::size_t idx);

    /// Smooth all conservative and primitive fields using explicit heat equation
    /// iterations (forward Euler with diffusion number 1/(2*dim)).
    /// Call after setting the sharp IC and before the time loop.
    void smoothFields(const RectilinearMesh& mesh, int nIterations);

private:
    int dim_ = 3;
};

} // namespace SemiImplicitFV

#endif // SOLUTION_STATE_HPP

#ifndef SOLUTION_STATE_HPP
#define SOLUTION_STATE_HPP

#include "State.hpp"
#include <vector>
#include <cstddef>

namespace SemiImplicitFV {

enum class VarSet {
    CONS,        // Conservative variables
    PRIM         // Primitive variables
};

/// Struct-of-arrays storage for all per-cell field data.
///
/// Arrays are sized to totalCells (including ghost cells).  Indexing is
/// done via RectilinearMesh::index(i,j,k), which returns the flat offset.
class SolutionState {
public:
    SolutionState() = default;

    /// Allocate field arrays to the given size, zero-filled.
    /// Only velocity components active for the given dimensionality are allocated.
    void allocate(std::size_t totalCells, int dim);

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

    // Auxiliary variable
    std::vector<double> aux;

private:
    int dim_ = 3;
};

} // namespace SemiImplicitFV

#endif // SOLUTION_STATE_HPP

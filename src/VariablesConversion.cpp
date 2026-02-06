#include "VariablesConversion.hpp"
#include "State.hpp"

namespace SemiImplicitFV {

void convertConservativeToPrimitiveVariables(
        const RectilinearMesh& mesh,
        SolutionState& state,
        const std::shared_ptr<EquationOfState>& eos
        )
{
    int dim = state.dim();
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                ConservativeState U = state.getConservativeState(idx);
                PrimitiveState W = eos->toPrimitive(U);
                state.velU[idx] = W.u[0];
                if (dim >= 2) state.velV[idx] = W.u[1];
                if (dim >= 3) state.velW[idx] = W.u[2];
                state.pres[idx] = W.p;
                state.temp[idx] = W.T;
            }
        }
    }
}

} // namespace SemiImplicitFV

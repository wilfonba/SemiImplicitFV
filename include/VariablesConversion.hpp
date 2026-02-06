#ifndef VARIABLES_CONVERSION_HPP
#define VARIABLES_CONVERSION_HPP

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "EquationOfState.hpp"
#include <memory>

namespace SemiImplicitFV {

void convertConservativeToPrimitiveVariables(
        const RectilinearMesh& mesh,
        SolutionState& state,
        const std::shared_ptr<EquationOfState>& eos
        );
}

#endif // VARIABLES_CONVERSION_HPP

#include "ICPatch.hpp"
#include "MixtureEOS.hpp"
#include "ExpressionEvaluator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace SemiImplicitFV {

// Helper: apply an ICState to a single cell
static void applyCellState(
    std::size_t idx,
    const ICState& st,
    SolutionState& state,
    const EquationOfState& eos,
    const SimulationConfig& config)
{
    PrimitiveState W;
    W.rho = st.rho;
    W.u   = {st.u, st.v, st.w};
    W.p   = st.p;
    W.sigma = st.sigma;
    W.T   = eos.temperature(W);

    state.setPrimitiveState(idx, W);

    if (config.isMultiPhase() && !st.alpha.empty()) {
        int nPhases = config.multiPhaseParams.nPhases;
        double rho = 0.0;
        for (int ph = 0; ph < nPhases; ++ph) {
            state.alpha[ph][idx] = st.alpha[ph];
            if (!st.alphaRho.empty()) {
                state.alphaRho[ph][idx] = st.alphaRho[ph];
            } else {
                // Default: alphaRho = alpha * rho
                state.alphaRho[ph][idx] = st.alpha[ph] * st.rho;
            }
            rho += state.alphaRho[ph][idx];
        }
        state.rho[idx] = rho;

        // Conservative variables via MixtureEOS
        double ke = 0.5 * rho * (st.u*st.u + st.v*st.v + st.w*st.w);
        state.rhoE[idx] = MixtureEOS::mixtureTotalEnergy(
            rho, st.p, st.alpha, ke, config.multiPhaseParams);
        state.rhoU[idx] = rho * st.u;
        if (config.dim >= 2) state.rhoV[idx] = rho * st.v;
        if (config.dim >= 3) state.rhoW[idx] = rho * st.w;
    } else {
        state.setConservativeState(idx, eos.toConservative(W));
    }
}

void applyInitialConditions(
    const RectilinearMesh& mesh,
    SolutionState& state,
    const EquationOfState& eos,
    const SimulationConfig& config,
    const ICState& defaultState,
    const std::vector<ICPatch>& patches)
{
    int dim = config.dim;

    // Step 1: Fill all cells with default state
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                applyCellState(idx, defaultState, state, eos, config);
            }
        }
    }

    // Step 2: Apply patches in order
    for (const auto& patch : patches) {
        if (!patch.geometry) continue;

        bool isAnalytic = !patch.expressions.empty();

        // Create expression evaluator if needed
        std::unique_ptr<ExpressionEvaluator> evaluator;
        if (isAnalytic) {
            evaluator = std::make_unique<ExpressionEvaluator>();
            for (const auto& [name, expr] : patch.expressions) {
                evaluator->addExpression(name, expr);
            }
        }

        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    double x = mesh.cellCentroidX(i);
                    double y = (dim >= 2) ? mesh.cellCentroidY(j) : 0.0;
                    double z = (dim >= 3) ? mesh.cellCentroidZ(k) : 0.0;

                    if (patch.geometry->contains(x, y, z)) {
                        std::size_t idx = mesh.index(i, j, k);

                        if (isAnalytic) {
                            // Evaluate expressions at this cell
                            ICState evalState = patch.state;
                            evaluator->setCoordinates(x, y, z);

                            for (const auto& [name, expr] : patch.expressions) {
                                double val = evaluator->evaluate(name);
                                if (name == "rho") evalState.rho = val;
                                else if (name == "u") evalState.u = val;
                                else if (name == "v") evalState.v = val;
                                else if (name == "w") evalState.w = val;
                                else if (name == "p") evalState.p = val;
                                else if (name == "sigma") evalState.sigma = val;
                                // alpha_0, alpha_1, etc.
                                else if (name.substr(0, 6) == "alpha_") {
                                    int ph = std::stoi(name.substr(6));
                                    if (ph >= 0 && ph < static_cast<int>(evalState.alpha.size()))
                                        evalState.alpha[ph] = val;
                                }
                                // alphaRho_0, alphaRho_1, etc.
                                else if (name.substr(0, 9) == "alphaRho_") {
                                    int ph = std::stoi(name.substr(9));
                                    if (ph >= 0 && ph < static_cast<int>(evalState.alphaRho.size()))
                                        evalState.alphaRho[ph] = val;
                                }
                            }
                            applyCellState(idx, evalState, state, eos, config);
                        } else {
                            applyCellState(idx, patch.state, state, eos, config);
                        }
                    }
                }
            }
        }
    }
}

} // namespace SemiImplicitFV

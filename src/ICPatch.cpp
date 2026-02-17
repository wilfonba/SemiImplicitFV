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

// Helper: blend ICState fields based on blend factor (0 = keep existing, 1 = full patch)
static void blendCellState(
    std::size_t idx,
    const ICState& patchState,
    const ICState& defaultState,
    double blendFactor,
    SolutionState& state,
    const EquationOfState& eos,
    const SimulationConfig& config)
{
    double bf = std::max(0.0, std::min(1.0, blendFactor));
    double ibf = 1.0 - bf;

    ICState blended;
    blended.rho = ibf * defaultState.rho + bf * patchState.rho;
    blended.u   = ibf * defaultState.u   + bf * patchState.u;
    blended.v   = ibf * defaultState.v   + bf * patchState.v;
    blended.w   = ibf * defaultState.w   + bf * patchState.w;
    blended.p   = ibf * defaultState.p   + bf * patchState.p;

    if (config.isMultiPhase() && !patchState.alpha.empty()) {
        int nPhases = config.multiPhaseParams.nPhases;
        blended.alpha.resize(nPhases);
        blended.alphaRho.resize(nPhases);
        for (int ph = 0; ph < nPhases; ++ph) {
            double aDefault = (ph < static_cast<int>(defaultState.alpha.size()))
                ? defaultState.alpha[ph] : config.multiPhaseParams.alphaMin;
            double aRhoDefault = (ph < static_cast<int>(defaultState.alphaRho.size()))
                ? defaultState.alphaRho[ph] : aDefault * defaultState.rho;

            blended.alpha[ph] = ibf * aDefault + bf * patchState.alpha[ph];
            if (!patchState.alphaRho.empty()) {
                blended.alphaRho[ph] = ibf * aRhoDefault + bf * patchState.alphaRho[ph];
            } else {
                blended.alphaRho[ph] = blended.alpha[ph] * blended.rho;
            }
        }
    }

    applyCellState(idx, blended, state, eos, config);
}

void applyInitialConditions(
    const RectilinearMesh& mesh,
    SolutionState& state,
    const EquationOfState& eos,
    const SimulationConfig& config,
    const ICState& defaultState,
    const std::vector<ICPatch>& patches)
{
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
        bool hasDiffuse = (patch.interface.diffuseWidth > 0.0);

        // Compute epsilon for diffuse interface
        double epsilon = 0.0;
        if (hasDiffuse) {
            double dxMin = mesh.dx(0);
            if (config.dim >= 2) dxMin = std::min(dxMin, mesh.dy(0));
            if (config.dim >= 3) dxMin = std::min(dxMin, mesh.dz(0));
            epsilon = patch.interface.diffuseWidth * dxMin;
        }

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
                    double y = mesh.cellCentroidY(j);
                    double z = mesh.cellCentroidZ(k);

                    if (hasDiffuse) {
                        double sd = patch.geometry->signedDistance(x, y, z);
                        double blendFactor;
                        if (patch.interface.profile == "linear") {
                            blendFactor = 0.5 - sd / (2.0 * epsilon);
                            blendFactor = std::max(0.0, std::min(1.0, blendFactor));
                        } else {
                            // tanh profile
                            blendFactor = 0.5 * (1.0 - std::tanh(sd / epsilon));
                        }

                        if (blendFactor > 1e-12) {
                            blendCellState(mesh.index(i,j,k), patch.state,
                                           defaultState, blendFactor, state, eos, config);
                        }
                    } else if (patch.geometry->contains(x, y, z)) {
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

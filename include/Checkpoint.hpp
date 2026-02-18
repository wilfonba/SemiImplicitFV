#ifndef CHECKPOINT_HPP
#define CHECKPOINT_HPP

#include "SimulationConfig.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"

#include <string>
#include <cstdint>

namespace SemiImplicitFV {

/// Magic number: ASCII "SIFV_CKP" as uint64_t (little-endian)
static constexpr uint64_t CHECKPOINT_MAGIC = 0x504B435F56464953ULL;
static constexpr int32_t  CHECKPOINT_VERSION = 1;

/// Write a binary checkpoint file to dir/checkpoint.RRRR.bin.
/// Only conservative variables and multi-phase fields are written.
void writeCheckpoint(
    const std::string& dir,
    const RectilinearMesh& mesh,
    const SolutionState& state,
    const SimulationConfig& config,
    int rank);

/// Load a binary checkpoint file and populate state + config.time/step.
/// Validates that dim, nx, ny, nz, nGhost match the current mesh.
/// Caller must call convertConservativeToPrimitiveVariables() and apply BCs after.
void loadCheckpoint(
    const std::string& filepath,
    const RectilinearMesh& mesh,
    SolutionState& state,
    SimulationConfig& config);

} // namespace SemiImplicitFV

#endif // CHECKPOINT_HPP

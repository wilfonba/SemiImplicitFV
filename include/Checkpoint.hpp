#ifndef CHECKPOINT_HPP
#define CHECKPOINT_HPP

#include "SimulationConfig.hpp"
#include <stdint.h>

struct RectilinearMesh;
struct SolutionState;

/* Magic number: ASCII "SIFV_CKP" as uint64_t (little-endian) */
static const uint64_t CHECKPOINT_MAGIC   = 0x504B435F56464953ULL;
static const int32_t  CHECKPOINT_VERSION = 1;

/* Write a binary checkpoint file to dir/checkpoint.RRRR.bin.
   Only conservative variables and multi-phase fields are written. */
void write_checkpoint(
    const char* dir,
    const struct RectilinearMesh* mesh,
    const struct SolutionState* state,
    const struct SimulationConfig* config,
    int rank);

/* Load a binary checkpoint file and populate state + config.time/step.
   Validates that dim, nx, ny, nz, nGhost match the current mesh.
   Caller must call state_cons_to_prim() and apply BCs after. */
void load_checkpoint(
    const char* filepath,
    const struct RectilinearMesh* mesh,
    struct SolutionState* state,
    struct SimulationConfig* config);

#endif /* CHECKPOINT_HPP */

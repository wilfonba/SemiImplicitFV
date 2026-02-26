#ifndef VTK_WRITER_HPP
#define VTK_WRITER_HPP

#include "SimulationConfig.hpp"
#include <stddef.h>

struct RectilinearMesh;
struct SolutionState;

enum VTKFormat {
    VTK_TEXT,
    VTK_RAW
};

/* Write a single .vtr piece file.
   pieceExtent: {i0, i1, j0, j1, k0, k1} local extent within global grid.
   If NULL, uses full grid (serial mode).
   rank: MPI rank for "Rank" cell data field (-1 = omit). */
void vtk_write_vtr(const char* filename,
                   const struct RectilinearMesh* mesh,
                   const struct SolutionState* state,
                   const struct SimulationConfig* config,
                   const int* pieceExtent,
                   int rank,
                   enum VTKFormat format);

/* Write .pvtr parallel meta-file referencing piece files.
   Only rank 0 calls this in MPI.
   pieceExtents: flat array of nPieces*6 ints.
   pieceFiles: array of nPieces null-terminated strings. */
void vtk_write_pvtr(const char* filename,
                    int globalNx, int globalNy, int globalNz,
                    const int* pieceExtents,
                    const char* const* pieceFiles,
                    int nPieces,
                    const struct SimulationConfig* config,
                    enum VTKFormat format);

/* Three-phase .pvd time-series file:
     mode='w' -- write header
     mode='a' -- append timestep entry
     mode='c' -- close (no-op, file is always valid) */
void vtk_write_pvd(const char* filename,
                   char mode,
                   double time,
                   const char* dataFile);

#endif /* VTK_WRITER_HPP */

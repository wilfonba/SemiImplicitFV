#ifndef MPI_CONTEXT_HPP
#define MPI_CONTEXT_HPP

#include <mpi.h>

struct MPIContext {
    MPI_Comm cartComm;
    int rank;
    int size;
    int dims[3];
    int coords[3];
    int neighbors[6];
    double* localXNodes;
    int localXNodesSize;
    double* localYNodes;
    int localYNodesSize;
    double* localZNodes;
    int localZNodesSize;
    int localNx;
    int localNy;
    int localNz;
    int localExtent[6];
};

/* Create a Cartesian topology from global grid parameters.
   globalNx/Ny/Nz: Number of cells in each direction (global).
   xNodes/yNodes/zNodes: Global node coordinate arrays (nNodes entries each).
   dim: Spatial dimensionality (1, 2, or 3).
   periods: Periodic flags per direction (x, y, z).
   procsPerDir: Desired process counts per direction (0 = auto). */
void mpi_context_create(struct MPIContext* ctx,
                         int globalNx, int globalNy, int globalNz,
                         const double* xNodes, int nxNodes,
                         const double* yNodes, int nyNodes,
                         const double* zNodes, int nzNodes,
                         int dim,
                         const int periods[3],
                         const int procsPerDir[3]);

void mpi_context_free(struct MPIContext* ctx);

/* True (non-zero) if this rank owns a physical (non-MPI) boundary on the given face. */
static inline int mpi_is_physical_boundary(const struct MPIContext* ctx, int face) {
    return ctx->neighbors[face] == MPI_PROC_NULL;
}

/* True (non-zero) iff the neighbour on this face is a different rank — i.e.,
 * an actual MPI exchange is needed.  Self (single-rank periodic wrap) and
 * MPI_PROC_NULL (physical boundary) both return false. */
static inline int mpi_neighbor_is_remote(const struct MPIContext* ctx, int face) {
    int n = ctx->neighbors[face];
    return (n != MPI_PROC_NULL) && (n != ctx->rank);
}

#endif /* MPI_CONTEXT_HPP */

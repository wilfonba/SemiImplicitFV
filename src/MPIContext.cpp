#include "MPIContext.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

/* Slice a global node array into local nodes for a given coordinate. */
static void slice_nodes(const double* globalNodes, int globalN, int nProcs, int coord,
                        double** outNodes, int* outSize) {
    int base = globalN / nProcs;
    int remainder = globalN % nProcs;

    int startCell = 0;
    for (int p = 0; p < coord; ++p) {
        startCell += base + (p < remainder ? 1 : 0);
    }
    int localN = base + (coord < remainder ? 1 : 0);

    /* localN+1 nodes */
    *outSize = localN + 1;
    *outNodes = (double*)std::malloc((size_t)(localN + 1) * sizeof(double));
    for (int i = 0; i <= localN; ++i) {
        (*outNodes)[i] = globalNodes[startCell + i];
    }
}

static int compute_start(int globalN, int nProcs, int coord) {
    int base = globalN / nProcs;
    int remainder = globalN % nProcs;
    int start = 0;
    for (int p = 0; p < coord; ++p) {
        start += base + (p < remainder ? 1 : 0);
    }
    return start;
}

void mpi_context_create(MPIContext* ctx,
                         int globalNx, int globalNy, int globalNz,
                         const double* xNodes, int nxNodes,
                         const double* yNodes, int nyNodes,
                         const double* zNodes, int nzNodes,
                         int dim,
                         const int periods[3],
                         const int procsPerDir[3])
{
    std::memset(ctx, 0, sizeof(MPIContext));
    ctx->cartComm = MPI_COMM_NULL;
    ctx->dims[0] = procsPerDir[0];
    ctx->dims[1] = procsPerDir[1];
    ctx->dims[2] = procsPerDir[2];

    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    /* For inactive dimensions, force 1 process */
    if (dim < 2) ctx->dims[1] = 1;
    if (dim < 3) ctx->dims[2] = 1;

    /* Let MPI fill in zeros with automatic decomposition */
    if (dim == 1) {
        if (ctx->dims[0] == 0) ctx->dims[0] = worldSize;
    } else if (dim == 2) {
        int autoDims = 0;
        for (int d = 0; d < 2; ++d) {
            if (ctx->dims[d] == 0) autoDims++;
        }
        if (autoDims > 0) {
            int activeDims[2] = {ctx->dims[0], ctx->dims[1]};
            MPI_Dims_create(worldSize, 2, activeDims);
            ctx->dims[0] = activeDims[0];
            ctx->dims[1] = activeDims[1];
        }
    } else {
        MPI_Dims_create(worldSize, 3, ctx->dims);
    }

    /* Verify the product matches worldSize */
    int product = ctx->dims[0] * ctx->dims[1] * ctx->dims[2];
    if (product != worldSize) {
        throw std::runtime_error(
            "mpi_context_create: dims product (" + std::to_string(product) +
            ") != worldSize (" + std::to_string(worldSize) + ")");
    }

    /* Create Cartesian communicator (allow reordering for topology optimization) */
    int mpiPeriods[3] = {periods[0], periods[1], periods[2]};
    MPI_Cart_create(MPI_COMM_WORLD, 3, ctx->dims, mpiPeriods, 1, &ctx->cartComm);

    MPI_Comm_rank(ctx->cartComm, &ctx->rank);
    MPI_Comm_size(ctx->cartComm, &ctx->size);
    MPI_Cart_coords(ctx->cartComm, ctx->rank, 3, ctx->coords);

    /* Get neighbors via MPI_Cart_shift */
    MPI_Cart_shift(ctx->cartComm, 0, 1, &ctx->neighbors[0], &ctx->neighbors[1]);
    MPI_Cart_shift(ctx->cartComm, 1, 1, &ctx->neighbors[2], &ctx->neighbors[3]);
    MPI_Cart_shift(ctx->cartComm, 2, 1, &ctx->neighbors[4], &ctx->neighbors[5]);

    /* Slice global node arrays into local pieces */
    slice_nodes(xNodes, globalNx, ctx->dims[0], ctx->coords[0],
                &ctx->localXNodes, &ctx->localXNodesSize);
    slice_nodes(yNodes, globalNy, ctx->dims[1], ctx->coords[1],
                &ctx->localYNodes, &ctx->localYNodesSize);
    slice_nodes(zNodes, globalNz, ctx->dims[2], ctx->coords[2],
                &ctx->localZNodes, &ctx->localZNodesSize);

    ctx->localNx = ctx->localXNodesSize - 1;
    ctx->localNy = ctx->localYNodesSize - 1;
    ctx->localNz = ctx->localZNodesSize - 1;

    /* Compute global cell extent for VTK output */
    int i0 = compute_start(globalNx, ctx->dims[0], ctx->coords[0]);
    int j0 = compute_start(globalNy, ctx->dims[1], ctx->coords[1]);
    int k0 = compute_start(globalNz, ctx->dims[2], ctx->coords[2]);
    ctx->localExtent[0] = i0;
    ctx->localExtent[1] = i0 + ctx->localNx;
    ctx->localExtent[2] = j0;
    ctx->localExtent[3] = j0 + ctx->localNy;
    ctx->localExtent[4] = k0;
    ctx->localExtent[5] = k0 + ctx->localNz;
}

void mpi_context_free(MPIContext* ctx) {
    if (ctx->cartComm != MPI_COMM_NULL && ctx->cartComm != MPI_COMM_WORLD) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Comm_free(&ctx->cartComm);
        }
    }
    ctx->cartComm = MPI_COMM_NULL;

    std::free(ctx->localXNodes);
    ctx->localXNodes = NULL;
    ctx->localXNodesSize = 0;

    std::free(ctx->localYNodes);
    ctx->localYNodes = NULL;
    ctx->localYNodesSize = 0;

    std::free(ctx->localZNodes);
    ctx->localZNodes = NULL;
    ctx->localZNodesSize = 0;
}

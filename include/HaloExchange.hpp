#ifndef HALO_EXCHANGE_HPP
#define HALO_EXCHANGE_HPP

#include "MPIContext.hpp"
#include <stddef.h>

#include "SolutionState.hpp"

struct RectilinearMesh;

struct HaloExchange {
    const struct MPIContext* mpi;
    int nGhost;
    int nx, ny, nz;
    int ngx, ngy, ngz;
    int dim;
    /* Cached pointer to mesh for index computations */
    const struct RectilinearMesh* mesh;
    /* Per-face send/receive buffers */
    double* sendBuf[6];
    double* recvBuf[6];
    size_t sendBufCapacity[6];
    size_t recvBufCapacity[6];
};

void halo_init(struct HaloExchange* h,
               const struct MPIContext* mpi,
               const struct RectilinearMesh* mesh);

void halo_free(struct HaloExchange* h);

/* Exchange ghost cells for state variables in all directions (x, then y, then z). */
void halo_exchange_state(struct HaloExchange* h,
                         struct SolutionState* state,
                         enum VarSet varSet);

/* Exchange ghost cells for a single scalar field in all directions. */
void halo_exchange_scalar(struct HaloExchange* h,
                          double* field);

/* Direction-specific exchange for state variables (direction: 0=x, 1=y, 2=z). */
void halo_exchange_state_direction(struct HaloExchange* h,
                                   struct SolutionState* state,
                                   enum VarSet varSet,
                                   int direction);

/* Direction-specific exchange for a single scalar field. */
void halo_exchange_scalar_direction(struct HaloExchange* h,
                                    double* field,
                                    int direction);

/* GPU-offload variant: packs interior-slab cells into sendBuf via a target
 * kernel (sendBuf is device-resident), transfers only the packed buffer
 * across PCIe, runs MPI on the host, transfers recvBuf back to the device,
 * and unpacks into ghost slabs with another target kernel.  The full
 * SolutionState is never copied — only ~one face slab per direction.
 *
 * Supports VARSET_PRIM and VARSET_CONS at any nPhases. */
void halo_exchange_state_direction_device(struct HaloExchange* h,
                                          struct SolutionState* state,
                                          enum VarSet varSet,
                                          int direction);

/* Single-scalar device halo exchange.  `field` must be a device-resident
 * pointer.  Pack/unpack run as omp target kernels; only the packed face
 * slab crosses PCIe. */
void halo_exchange_scalar_direction_device(struct HaloExchange* h,
                                           double* field,
                                           int direction);

#endif /* HALO_EXCHANGE_HPP */

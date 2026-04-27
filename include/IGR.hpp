#ifndef IGR_HPP
#define IGR_HPP

#include "SimulationConfig.hpp"
#include <stddef.h>

struct RectilinearMesh;
struct SolutionState;
struct HaloExchange;

/* Velocity gradient tensor (3x3) */
typedef double GradientTensor[3][3];

/* IGRParams is defined in SimulationConfig.hpp */

/* Compute alpha from mesh spacing */
double igr_compute_alpha(const IGRParams* params, double dx);

/* Compute the RHS of the elliptic equation from velocity gradients
   RHS = alpha * [tr(nabla u)^2 + tr^2(nabla u)] */
double igr_compute_rhs(const SimulationConfig* config,
                       const GradientTensor gradU,
                       double alpha);

/* Solve for entropic pressure sigma over the entire mesh.
   Uses Gauss-Seidel iteration with warm start. */
void igr_solve_entropic_pressure(const SimulationConfig* config,
                                 const IGRParams* params,
                                 const struct RectilinearMesh* mesh,
                                 struct SolutionState* state,
                                 const GradientTensor* gradU);

/* MPI-aware variant with halo exchange */
void igr_solve_entropic_pressure_mpi(const SimulationConfig* config,
                                     const IGRParams* params,
                                     const struct RectilinearMesh* mesh,
                                     struct SolutionState* state,
                                     const GradientTensor* gradU,
                                     struct HaloExchange* halo);

/* GPU-offload variant.  state->rho, state->sigma, state->aux and gradU must
 * be device-resident.  Uses Jacobi iteration (not Gauss-Seidel) so cells can
 * be updated in parallel; convergence rate differs from the host solver but
 * the fixed-point is the same.  state->aux is used as the ping-pong buffer. */
void igr_solve_entropic_pressure_mpi_device(const SimulationConfig* config,
                                            const IGRParams* params,
                                            const struct RectilinearMesh* mesh,
                                            struct SolutionState* state,
                                            const GradientTensor* gradU,
                                            struct HaloExchange* halo);

/* Compute the velocity gradient tensor at every physical cell on device.
 * Reads velU/velV/velW from device memory, writes gradU (flat 9*totalCells
 * array of doubles, interpreted as GradientTensor per cell). */
void igr_compute_velocity_gradients_device(
    const struct SimulationConfig* config,
    const struct RectilinearMesh* mesh,
    const struct SolutionState* state,
    double* gradU);

/* Compute velocity gradient tensor from cell-centered velocities
   using central differences */
void igr_compute_velocity_gradient(
    const double u_xm[3],  /* u at x-1 */
    const double u_xp[3],  /* u at x+1 */
    const double u_ym[3],  /* u at y-1 */
    const double u_yp[3],  /* u at y+1 */
    const double u_zm[3],  /* u at z-1 */
    const double u_zp[3],  /* u at z+1 */
    double dx, double dy, double dz,
    int dim,
    GradientTensor grad);

#endif /* IGR_HPP */

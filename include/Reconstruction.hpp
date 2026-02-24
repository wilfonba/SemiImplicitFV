#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include "SimulationConfig.hpp"
#include "State.hpp"
#include <stddef.h>

struct RectilinearMesh;
struct SolutionState;

struct ReconstructorData {
    enum ReconstructionOrder order;
    double wenoEps;
    double gamma;
    double pInf;
    int dim;
    int nx, ny, nz;
    size_t numXFaces, numYFaces, numZFaces;
    PrimitiveState* xLeft;
    PrimitiveState* xRight;
    PrimitiveState* yLeft;
    PrimitiveState* yRight;
    PrimitiveState* zLeft;
    PrimitiveState* zRight;
};

void reconstructor_init(struct ReconstructorData* r,
                        enum ReconstructionOrder order,
                        double wenoEps, double gamma, double pInf);

void reconstructor_allocate(struct ReconstructorData* r,
                            const struct RectilinearMesh* mesh);

void reconstructor_free(struct ReconstructorData* r);

void reconstruct(struct ReconstructorData* r,
                 const struct SimulationConfig* config,
                 const struct RectilinearMesh* mesh,
                 const struct SolutionState* state);

int reconstructor_required_ghost_cells(const struct ReconstructorData* r);

/* Inline face index functions */
static inline size_t x_face_index(const struct ReconstructorData* r, int i, int j, int k) {
    return (size_t)(i + (r->nx + 1) * (j + r->ny * k));
}

static inline size_t y_face_index(const struct ReconstructorData* r, int i, int j, int k) {
    return (size_t)(i + r->nx * (j + (r->ny + 1) * k));
}

static inline size_t z_face_index(const struct ReconstructorData* r, int i, int j, int k) {
    return (size_t)(i + r->nx * (j + r->ny * k));
}

/* Inline face state accessors */
static inline const PrimitiveState* x_face_left(const struct ReconstructorData* r, size_t f) {
    return &r->xLeft[f];
}

static inline const PrimitiveState* x_face_right(const struct ReconstructorData* r, size_t f) {
    return &r->xRight[f];
}

static inline const PrimitiveState* y_face_left(const struct ReconstructorData* r, size_t f) {
    return &r->yLeft[f];
}

static inline const PrimitiveState* y_face_right(const struct ReconstructorData* r, size_t f) {
    return &r->yRight[f];
}

static inline const PrimitiveState* z_face_left(const struct ReconstructorData* r, size_t f) {
    return &r->zLeft[f];
}

static inline const PrimitiveState* z_face_right(const struct ReconstructorData* r, size_t f) {
    return &r->zRight[f];
}

#endif /* RECONSTRUCTION_HPP */

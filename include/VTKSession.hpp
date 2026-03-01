#ifndef VTK_SESSION_HPP
#define VTK_SESSION_HPP

#include "VTKWriter.hpp"

struct Runtime;
struct RectilinearMesh;
struct SolutionState;
struct SimulationConfig;

struct VTKSession {
    char baseName[256];
    char dir[256];
    enum VTKFormat format;
    int fileNum;
    int localExtent[6];
    struct Runtime* rt;
    const struct RectilinearMesh* mesh;
    const struct SimulationConfig* config;
};

void vtk_session_init(struct VTKSession* s,
                      struct Runtime* rt,
                      const char* baseName,
                      const struct RectilinearMesh* mesh,
                      const struct SimulationConfig* config,
                      const char* dir,
                      enum VTKFormat format,
                      int restarting);

/* Write current state at the given time. Collective in MPI mode. */
void vtk_session_write(struct VTKSession* s,
                       const struct SolutionState* state,
                       double time);

/* Close the PVD time-series file. */
void vtk_session_finalize(struct VTKSession* s);

#endif /* VTK_SESSION_HPP */

#ifndef INPUT_PARSER_HPP
#define INPUT_PARSER_HPP

#include "SimulationConfig.hpp"
#include "RectilinearMesh.hpp"
#include "VTKWriter.hpp"
#include "ICPatch.hpp"
#include "RiemannSolver.hpp"
#include "PressureSolver.hpp"

struct MeshParams {
    int nx, ny, nz;
    double xMin, xMax;
    double yMin, yMax;
    double zMin, zMax;
};

struct EOSParams {
    char type[64];
    double gamma;
    double R;
    double pInf;
};

struct TimeLoopInputParams {
    double endTime;
    double outputInterval;
    int printInterval;
    int checkNaN;
};

struct OutputParams {
    char baseName[64];
    char directory[64];
    enum VTKFormat format;
};

struct SmoothingParams {
    int iterations;
};

struct RestartParams {
    char file[256];
    int checkpoint;
};

struct InputData {
    SimulationConfig config;
    MeshParams meshParams;
    EOSParams eosParams;
    enum RiemannSolverType riemannSolverType;
    enum PressureSolverType pressureSolverType;
    enum BoundaryCondition bc[6];
    TimeLoopInputParams timeLoopParams;
    OutputParams outputParams;
    SmoothingParams smoothingParams;
    RestartParams restartParams;
    ICState defaultState;
    ICPatch* patches;
    int nPatches;
};

InputData input_data_defaults(void);
void input_data_free(InputData* d);
InputData parse_input_file(const char* filename);
InputData parse_input_string(const char* jsonStr);

#endif /* INPUT_PARSER_HPP */

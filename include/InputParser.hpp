#ifndef INPUT_PARSER_HPP
#define INPUT_PARSER_HPP

#include "SimulationConfig.hpp"
#include "RectilinearMesh.hpp"
#include "ICPatch.hpp"

#include <array>
#include <string>
#include <vector>

namespace SemiImplicitFV {

struct MeshParams {
    int nx = 100;  double xMin = 0.0; double xMax = 1.0;
    int ny = 1;    double yMin = 0.0; double yMax = 1.0;
    int nz = 1;    double zMin = 0.0; double zMax = 1.0;
};

struct EOSParams {
    std::string type = "IdealGas";  // "IdealGas" or "StiffenedGas"
    double gamma = 1.4;
    double R     = 287.0;
    double pInf  = 0.0;
};

struct BCParams {
    std::array<BoundaryCondition, 6> bc = {
        BoundaryCondition::Outflow, BoundaryCondition::Outflow,
        BoundaryCondition::Outflow, BoundaryCondition::Outflow,
        BoundaryCondition::Outflow, BoundaryCondition::Outflow
    };
};

struct TimeLoopInputParams {
    double endTime        = 1.0;
    double outputInterval = 0.01;
    int    printInterval  = 1;
    bool   checkNaN       = true;
};

struct OutputParams {
    std::string baseName  = "output";
    std::string directory = "VTK";
};

struct SmoothingParams {
    int iterations = 0;
};

struct InputData {
    SimulationConfig config;
    MeshParams meshParams;
    EOSParams eosParams;
    std::string riemannSolverType = "HLLC";
    std::string pressureSolverType = "GaussSeidel";
    BCParams bcParams;
    TimeLoopInputParams timeLoopParams;
    OutputParams outputParams;
    SmoothingParams smoothingParams;
    ICState defaultState;
    std::vector<ICPatch> patches;
};

namespace InputParser {

/// Parse a JSON/JSONC input file and return an InputData struct.
/// Throws std::runtime_error on parse failure.
InputData parseFile(const std::string& filename);

/// Parse a JSON string and return an InputData struct.
InputData parseString(const std::string& jsonStr);

} // namespace InputParser

} // namespace SemiImplicitFV

#endif // INPUT_PARSER_HPP

#include "InputParser.hpp"
#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

using json = nlohmann::json;

namespace SemiImplicitFV {
namespace InputParser {

// -----------------------------------------------------------------
//  Helper: strip C/C++ style comments from JSONC
// -----------------------------------------------------------------
static std::string stripComments(const std::string& input) {
    std::string result;
    result.reserve(input.size());
    std::size_t i = 0;
    bool inString = false;

    while (i < input.size()) {
        if (inString) {
            if (input[i] == '\\' && i + 1 < input.size()) {
                result += input[i]; result += input[i+1]; i += 2;
            } else {
                if (input[i] == '"') inString = false;
                result += input[i]; ++i;
            }
        } else {
            if (input[i] == '"') {
                inString = true;
                result += input[i]; ++i;
            } else if (input[i] == '/' && i + 1 < input.size()) {
                if (input[i+1] == '/') {
                    // Line comment: skip to end of line
                    i += 2;
                    while (i < input.size() && input[i] != '\n') ++i;
                } else if (input[i+1] == '*') {
                    // Block comment: skip to */
                    i += 2;
                    while (i + 1 < input.size() && !(input[i] == '*' && input[i+1] == '/')) ++i;
                    if (i + 1 < input.size()) i += 2;
                } else {
                    result += input[i]; ++i;
                }
            } else {
                result += input[i]; ++i;
            }
        }
    }
    return result;
}

// -----------------------------------------------------------------
//  Enum parsers
// -----------------------------------------------------------------
static ReconstructionOrder parseReconOrder(const std::string& s) {
    if (s == "WENO1")   return ReconstructionOrder::WENO1;
    if (s == "WENO3")   return ReconstructionOrder::WENO3;
    if (s == "WENO5")   return ReconstructionOrder::WENO5;
    if (s == "UPWIND1") return ReconstructionOrder::UPWIND1;
    if (s == "UPWIND3") return ReconstructionOrder::UPWIND3;
    if (s == "UPWIND5") return ReconstructionOrder::UPWIND5;
    throw std::runtime_error("Unknown reconstruction order: '" + s + "'");
}

static BoundaryCondition parseBC(const std::string& s) {
    if (s == "Symmetry")   return BoundaryCondition::Symmetry;
    if (s == "Outflow")    return BoundaryCondition::Outflow;
    if (s == "Periodic")   return BoundaryCondition::Periodic;
    if (s == "SlipWall")   return BoundaryCondition::SlipWall;
    if (s == "NoSlipWall") return BoundaryCondition::NoSlipWall;
    throw std::runtime_error("Unknown boundary condition: '" + s + "'");
}

// -----------------------------------------------------------------
//  Helper: read a 3-element array with default
// -----------------------------------------------------------------
static std::array<double,3> readArray3(const json& j, const std::array<double,3>& def = {0,0,0}) {
    if (!j.is_array() || j.size() < 3) return def;
    return {j[0].get<double>(), j[1].get<double>(), j[2].get<double>()};
}

// -----------------------------------------------------------------
//  Helper: safe json access with default
// -----------------------------------------------------------------
template <typename T>
static T get(const json& j, const std::string& key, const T& def) {
    if (j.contains(key)) return j[key].get<T>();
    return def;
}

// -----------------------------------------------------------------
//  Parse geometry from JSON
// -----------------------------------------------------------------
static std::unique_ptr<ICGeometry> parseGeometry(const json& j) {
    std::string type = get<std::string>(j, "type", "box");

    if (type == "box") {
        auto lo = readArray3(j.value("min", json::array({0,0,0})));
        auto hi = readArray3(j.value("max", json::array({1,1,1})));
        return std::make_unique<BoxGeometry>(lo, hi);
    }
    if (type == "sphere") {
        auto center = readArray3(j.value("center", json::array({0,0,0})));
        double radius = get<double>(j, "radius", 1.0);
        return std::make_unique<SphereGeometry>(center, radius);
    }
    if (type == "plane") {
        auto point  = readArray3(j.value("point", json::array({0,0,0})));
        auto normal = readArray3(j.value("normal", json::array({1,0,0})));
        return std::make_unique<PlaneGeometry>(point, normal);
    }
    if (type == "analytic") {
        std::unique_ptr<ICGeometry> region;
        if (j.contains("region")) {
            region = parseGeometry(j["region"]);
        }
        return std::make_unique<AnalyticGeometry>(std::move(region));
    }

    throw std::runtime_error("Unknown geometry type: '" + type + "'");
}

// -----------------------------------------------------------------
//  Parse ICState from JSON
// -----------------------------------------------------------------
static ICState parseICState(const json& j, const ICState& inherit) {
    ICState st = inherit;
    if (j.contains("rho"))   st.rho   = j["rho"].get<double>();
    if (j.contains("u"))     st.u     = j["u"].get<double>();
    if (j.contains("v"))     st.v     = j["v"].get<double>();
    if (j.contains("w"))     st.w     = j["w"].get<double>();
    if (j.contains("p"))     st.p     = j["p"].get<double>();
    if (j.contains("sigma")) st.sigma = j["sigma"].get<double>();

    if (j.contains("alpha")) {
        st.alpha.clear();
        for (const auto& a : j["alpha"]) st.alpha.push_back(a.get<double>());
    }
    if (j.contains("alphaRho")) {
        st.alphaRho.clear();
        for (const auto& a : j["alphaRho"]) st.alphaRho.push_back(a.get<double>());
    }
    return st;
}

// -----------------------------------------------------------------
//  Main parse function
// -----------------------------------------------------------------
static InputData parseJson(const json& root) {
    InputData data;

    // --- config ---
    if (root.contains("config")) {
        const auto& c = root["config"];
        auto& cfg = data.config;

        cfg.dim       = get<int>(c, "dim", cfg.dim);
        cfg.nGhost    = get<int>(c, "nGhost", cfg.nGhost);
        cfg.RKOrder   = get<int>(c, "RKOrder", cfg.RKOrder);
        cfg.useIGR    = get<bool>(c, "useIGR", cfg.useIGR);
        cfg.semiImplicit = get<bool>(c, "semiImplicit", cfg.semiImplicit);
        cfg.wenoEps   = get<double>(c, "wenoEps", cfg.wenoEps);

        if (c.contains("reconOrder")) {
            cfg.reconOrder = parseReconOrder(c["reconOrder"].get<std::string>());
        }

        if (c.contains("explicitParams")) {
            const auto& ep = c["explicitParams"];
            cfg.explicitParams.cfl     = get<double>(ep, "cfl", cfg.explicitParams.cfl);
            cfg.explicitParams.constDt = get<double>(ep, "constDt", cfg.explicitParams.constDt);
            cfg.explicitParams.maxDt   = get<double>(ep, "maxDt", cfg.explicitParams.maxDt);
            cfg.explicitParams.minDt   = get<double>(ep, "minDt", cfg.explicitParams.minDt);
        }

        if (c.contains("semiImplicitParams")) {
            const auto& sp = c["semiImplicitParams"];
            cfg.semiImplicitParams.cfl              = get<double>(sp, "cfl", cfg.semiImplicitParams.cfl);
            cfg.semiImplicitParams.maxDt             = get<double>(sp, "maxDt", cfg.semiImplicitParams.maxDt);
            cfg.semiImplicitParams.minDt             = get<double>(sp, "minDt", cfg.semiImplicitParams.minDt);
            cfg.semiImplicitParams.maxPressureIters  = get<int>(sp, "maxPressureIters", cfg.semiImplicitParams.maxPressureIters);
            cfg.semiImplicitParams.pressureTol       = get<double>(sp, "pressureTol", cfg.semiImplicitParams.pressureTol);
        }

        if (c.contains("igrParams")) {
            const auto& ip = c["igrParams"];
            cfg.igrParams.alphaCoeff       = get<double>(ip, "alphaCoeff", cfg.igrParams.alphaCoeff);
            cfg.igrParams.IGRIters         = get<int>(ip, "IGRIters", cfg.igrParams.IGRIters);
            cfg.igrParams.IGRWarmStartIters = get<int>(ip, "IGRWarmStartIters", cfg.igrParams.IGRWarmStartIters);
        }

        if (c.contains("multiPhaseParams")) {
            const auto& mp = c["multiPhaseParams"];
            cfg.multiPhaseParams.nPhases  = get<int>(mp, "nPhases", cfg.multiPhaseParams.nPhases);
            cfg.multiPhaseParams.alphaMin = get<double>(mp, "alphaMin", cfg.multiPhaseParams.alphaMin);
            if (mp.contains("phases")) {
                cfg.multiPhaseParams.phases.clear();
                for (const auto& ph : mp["phases"]) {
                    PhaseEOS peos;
                    peos.gamma = get<double>(ph, "gamma", 1.4);
                    peos.pInf  = get<double>(ph, "pInf", 0.0);
                    cfg.multiPhaseParams.phases.push_back(peos);
                }
            }
        }

        if (c.contains("bodyForceParams")) {
            const auto& bf = c["bodyForceParams"];
            if (bf.contains("a")) cfg.bodyForceParams.a = readArray3(bf["a"]);
            if (bf.contains("b")) cfg.bodyForceParams.b = readArray3(bf["b"]);
            if (bf.contains("c")) cfg.bodyForceParams.c = readArray3(bf["c"]);
            if (bf.contains("d")) cfg.bodyForceParams.d = readArray3(bf["d"]);
        }

        if (c.contains("viscousParams")) {
            const auto& vp = c["viscousParams"];
            cfg.viscousParams.mu = get<double>(vp, "mu", cfg.viscousParams.mu);
            if (vp.contains("phaseMu")) {
                cfg.viscousParams.phaseMu.clear();
                for (const auto& m : vp["phaseMu"])
                    cfg.viscousParams.phaseMu.push_back(m.get<double>());
            }
        }

        if (c.contains("surfaceTensionParams")) {
            const auto& st = c["surfaceTensionParams"];
            cfg.surfaceTensionParams.sigma        = get<double>(st, "sigma", cfg.surfaceTensionParams.sigma);
            cfg.surfaceTensionParams.epsGradAlpha = get<double>(st, "epsGradAlpha", cfg.surfaceTensionParams.epsGradAlpha);
        }
    }

    // --- EOS ---
    if (root.contains("eos")) {
        const auto& e = root["eos"];
        data.eosParams.type  = get<std::string>(e, "type", data.eosParams.type);
        data.eosParams.gamma = get<double>(e, "gamma", data.eosParams.gamma);
        data.eosParams.R     = get<double>(e, "R", data.eosParams.R);
        data.eosParams.pInf  = get<double>(e, "pInf", data.eosParams.pInf);
    }

    // --- Riemann solver ---
    data.riemannSolverType = get<std::string>(root, "riemannSolver", data.riemannSolverType);

    // --- Pressure solver ---
    data.pressureSolverType = get<std::string>(root, "pressureSolver", data.pressureSolverType);

    // --- Mesh ---
    if (root.contains("mesh")) {
        const auto& m = root["mesh"];
        data.meshParams.nx   = get<int>(m, "nx", data.meshParams.nx);
        data.meshParams.xMin = get<double>(m, "xMin", data.meshParams.xMin);
        data.meshParams.xMax = get<double>(m, "xMax", data.meshParams.xMax);
        data.meshParams.ny   = get<int>(m, "ny", data.meshParams.ny);
        data.meshParams.yMin = get<double>(m, "yMin", data.meshParams.yMin);
        data.meshParams.yMax = get<double>(m, "yMax", data.meshParams.yMax);
        data.meshParams.nz   = get<int>(m, "nz", data.meshParams.nz);
        data.meshParams.zMin = get<double>(m, "zMin", data.meshParams.zMin);
        data.meshParams.zMax = get<double>(m, "zMax", data.meshParams.zMax);
    }

    // --- Boundary conditions ---
    if (root.contains("boundaryConditions")) {
        const auto& bc = root["boundaryConditions"];
        const std::array<std::string, 6> faceNames = {"xLow", "xHigh", "yLow", "yHigh", "zLow", "zHigh"};
        for (int f = 0; f < 6; ++f) {
            if (bc.contains(faceNames[f])) {
                data.bcParams.bc[f] = parseBC(bc[faceNames[f]].get<std::string>());
            }
        }
    }

    // --- Time loop ---
    if (root.contains("timeLoop")) {
        const auto& t = root["timeLoop"];
        data.timeLoopParams.endTime        = get<double>(t, "endTime", data.timeLoopParams.endTime);
        data.timeLoopParams.outputInterval = get<double>(t, "outputInterval", data.timeLoopParams.outputInterval);
        data.timeLoopParams.printInterval  = get<int>(t, "printInterval", data.timeLoopParams.printInterval);
        data.timeLoopParams.checkNaN       = get<bool>(t, "checkNaN", data.timeLoopParams.checkNaN);
    }

    // --- Output ---
    if (root.contains("output")) {
        const auto& o = root["output"];
        data.outputParams.baseName  = get<std::string>(o, "baseName", data.outputParams.baseName);
        data.outputParams.directory = get<std::string>(o, "directory", data.outputParams.directory);
    }

    // --- Smoothing ---
    if (root.contains("smoothing")) {
        const auto& s = root["smoothing"];
        data.smoothingParams.iterations = get<int>(s, "iterations", data.smoothingParams.iterations);
    }

    // --- Initial conditions ---
    if (root.contains("initialConditions")) {
        const auto& ic = root["initialConditions"];

        // Default state
        if (ic.contains("default")) {
            data.defaultState = parseICState(ic["default"], data.defaultState);
        }

        // Patches
        if (ic.contains("patches")) {
            for (const auto& pj : ic["patches"]) {
                ICPatch patch;

                // Geometry
                if (pj.contains("geometry")) {
                    patch.geometry = parseGeometry(pj["geometry"]);
                }

                // State (inherits from default)
                if (pj.contains("state")) {
                    patch.state = parseICState(pj["state"], data.defaultState);
                } else {
                    patch.state = data.defaultState;
                }

                // Expressions (for analytic patches)
                if (pj.contains("expressions")) {
                    for (const auto& [key, val] : pj["expressions"].items()) {
                        patch.expressions[key] = val.get<std::string>();
                    }
                }

                data.patches.push_back(std::move(patch));
            }
        }
    }

    return data;
}

InputData parseFile(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    std::ostringstream oss;
    oss << ifs.rdbuf();
    return parseString(oss.str());
}

InputData parseString(const std::string& jsonStr) {
    std::string cleaned = stripComments(jsonStr);

    json root;
    try {
        root = json::parse(cleaned);
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("JSON parse error: ") + e.what());
    }

    return parseJson(root);
}

} // namespace InputParser
} // namespace SemiImplicitFV

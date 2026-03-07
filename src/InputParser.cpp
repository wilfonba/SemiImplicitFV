#include "InputParser.hpp"
#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <cstdlib>

using json = nlohmann::json;

/* -----------------------------------------------------------------
   Helper: strip C/C++ style comments from JSONC
   ----------------------------------------------------------------- */
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
                    /* Line comment: skip to end of line */
                    i += 2;
                    while (i < input.size() && input[i] != '\n') ++i;
                } else if (input[i+1] == '*') {
                    /* Block comment: skip to */
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

/* -----------------------------------------------------------------
   Enum parsers
   ----------------------------------------------------------------- */
static enum ReconstructionOrder parseReconOrder(const std::string& s) {
    if (s == "WENO1")   return WENO1;
    if (s == "WENO3")   return WENO3;
    if (s == "WENO5")   return WENO5;
    if (s == "UPWIND1") return UPWIND1;
    if (s == "UPWIND3") return UPWIND3;
    if (s == "UPWIND5") return UPWIND5;
    throw std::runtime_error("Unknown reconstruction order: '" + s + "'");
}

static enum BoundaryCondition parseBC(const std::string& s) {
    if (s == "Symmetry")   return BC_SYMMETRY;
    if (s == "Outflow")    return BC_OUTFLOW;
    if (s == "Periodic")   return BC_PERIODIC;
    if (s == "SlipWall")   return BC_SLIP_WALL;
    if (s == "NoSlipWall") return BC_NO_SLIP_WALL;
    throw std::runtime_error("Unknown boundary condition: '" + s + "'");
}

static enum RiemannSolverType parseRiemannSolver(const std::string& s) {
    if (s == "LF")      return RS_LF;
    if (s == "Rusanov")  return RS_RUSANOV;
    if (s == "HLLC")     return RS_HLLC;
    throw std::runtime_error("Unknown Riemann solver type: '" + s + "'");
}

static enum PressureSolverType parsePressureSolver(const std::string& s) {
    if (s == "GaussSeidel") return PS_GAUSS_SEIDEL;
    if (s == "Jacobi")      return PS_JACOBI;
    if (s == "PETSc")       return PS_PETSC;
    throw std::runtime_error("Unknown pressure solver type: '" + s + "'");
}

/* -----------------------------------------------------------------
   Helper: read a 3-element array with default
   ----------------------------------------------------------------- */
static void readArray3(const json& j, double out[3], const double def[3]) {
    if (!j.is_array() || j.size() < 3) {
        out[0] = def[0]; out[1] = def[1]; out[2] = def[2];
        return;
    }
    out[0] = j[0].get<double>();
    out[1] = j[1].get<double>();
    out[2] = j[2].get<double>();
}

static void readArray3_default(const json& j, double out[3]) {
    double def[3] = {0.0, 0.0, 0.0};
    readArray3(j, out, def);
}

/* -----------------------------------------------------------------
   Helper: safe json access with default
   ----------------------------------------------------------------- */
template <typename T>
static T jget(const json& j, const std::string& key, const T& def) {
    if (j.contains(key)) return j[key].get<T>();
    return def;
}

/* -----------------------------------------------------------------
   Helper: safely copy string to fixed-size char buffer
   ----------------------------------------------------------------- */
static void strncpy_safe(char* dst, const char* src, size_t dstSize) {
    std::strncpy(dst, src, dstSize - 1);
    dst[dstSize - 1] = '\0';
}

/* -----------------------------------------------------------------
   Parse geometry from JSON into ICGeometry tagged union
   ----------------------------------------------------------------- */
static void parseGeometry(const json& j, ICGeometry* geom) {
    std::string type = jget<std::string>(j, "type", "box");

    if (type == "box") {
        geom->type = IC_GEOM_BOX;
        double defLo[3] = {0, 0, 0};
        double defHi[3] = {1, 1, 1};
        if (j.contains("min")) {
            readArray3(j["min"], geom->box.lo, defLo);
        } else {
            geom->box.lo[0] = defLo[0]; geom->box.lo[1] = defLo[1]; geom->box.lo[2] = defLo[2];
        }
        if (j.contains("max")) {
            readArray3(j["max"], geom->box.hi, defHi);
        } else {
            geom->box.hi[0] = defHi[0]; geom->box.hi[1] = defHi[1]; geom->box.hi[2] = defHi[2];
        }
        geom->subRegion = NULL;
    }
    else if (type == "sphere") {
        geom->type = IC_GEOM_SPHERE;
        double defCenter[3] = {0, 0, 0};
        if (j.contains("center")) {
            readArray3(j["center"], geom->sphere.center, defCenter);
        } else {
            geom->sphere.center[0] = 0; geom->sphere.center[1] = 0; geom->sphere.center[2] = 0;
        }
        geom->sphere.radius = jget<double>(j, "radius", 1.0);
        geom->subRegion = NULL;
    }
    else if (type == "plane") {
        geom->type = IC_GEOM_PLANE;
        double defPt[3] = {0, 0, 0};
        double defNorm[3] = {1, 0, 0};
        if (j.contains("point")) {
            readArray3(j["point"], geom->plane.point, defPt);
        } else {
            geom->plane.point[0] = 0; geom->plane.point[1] = 0; geom->plane.point[2] = 0;
        }
        if (j.contains("normal")) {
            readArray3(j["normal"], geom->plane.normal, defNorm);
        } else {
            geom->plane.normal[0] = 1; geom->plane.normal[1] = 0; geom->plane.normal[2] = 0;
        }
        geom->subRegion = NULL;
    }
    else if (type == "analytic") {
        geom->type = IC_GEOM_ANALYTIC;
        if (j.contains("region")) {
            geom->subRegion = (ICGeometry*)std::calloc(1, sizeof(ICGeometry));
            parseGeometry(j["region"], geom->subRegion);
        } else {
            geom->subRegion = NULL;
        }
    }
    else {
        throw std::runtime_error("Unknown geometry type: '" + type + "'");
    }
}

/* -----------------------------------------------------------------
   Parse ICState from JSON
   ----------------------------------------------------------------- */
static ICState parseICState(const json& j, const ICState* inherit) {
    ICState st = *inherit;
    if (j.contains("rho"))   st.rho   = j["rho"].get<double>();
    if (j.contains("u"))     st.u     = j["u"].get<double>();
    if (j.contains("v"))     st.v     = j["v"].get<double>();
    if (j.contains("w"))     st.w     = j["w"].get<double>();
    if (j.contains("p"))     st.p     = j["p"].get<double>();
    if (j.contains("sigma")) st.sigma = j["sigma"].get<double>();

    if (j.contains("alpha")) {
        st.nAlpha = 0;
        for (const auto& a : j["alpha"]) {
            if (st.nAlpha < MAX_PHASES) {
                st.alpha[st.nAlpha++] = a.get<double>();
            }
        }
    }
    if (j.contains("alphaRho")) {
        st.nAlphaRho = 0;
        for (const auto& a : j["alphaRho"]) {
            if (st.nAlphaRho < MAX_PHASES) {
                st.alphaRho[st.nAlphaRho++] = a.get<double>();
            }
        }
    }
    return st;
}

/* -----------------------------------------------------------------
   Main parse function
   ----------------------------------------------------------------- */
static InputData parseJson(const json& root) {
    InputData data = input_data_defaults();

    /* --- config --- */
    if (root.contains("config")) {
        const auto& c = root["config"];
        SimulationConfig* cfg = &data.config;

        cfg->dim       = jget<int>(c, "dim", cfg->dim);
        cfg->RKOrder   = jget<int>(c, "RKOrder", cfg->RKOrder);
        cfg->useIGR    = jget<bool>(c, "useIGR", cfg->useIGR != 0) ? 1 : 0;
        cfg->semiImplicit = jget<bool>(c, "semiImplicit", cfg->semiImplicit != 0) ? 1 : 0;
        cfg->wenoEps   = jget<double>(c, "wenoEps", cfg->wenoEps);

        if (c.contains("reconOrder")) {
            cfg->reconOrder = parseReconOrder(c["reconOrder"].get<std::string>());
        }

        if (c.contains("explicitParams")) {
            const auto& ep = c["explicitParams"];
            cfg->explicitParams.cfl     = jget<double>(ep, "cfl", cfg->explicitParams.cfl);
            cfg->explicitParams.constDt = jget<double>(ep, "constDt", cfg->explicitParams.constDt);
            cfg->explicitParams.maxDt   = jget<double>(ep, "maxDt", cfg->explicitParams.maxDt);
            cfg->explicitParams.minDt   = jget<double>(ep, "minDt", cfg->explicitParams.minDt);
        }

        if (c.contains("semiImplicitParams")) {
            const auto& sp = c["semiImplicitParams"];
            cfg->semiImplicitParams.cfl              = jget<double>(sp, "cfl", cfg->semiImplicitParams.cfl);
            cfg->semiImplicitParams.constDt           = jget<double>(sp, "constDt", cfg->semiImplicitParams.constDt);
            cfg->semiImplicitParams.maxDt             = jget<double>(sp, "maxDt", cfg->semiImplicitParams.maxDt);
            cfg->semiImplicitParams.minDt             = jget<double>(sp, "minDt", cfg->semiImplicitParams.minDt);
            cfg->semiImplicitParams.maxAcousticCFL    = jget<double>(sp, "maxAcousticCFL", cfg->semiImplicitParams.maxAcousticCFL);
            cfg->semiImplicitParams.maxPressureIters  = jget<int>(sp, "maxPressureIters", cfg->semiImplicitParams.maxPressureIters);
            cfg->semiImplicitParams.pressureTol       = jget<double>(sp, "pressureTol", cfg->semiImplicitParams.pressureTol);
            cfg->semiImplicitParams.singlePressureSolve = jget<bool>(sp, "singlePressureSolve", cfg->semiImplicitParams.singlePressureSolve != 0) ? 1 : 0;
        }

        if (c.contains("igrParams")) {
            const auto& ip = c["igrParams"];
            cfg->igrParams.alphaCoeff       = jget<double>(ip, "alphaCoeff", cfg->igrParams.alphaCoeff);
            cfg->igrParams.IGRIters         = jget<int>(ip, "IGRIters", cfg->igrParams.IGRIters);
            cfg->igrParams.IGRWarmStartIters = jget<int>(ip, "IGRWarmStartIters", cfg->igrParams.IGRWarmStartIters);
        }

        if (c.contains("multiPhaseParams")) {
            const auto& mp = c["multiPhaseParams"];
            cfg->multiPhaseParams.nPhases  = jget<int>(mp, "nPhases", cfg->multiPhaseParams.nPhases);
            cfg->multiPhaseParams.alphaMin = jget<double>(mp, "alphaMin", cfg->multiPhaseParams.alphaMin);
            if (mp.contains("phases")) {
                int idx = 0;
                for (const auto& ph : mp["phases"]) {
                    if (idx < MAX_PHASES) {
                        cfg->multiPhaseParams.phases[idx].gamma = jget<double>(ph, "gamma", 1.4);
                        cfg->multiPhaseParams.phases[idx].pInf  = jget<double>(ph, "pInf", 0.0);
                        idx++;
                    }
                }
            }
        }

        if (c.contains("bodyForceParams")) {
            const auto& bf = c["bodyForceParams"];
            if (bf.contains("a")) readArray3_default(bf["a"], cfg->bodyForceParams.a);
            if (bf.contains("b")) readArray3_default(bf["b"], cfg->bodyForceParams.b);
            if (bf.contains("c")) readArray3_default(bf["c"], cfg->bodyForceParams.c);
            if (bf.contains("d")) readArray3_default(bf["d"], cfg->bodyForceParams.d);
        }

        if (c.contains("viscousParams")) {
            const auto& vp = c["viscousParams"];
            cfg->viscousParams.mu = jget<double>(vp, "mu", cfg->viscousParams.mu);
            if (vp.contains("phaseMu")) {
                cfg->viscousParams.nPhaseMu = 0;
                for (const auto& m : vp["phaseMu"]) {
                    if (cfg->viscousParams.nPhaseMu < MAX_PHASES) {
                        cfg->viscousParams.phaseMu[cfg->viscousParams.nPhaseMu++] = m.get<double>();
                    }
                }
            }
        }

        if (c.contains("surfaceTensionParams")) {
            const auto& st = c["surfaceTensionParams"];
            cfg->surfaceTensionParams.sigma        = jget<double>(st, "sigma", cfg->surfaceTensionParams.sigma);
            cfg->surfaceTensionParams.epsGradAlpha = jget<double>(st, "epsGradAlpha", cfg->surfaceTensionParams.epsGradAlpha);
        }

        if (c.contains("mthincParams")) {
            const auto& mt = c["mthincParams"];
            cfg->mthincParams.enabled = jget<bool>(mt, "enabled", cfg->mthincParams.enabled != 0) ? 1 : 0;
            cfg->mthincParams.beta    = jget<double>(mt, "beta", cfg->mthincParams.beta);
        }
    }

    /* --- EOS --- */
    if (root.contains("eos")) {
        const auto& e = root["eos"];
        std::string typeStr = jget<std::string>(e, "type", std::string(data.eosParams.type));
        strncpy_safe(data.eosParams.type, typeStr.c_str(), sizeof(data.eosParams.type));
        data.eosParams.gamma = jget<double>(e, "gamma", data.eosParams.gamma);
        data.eosParams.R     = jget<double>(e, "R", data.eosParams.R);
        data.eosParams.pInf  = jget<double>(e, "pInf", data.eosParams.pInf);
    }

    /* --- Riemann solver --- */
    if (root.contains("riemannSolver")) {
        data.riemannSolverType = parseRiemannSolver(root["riemannSolver"].get<std::string>());
    }

    /* --- Pressure solver --- */
    if (root.contains("pressureSolver")) {
        data.pressureSolverType = parsePressureSolver(root["pressureSolver"].get<std::string>());
    }

    /* --- Mesh --- */
    if (root.contains("mesh")) {
        const auto& m = root["mesh"];
        data.meshParams.nx   = jget<int>(m, "nx", data.meshParams.nx);
        data.meshParams.xMin = jget<double>(m, "xMin", data.meshParams.xMin);
        data.meshParams.xMax = jget<double>(m, "xMax", data.meshParams.xMax);
        data.meshParams.ny   = jget<int>(m, "ny", data.meshParams.ny);
        data.meshParams.yMin = jget<double>(m, "yMin", data.meshParams.yMin);
        data.meshParams.yMax = jget<double>(m, "yMax", data.meshParams.yMax);
        data.meshParams.nz   = jget<int>(m, "nz", data.meshParams.nz);
        data.meshParams.zMin = jget<double>(m, "zMin", data.meshParams.zMin);
        data.meshParams.zMax = jget<double>(m, "zMax", data.meshParams.zMax);
    }

    /* --- Boundary conditions --- */
    if (root.contains("boundaryConditions")) {
        const auto& bc = root["boundaryConditions"];
        const char* faceNames[6] = {"xLow", "xHigh", "yLow", "yHigh", "zLow", "zHigh"};
        for (int f = 0; f < 6; ++f) {
            if (bc.contains(faceNames[f])) {
                data.bc[f] = parseBC(bc[faceNames[f]].get<std::string>());
            }
        }
    }

    /* --- Time loop --- */
    if (root.contains("timeLoop")) {
        const auto& t = root["timeLoop"];
        data.timeLoopParams.endTime        = jget<double>(t, "endTime", data.timeLoopParams.endTime);
        data.timeLoopParams.outputInterval = jget<double>(t, "outputInterval", data.timeLoopParams.outputInterval);
        data.timeLoopParams.printInterval  = jget<int>(t, "printInterval", data.timeLoopParams.printInterval);
        data.timeLoopParams.checkNaN       = jget<bool>(t, "checkNaN", data.timeLoopParams.checkNaN != 0) ? 1 : 0;
    }

    /* --- Output --- */
    if (root.contains("output")) {
        const auto& o = root["output"];
        std::string bn = jget<std::string>(o, "baseName", std::string(data.outputParams.baseName));
        strncpy_safe(data.outputParams.baseName, bn.c_str(), sizeof(data.outputParams.baseName));
        std::string dir = jget<std::string>(o, "directory", std::string(data.outputParams.directory));
        strncpy_safe(data.outputParams.directory, dir.c_str(), sizeof(data.outputParams.directory));
        std::string fmt = jget<std::string>(o, "format", "VTKText");
        if (fmt == "VTKRaw") {
            data.outputParams.format = VTK_RAW;
        } else if (fmt == "VTKText") {
            data.outputParams.format = VTK_TEXT;
        } else {
            throw std::runtime_error("Unknown output format: \"" + fmt
                                     + "\" (expected \"VTKText\" or \"VTKRaw\")");
        }
    }

    /* --- Smoothing --- */
    if (root.contains("smoothing")) {
        const auto& s = root["smoothing"];
        data.smoothingParams.iterations = jget<int>(s, "iterations", data.smoothingParams.iterations);
    }

    /* --- Restart / checkpoint --- */
    if (root.contains("restart")) {
        const auto& r = root["restart"];
        std::string f = jget<std::string>(r, "file", std::string(data.restartParams.file));
        strncpy_safe(data.restartParams.file, f.c_str(), sizeof(data.restartParams.file));
        data.restartParams.checkpoint = jget<bool>(r, "checkpoint", data.restartParams.checkpoint != 0) ? 1 : 0;
    }

    /* --- Initial conditions --- */
    if (root.contains("initialConditions")) {
        const auto& ic = root["initialConditions"];

        /* Default state */
        if (ic.contains("default")) {
            data.defaultState = parseICState(ic["default"], &data.defaultState);
        }

        /* Patches */
        if (ic.contains("patches")) {
            const auto& patchesArr = ic["patches"];
            int n = (int)patchesArr.size();
            if (n > 0) {
                data.patches = (ICPatch*)std::calloc((size_t)n, sizeof(ICPatch));
                data.nPatches = n;

                for (int pi = 0; pi < n; ++pi) {
                    const auto& pj = patchesArr[pi];
                    ICPatch* patch = &data.patches[pi];
                    ic_patch_init(patch);

                    /* Geometry */
                    if (pj.contains("geometry")) {
                        parseGeometry(pj["geometry"], &patch->geometry);
                    }

                    /* State (inherits from default) */
                    if (pj.contains("state")) {
                        patch->state = parseICState(pj["state"], &data.defaultState);
                    } else {
                        patch->state = data.defaultState;
                    }

                    /* Expressions (for analytic patches) */
                    if (pj.contains("expressions")) {
                        const auto& exprs = pj["expressions"];
                        int nExprs = (int)exprs.size();
                        if (nExprs > 0) {
                            patch->exprNames   = (char**)std::malloc(sizeof(char*) * (size_t)nExprs);
                            patch->exprStrings = (char**)std::malloc(sizeof(char*) * (size_t)nExprs);
                            patch->nExpressions = nExprs;
                            int ei = 0;
                            for (const auto& [key, val] : exprs.items()) {
                                patch->exprNames[ei]   = strdup(key.c_str());
                                patch->exprStrings[ei] = strdup(val.get<std::string>().c_str());
                                ei++;
                            }
                        }
                    }
                }
            }
        }
    }

    return data;
}

/* -----------------------------------------------------------------
   Public API
   ----------------------------------------------------------------- */

InputData input_data_defaults(void) {
    InputData d;
    std::memset(&d, 0, sizeof(d));

    d.config = config_defaults();

    d.meshParams.nx = 100;  d.meshParams.xMin = 0.0; d.meshParams.xMax = 1.0;
    d.meshParams.ny = 1;    d.meshParams.yMin = 0.0; d.meshParams.yMax = 1.0;
    d.meshParams.nz = 1;    d.meshParams.zMin = 0.0; d.meshParams.zMax = 1.0;

    strncpy_safe(d.eosParams.type, "IdealGas", sizeof(d.eosParams.type));
    d.eosParams.gamma = 1.4;
    d.eosParams.R     = 287.0;
    d.eosParams.pInf  = 0.0;

    d.riemannSolverType = RS_HLLC;
    d.pressureSolverType = PS_GAUSS_SEIDEL;

    for (int f = 0; f < 6; ++f) {
        d.bc[f] = BC_OUTFLOW;
    }

    d.timeLoopParams.endTime        = 1.0;
    d.timeLoopParams.outputInterval = 0.01;
    d.timeLoopParams.printInterval  = 1;
    d.timeLoopParams.checkNaN       = 1;

    strncpy_safe(d.outputParams.baseName, "output", sizeof(d.outputParams.baseName));
    strncpy_safe(d.outputParams.directory, "VTK", sizeof(d.outputParams.directory));
    d.outputParams.format = VTK_TEXT;

    d.smoothingParams.iterations = 0;

    d.restartParams.file[0] = '\0';
    d.restartParams.checkpoint = 0;

    d.defaultState = ic_state_defaults();

    d.patches  = NULL;
    d.nPatches = 0;

    return d;
}

void input_data_free(InputData* d) {
    if (d->patches) {
        for (int i = 0; i < d->nPatches; ++i) {
            ic_patch_free(&d->patches[i]);
        }
        std::free(d->patches);
        d->patches = NULL;
    }
    d->nPatches = 0;
}

InputData parse_input_file(const char* filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error(std::string("Cannot open input file: ") + filename);
    }

    std::ostringstream oss;
    oss << ifs.rdbuf();
    std::string content = oss.str();
    return parse_input_string(content.c_str());
}

InputData parse_input_string(const char* jsonStr) {
    std::string cleaned = stripComments(std::string(jsonStr));

    json root;
    try {
        root = json::parse(cleaned);
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("JSON parse error: ") + e.what());
    }

    return parseJson(root);
}

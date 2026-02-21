#!/usr/bin/env python3
"""
Code generation tool for SemiImplicitFV.

Reads a JSON/JSONC input file and produces a standalone C++ main() function
with hardcoded config values for maximum performance.

Usage:
    python codegen.py input.jsonc --output generated_case.cpp

The generated source can be compiled and linked against SemiImplicitFV:
    g++ -O3 -std=c++17 generated_case.cpp -lSemiImplicitFV -lMPI ...
"""

import argparse
import json
import re
import sys
import os


def strip_comments(text):
    """Strip C/C++ style comments from JSONC text."""
    result = []
    i = 0
    in_string = False
    while i < len(text):
        if in_string:
            if text[i] == '\\' and i + 1 < len(text):
                result.append(text[i:i+2])
                i += 2
            else:
                if text[i] == '"':
                    in_string = False
                result.append(text[i])
                i += 1
        else:
            if text[i] == '"':
                in_string = True
                result.append(text[i])
                i += 1
            elif text[i] == '/' and i + 1 < len(text):
                if text[i+1] == '/':
                    i += 2
                    while i < len(text) and text[i] != '\n':
                        i += 1
                elif text[i+1] == '*':
                    i += 2
                    while i + 1 < len(text) and not (text[i] == '*' and text[i+1] == '/'):
                        i += 1
                    if i + 1 < len(text):
                        i += 2
                else:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
    return ''.join(result)


def _convert_pow(expr):
    """Convert ^ (exponentiation) to std::pow() in a C++ expression.

    Handles nested parentheses by scanning for balanced groups.
    E.g. '((x - 0.5) / 0.05)^2' -> 'std::pow((x - 0.5) / 0.05, 2)'
    """
    while '^' in expr:
        pos = expr.index('^')

        # Find base: scan left from ^ to find balanced paren group or token
        base_end = pos
        if pos > 0 and expr[pos - 1] == ')':
            # Walk backward to find matching '('
            depth = 0
            i = pos - 1
            while i >= 0:
                if expr[i] == ')':
                    depth += 1
                elif expr[i] == '(':
                    depth -= 1
                    if depth == 0:
                        break
                i -= 1
            base_start = i
        else:
            # Token (number, variable, etc.)
            i = pos - 1
            while i >= 0 and (expr[i].isalnum() or expr[i] in '_.'):
                i -= 1
            base_start = i + 1

        # Find exponent: scan right from ^ to find token or balanced paren group
        exp_start = pos + 1
        if exp_start < len(expr) and expr[exp_start] == '(':
            depth = 0
            i = exp_start
            while i < len(expr):
                if expr[i] == '(':
                    depth += 1
                elif expr[i] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            exp_end = i + 1
        else:
            # Token (possibly with leading minus for negative exponents)
            i = exp_start
            if i < len(expr) and expr[i] == '-':
                i += 1
            while i < len(expr) and (expr[i].isalnum() or expr[i] in '_.'):
                i += 1
            exp_end = i

        base = expr[base_start:base_end]
        exponent = expr[exp_start:exp_end]

        if not base or not exponent:
            break  # safety

        # For parenthesized base, unwrap the outer parens for cleaner output
        if base.startswith('(') and base.endswith(')'):
            inner = base[1:-1]
            replacement = f'std::pow({inner}, {exponent})'
        else:
            replacement = f'std::pow({base}, {exponent})'

        expr = expr[:base_start] + replacement + expr[exp_end:]

    return expr


def load_jsonc(filename):
    """Load a JSONC file (JSON with comments)."""
    with open(filename, 'r') as f:
        text = f.read()
    cleaned = strip_comments(text)
    return json.loads(cleaned)


RECON_ORDER_MAP = {
    "WENO1": "ReconstructionOrder::WENO1",
    "WENO3": "ReconstructionOrder::WENO3",
    "WENO5": "ReconstructionOrder::WENO5",
    "UPWIND1": "ReconstructionOrder::UPWIND1",
    "UPWIND3": "ReconstructionOrder::UPWIND3",
    "UPWIND5": "ReconstructionOrder::UPWIND5",
}

BC_MAP = {
    "Symmetry": "BoundaryCondition::Symmetry",
    "Outflow": "BoundaryCondition::Outflow",
    "Periodic": "BoundaryCondition::Periodic",
    "SlipWall": "BoundaryCondition::SlipWall",
    "NoSlipWall": "BoundaryCondition::NoSlipWall",
}

RIEMANN_MAP = {
    "LF": "LFSolver",
    "Rusanov": "RusanovSolver",
    "HLLC": "HLLCSolver",
}

RIEMANN_HEADERS = {
    "LF": "LFSolver.hpp",
    "Rusanov": "RusanovSolver.hpp",
    "HLLC": "HLLCSolver.hpp",
}


def generate_config(cfg):
    """Generate config setup code."""
    lines = []
    lines.append("    SimulationConfig config;")

    simple_fields = {
        "dim": "dim", "nGhost": "nGhost", "RKOrder": "RKOrder",
        "useIGR": "useIGR", "semiImplicit": "semiImplicit",
        "wenoEps": "wenoEps",
    }
    for jkey, ckey in simple_fields.items():
        if jkey in cfg:
            val = cfg[jkey]
            if isinstance(val, bool):
                val = "true" if val else "false"
            lines.append(f"    config.{ckey} = {val};")

    if "reconOrder" in cfg:
        ro = RECON_ORDER_MAP.get(cfg["reconOrder"], "ReconstructionOrder::WENO1")
        lines.append(f"    config.reconOrder = {ro};")

    # Explicit params
    if "explicitParams" in cfg:
        ep = cfg["explicitParams"]
        for k, v in ep.items():
            lines.append(f"    config.explicitParams.{k} = {v};")

    # Semi-implicit params
    if "semiImplicitParams" in cfg:
        sp = cfg["semiImplicitParams"]
        for k, v in sp.items():
            lines.append(f"    config.semiImplicitParams.{k} = {v};")

    # IGR params
    if "igrParams" in cfg:
        ip = cfg["igrParams"]
        for k, v in ip.items():
            lines.append(f"    config.igrParams.{k} = {v};")

    # Multi-phase params
    if "multiPhaseParams" in cfg:
        mp = cfg["multiPhaseParams"]
        if "nPhases" in mp:
            lines.append(f"    config.multiPhaseParams.nPhases = {mp['nPhases']};")
        if "alphaMin" in mp:
            lines.append(f"    config.multiPhaseParams.alphaMin = {mp['alphaMin']};")
        if "phases" in mp:
            phases_str = ", ".join(
                f"{{{ph.get('gamma', 1.4)}, {ph.get('pInf', 0.0)}}}"
                for ph in mp["phases"]
            )
            lines.append(f"    config.multiPhaseParams.phases = {{{phases_str}}};")

    # Body force
    if "bodyForceParams" in cfg:
        bf = cfg["bodyForceParams"]
        for k in ["a", "b", "c", "d"]:
            if k in bf:
                arr = bf[k]
                lines.append(f"    config.bodyForceParams.{k} = {{{arr[0]}, {arr[1]}, {arr[2]}}};")

    # Viscous
    if "viscousParams" in cfg:
        vp = cfg["viscousParams"]
        if "mu" in vp:
            lines.append(f"    config.viscousParams.mu = {vp['mu']};")
        if "phaseMu" in vp:
            mu_str = ", ".join(str(m) for m in vp["phaseMu"])
            lines.append(f"    config.viscousParams.phaseMu = {{{mu_str}}};")

    # Surface tension
    if "surfaceTensionParams" in cfg:
        st = cfg["surfaceTensionParams"]
        if "sigma" in st:
            lines.append(f"    config.surfaceTensionParams.sigma = {st['sigma']};")
        if "epsGradAlpha" in st:
            lines.append(f"    config.surfaceTensionParams.epsGradAlpha = {st['epsGradAlpha']};")

    lines.append("    config.validate();")
    return "\n".join(lines)


def generate_ic_loop(data, dim):
    """Generate the initial conditions loop."""
    default_st = data.get("initialConditions", {}).get("default", {})
    patches = data.get("initialConditions", {}).get("patches", [])
    is_multi = data.get("config", {}).get("multiPhaseParams", {}).get("nPhases", 0) >= 2

    lines = []

    # Default state
    lines.append("    // Default state")
    lines.append(f"    const double def_rho = {default_st.get('rho', 1.0)};")
    lines.append(f"    const double def_u   = {default_st.get('u', 0.0)};")
    lines.append(f"    const double def_v   = {default_st.get('v', 0.0)};")
    lines.append(f"    const double def_w   = {default_st.get('w', 0.0)};")
    lines.append(f"    const double def_p   = {default_st.get('p', 1.0)};")
    lines.append("")

    # Main loop
    k_loop = "    for (int k = 0; k < mesh.nz(); ++k) {" if dim >= 3 else "    { int k = 0;"
    j_loop = "    for (int j = 0; j < mesh.ny(); ++j) {" if dim >= 2 else "    { int j = 0;"
    lines.append(k_loop)
    lines.append(j_loop)
    lines.append("    for (int i = 0; i < mesh.nx(); ++i) {")
    lines.append("        std::size_t idx = mesh.index(i, j, k);")
    lines.append("        double x = mesh.cellCentroidX(i);")
    if dim >= 2:
        lines.append("        double y = mesh.cellCentroidY(j);")
    else:
        lines.append("        [[maybe_unused]] double y = 0.0;")
    if dim >= 3:
        lines.append("        double z = mesh.cellCentroidZ(k);")
    else:
        lines.append("        [[maybe_unused]] double z = 0.0;")
    lines.append("")
    lines.append("        double rho = def_rho, u = def_u, v = def_v, w = def_w, p = def_p;")

    if is_multi and "alpha" in default_st:
        alpha_str = ", ".join(str(a) for a in default_st["alpha"])
        lines.append(f"        std::vector<double> alphas = {{{alpha_str}}};")
        if "alphaRho" in default_st:
            arho_str = ", ".join(str(a) for a in default_st["alphaRho"])
            lines.append(f"        std::vector<double> alphaRhos = {{{arho_str}}};")
    lines.append("")

    # Apply patches — each patch is independent (not if/else)
    for pi, patch in enumerate(patches):
        geom = patch.get("geometry", {})
        gtype = geom.get("type", "box")
        state = patch.get("state", {})
        expressions = patch.get("expressions", {})

        # Generate condition
        if gtype == "sphere":
            c = geom.get("center", [0, 0, 0])
            r = geom.get("radius", 1.0)
            cond = (f"((x-{c[0]})*(x-{c[0]}) + (y-{c[1]})*(y-{c[1]}) "
                    f"+ (z-{c[2]})*(z-{c[2]})) <= {r*r}")
        elif gtype == "plane":
            pt = geom.get("point", [0, 0, 0])
            n = geom.get("normal", [1, 0, 0])
            cond = f"(x - {pt[0]}) * {n[0]} + (y - {pt[1]}) * {n[1]} + (z - {pt[2]}) * {n[2]} > 0.0"
        elif gtype == "box":
            lo = geom.get("min", [0, 0, 0])
            hi = geom.get("max", [1, 1, 1])
            cond = (f"x >= {lo[0]} && x <= {hi[0]} && "
                    f"y >= {lo[1]} && y <= {hi[1]} && z >= {lo[2]} && z <= {hi[2]}")
        elif gtype == "analytic":
            cond = "true"
        else:
            cond = "true"

        # Determine what state assignments to generate
        state_assignments = []
        for field in ["rho", "u", "v", "w", "p"]:
            if field in state:
                state_assignments.append(f"            {field} = {state[field]};")
        if is_multi and "alpha" in state:
            for ai, a in enumerate(state["alpha"]):
                state_assignments.append(f"            alphas[{ai}] = {a};")
            if "alphaRho" in state:
                for ai, a in enumerate(state["alphaRho"]):
                    state_assignments.append(f"            alphaRhos[{ai}] = {a};")

        # Expression assignments (analytic patches)
        expr_assignments = []
        for name, expr in expressions.items():
            # Convert expression to C++ (replace common math functions)
            cpp_expr = expr
            for fn in ["sin", "cos", "tan", "exp", "log", "sqrt", "abs", "pow",
                        "asin", "acos", "atan", "atan2", "fabs", "tanh"]:
                cpp_expr = re.sub(rf'\b{fn}\b', f'std::{fn}', cpp_expr)
            # Convert ^ (exponentiation) to std::pow() calls
            cpp_expr = _convert_pow(cpp_expr)
            if name in ["rho", "u", "v", "w", "p"]:
                expr_assignments.append(f"            {name} = {cpp_expr};")
            elif name.startswith("alpha_") and is_multi:
                idx = name[6:]
                expr_assignments.append(f"            alphas[{idx}] = {cpp_expr};")
            elif name.startswith("alphaRho_") and is_multi:
                idx = name[9:]
                expr_assignments.append(f"            alphaRhos[{idx}] = {cpp_expr};")

        has_state = bool(state_assignments)
        has_expr = bool(expr_assignments)

        if has_state or has_expr:
            if cond == "true":
                lines.append(f"        {{ // Patch {pi}")
            else:
                lines.append(f"        if ({cond}) {{ // Patch {pi}")
            for line in state_assignments:
                lines.append(line)
            for line in expr_assignments:
                lines.append(line)
            lines.append("        }")

    lines.append("")

    # Set state
    lines.append("        PrimitiveState W;")
    lines.append("        W.rho = rho; W.u = {u, v, w}; W.p = p;")
    lines.append("        W.T = eos->temperature(W);")
    lines.append("        state.setPrimitiveState(idx, W);")

    if is_multi:
        nph = data["config"]["multiPhaseParams"]["nPhases"]
        lines.append(f"        double mixRho = 0.0;")
        lines.append(f"        for (int ph = 0; ph < {nph}; ++ph) {{")
        lines.append(f"            state.alpha[ph][idx] = alphas[ph];")
        lines.append(f"            state.alphaRho[ph][idx] = alphaRhos[ph];")
        lines.append(f"            mixRho += alphaRhos[ph];")
        lines.append(f"        }}")
        lines.append(f"        state.rho[idx] = mixRho;")
        lines.append(f"        double ke = 0.5 * mixRho * (u*u + v*v + w*w);")
        lines.append(f"        state.rhoE[idx] = MixtureEOS::mixtureTotalEnergy(mixRho, p, alphas, ke, config.multiPhaseParams);")
        lines.append(f"        state.rhoU[idx] = mixRho * u;")
        if dim >= 2:
            lines.append(f"        state.rhoV[idx] = mixRho * v;")
        if dim >= 3:
            lines.append(f"        state.rhoW[idx] = mixRho * w;")
    else:
        lines.append("        state.setConservativeState(idx, eos->toConservative(W));")

    lines.append("    }")  # i loop
    if dim >= 2:
        lines.append("    }")  # j loop
    else:
        lines.append("    }")  # j block
    if dim >= 3:
        lines.append("    }")  # k loop
    else:
        lines.append("    }")  # k block

    return "\n".join(lines)


def generate(data, input_filename):
    """Generate the complete C++ source."""
    cfg = data.get("config", {})
    dim = cfg.get("dim", 1)
    eos_cfg = data.get("eos", {})
    riemann_type = data.get("riemannSolver", "HLLC")
    mesh_cfg = data.get("mesh", {})
    bc_cfg = data.get("boundaryConditions", {})
    tl_cfg = data.get("timeLoop", {})
    out_cfg = data.get("output", {})
    smooth_cfg = data.get("smoothing", {})
    restart_cfg = data.get("restart", {})
    is_semi = cfg.get("semiImplicit", False)
    is_multi = cfg.get("multiPhaseParams", {}).get("nPhases", 0) >= 2
    do_checkpoint = restart_cfg.get("checkpoint", False)
    restart_file = restart_cfg.get("file", "")

    # Includes
    includes = [
        "RectilinearMesh.hpp", "SolutionState.hpp", "State.hpp",
        RIEMANN_HEADERS[riemann_type],
        "SimulationConfig.hpp", "Runtime.hpp", "VTKSession.hpp",
        "RKTimeStepping.hpp",
    ]

    if eos_cfg.get("type", "IdealGas") == "StiffenedGas":
        includes.append("StiffenedGasEOS.hpp")
    else:
        includes.append("IdealGasEOS.hpp")

    pressure_solver_type = data.get("pressureSolver", "GaussSeidel") if is_semi else None

    if is_semi:
        includes.append("SemiImplicitSolver.hpp")
        if pressure_solver_type == "Jacobi":
            includes.append("JacobiPressureSolver.hpp")
        elif pressure_solver_type != "PETSc":
            includes.append("GaussSeidelPressureSolver.hpp")
    else:
        includes.append("ExplicitSolver.hpp")

    if cfg.get("useIGR", False):
        includes.append("IGR.hpp")

    if is_multi:
        includes.append("MixtureEOS.hpp")

    if do_checkpoint or restart_file:
        includes.append("Checkpoint.hpp")

    include_str = "\n".join(f'#include "{h}"' for h in includes)

    # PETSc include is conditional on SIFV_HAS_PETSC (matches driver pattern)
    if pressure_solver_type == "PETSc":
        include_str += ("\n#ifdef SIFV_HAS_PETSC\n"
                        '#include "PETScPressureSolver.hpp"\n'
                        "#endif")

    # Standard includes
    std_inc_list = [
        "#include <iostream>",
        "#include <memory>",
        "#include <cmath>",
        "#include <vector>",
        "#include <functional>",
    ]
    if restart_file:
        std_inc_list.append("#include <iomanip>")
        std_inc_list.append("#include <sstream>")
    std_includes = "\n".join(std_inc_list)

    # Config code
    config_code = generate_config(cfg)

    # Derive periodic flags from BCs for MPI topology
    px = 1 if bc_cfg.get("xLow", "") == "Periodic" else 0
    py = 1 if bc_cfg.get("yLow", "") == "Periodic" else 0
    pz = 1 if bc_cfg.get("zLow", "") == "Periodic" else 0
    periods_arg = f", std::array<int,3>{{{{{px}, {py}, {pz}}}}})"

    # Mesh creation
    if dim == 1:
        mesh_code = (f"    RectilinearMesh mesh = rt.createUniformMesh(config, "
                     f"{mesh_cfg.get('nx', 100)}, {mesh_cfg.get('xMin', 0.0)}, {mesh_cfg.get('xMax', 1.0)}"
                     f"{periods_arg};")
    elif dim == 2:
        mesh_code = (f"    RectilinearMesh mesh = rt.createUniformMesh(config, "
                     f"{mesh_cfg.get('nx', 100)}, {mesh_cfg.get('xMin', 0.0)}, {mesh_cfg.get('xMax', 1.0)}, "
                     f"{mesh_cfg.get('ny', 1)}, {mesh_cfg.get('yMin', 0.0)}, {mesh_cfg.get('yMax', 1.0)}"
                     f"{periods_arg};")
    else:
        mesh_code = (f"    RectilinearMesh mesh = rt.createUniformMesh(config, "
                     f"{mesh_cfg.get('nx', 100)}, {mesh_cfg.get('xMin', 0.0)}, {mesh_cfg.get('xMax', 1.0)}, "
                     f"{mesh_cfg.get('ny', 1)}, {mesh_cfg.get('yMin', 0.0)}, {mesh_cfg.get('yMax', 1.0)}, "
                     f"{mesh_cfg.get('nz', 1)}, {mesh_cfg.get('zMin', 0.0)}, {mesh_cfg.get('zMax', 1.0)}"
                     f"{periods_arg};")

    # BCs
    face_names = ["xLow", "xHigh", "yLow", "yHigh", "zLow", "zHigh"]
    bc_lines = []
    for fi, fn in enumerate(face_names):
        if fn in bc_cfg:
            bc_val = BC_MAP.get(bc_cfg[fn], "BoundaryCondition::Outflow")
            bc_lines.append(f"    rt.setBoundaryCondition(mesh, {fi}, {bc_val});")
    bc_code = "\n".join(bc_lines)

    # EOS
    eos_type = eos_cfg.get("type", "IdealGas")
    gamma = eos_cfg.get("gamma", 1.4)
    R = eos_cfg.get("R", 287.0)
    pInf = eos_cfg.get("pInf", 0.0)
    if eos_type == "StiffenedGas":
        eos_code = f"    auto eos = std::make_shared<StiffenedGasEOS>({gamma}, {pInf}, {R}, config);"
    else:
        eos_code = f"    auto eos = std::make_shared<IdealGasEOS>({gamma}, {R}, config);"

    # Riemann solver
    rs_class = RIEMANN_MAP[riemann_type]
    rs_code = f"    auto riemannSolver = std::make_shared<{rs_class}>(eos, config);"

    # IC loop
    ic_code = generate_ic_loop(data, dim)

    # Smoothing (placed after solver attachment since attachSolver creates HaloExchange)
    smooth_iters = smooth_cfg.get("iterations", 0)
    smooth_code = f"    rt.smoothFields(state, mesh, {smooth_iters}, config);" if smooth_iters > 0 else ""

    # Solver
    if is_semi:
        igr = "igrSolver" if cfg.get("useIGR") else "nullptr"
        if pressure_solver_type == "PETSc":
            # Detect periodic flags from BCs
            px = 1 if bc_cfg.get("xLow", "") == "Periodic" else 0
            py = 1 if bc_cfg.get("yLow", "") == "Periodic" else 0
            pz = 1 if bc_cfg.get("zLow", "") == "Periodic" else 0
            ps_lines = [
                "#ifdef SIFV_HAS_PETSC",
                "    std::shared_ptr<PressureSolver> pressureSolver;",
                "    if (rt.size() > 1) {",
                "        pressureSolver = std::make_shared<PETScPressureSolver>(",
                "            rt.mpiContext().comm(),",
                "            rt.mpiContext().localExtent(),",
                f"            std::array<int,3>{{{px}, {py}, {pz}}},",
                "            rt.mpiContext().dims());",
                "    } else {",
                "        pressureSolver = std::make_shared<PETScPressureSolver>();",
                "    }",
                '#else',
                '    #error "This case requires PETSc. Build with -DENABLE_PETSC=ON."',
                "#endif",
            ]
            ps_code = "\n".join(ps_lines)
        elif pressure_solver_type == "Jacobi":
            ps_code = "    auto pressureSolver = std::make_shared<JacobiPressureSolver>();"
        else:
            ps_code = "    auto pressureSolver = std::make_shared<GaussSeidelPressureSolver>();"

        solver_code = (ps_code + "\n" +
            "    auto solver = std::make_unique<SemiImplicitSolver>(\n"
            f"        mesh, riemannSolver, pressureSolver, eos, {igr}, config);\n"
            "    rt.attachSolver(*solver, mesh);\n"
            "    std::function<double(double)> stepFn = [&](double targetDt) {\n"
            "        return solver->step(config, mesh, state, targetDt);\n"
            "    };")
    else:
        igr = "igrSolver" if cfg.get("useIGR") else "nullptr"
        solver_code = (
            "    auto solver = std::make_unique<ExplicitSolver>(\n"
            f"        mesh, riemannSolver, eos, {igr}, config);\n"
            "    rt.attachSolver(*solver, mesh);\n"
            "    std::function<double(double)> stepFn = [&](double targetDt) {\n"
            "        return solver->step(config, mesh, state, targetDt);\n"
            "    };")

    # IGR
    igr_code = ""
    if cfg.get("useIGR", False):
        igr_code = "    auto igrSolver = std::make_shared<IGRSolver>(config.igrParams);"

    # VTK + time loop
    base = out_cfg.get("baseName", "output")
    vtk_dir = out_cfg.get("directory", "VTK")
    end_time = tl_cfg.get("endTime", 1.0)
    out_int = tl_cfg.get("outputInterval", 0.01)
    print_int = tl_cfg.get("printInterval", 1)

    # Restart / checkpoint code
    if restart_file:
        restart_load_code = f'''    // ---- Load checkpoint or apply ICs ----
    const std::string restartFile = "{restart_file}";
    {{
        std::string ckptFile = restartFile;
        std::string placeholder = "{{rank}}";
        auto pos = ckptFile.find(placeholder);
        if (pos != std::string::npos) {{
            std::ostringstream rankStr;
            rankStr << std::setw(4) << std::setfill('0') << rt.rank();
            ckptFile.replace(pos, placeholder.size(), rankStr.str());
        }}
        loadCheckpoint(ckptFile, mesh, state, config);
        state.convertConservativeToPrimitiveVariables(mesh, eos);
        rt.print("  Restarting from checkpoint: ", ckptFile, "\\n");
        rt.print("  Restart time: ", config.time, ", step: ", config.step, "\\n\\n");
    }}'''
        init_code = restart_load_code
    else:
        init_code = f"    // Initialize fields\n{ic_code}"

    checkpoint_line = f"    tlp.checkpoint = true;" if do_checkpoint else ""

    # Assemble
    source = f"""// Auto-generated from {os.path.basename(input_filename)}
// Compile: link against SemiImplicitFV library

{include_str}

{std_includes}

using namespace SemiImplicitFV;

int main(int argc, char** argv) {{
    Runtime rt(argc, argv);

{config_code}

{mesh_code}
{bc_code}

    SolutionState state;
    state.allocate(mesh.totalCells(), config);

{eos_code}
{rs_code}
{igr_code}

{init_code}

{solver_code}

{smooth_code if not restart_file else ""}

    VTKSession vtk(rt, "{base}", mesh, config, "{vtk_dir}");

    TimeLoopParams tlp;
    tlp.endTime = {end_time};
    tlp.outputInterval = {out_int};
    tlp.printInterval = {print_int};
{checkpoint_line}
    tlp.startTime = config.time;
{f"""    tlp.acousticDtFn = [&]() {{
        return computeAcousticTimeStep(mesh, state, *eos, config,
                                       1.0, 1e30, rt.mpiContext().comm());
    }};""" if is_semi else ""}
    runTimeLoop(rt, config, mesh, state, vtk, stepFn, tlp);

    return 0;
}}
"""
    return source


def main():
    parser = argparse.ArgumentParser(
        description="Generate a compiled C++ main() from a SemiImplicitFV JSON input file")
    parser.add_argument("input", help="JSON/JSONC input file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output C++ file (default: stdout)")
    args = parser.parse_args()

    data = load_jsonc(args.input)
    source = generate(data, args.input)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(source)
        print(f"Generated {args.output}", file=sys.stderr)
    else:
        print(source)


if __name__ == "__main__":
    main()

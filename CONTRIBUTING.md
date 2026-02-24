# Contributing to SemiImplicitFV

## Adding a New Case

The preferred way to define a simulation is with a **JSON input file**. No C++ coding is required for standard cases.

### JSON Case (Preferred)

1. Create `cases/<name>/<name>.jsonc`
2. Define the simulation using the JSON schema (see below)
3. Test: `./run_case.sh <name>`
4. VTK output appears in `cases/<name>/VTK/`

Example (`cases/my_case/my_case.jsonc`):

```jsonc
{
    "config": {
        "dim": 1,
        "nGhost": 3,
        "RKOrder": 3,
        "reconOrder": "WENO5",
        "explicitParams": { "cfl": 0.6 }
    },
    "eos": { "type": "IdealGas", "gamma": 1.4, "R": 287.0 },
    "riemannSolver": "HLLC",
    "mesh": { "nx": 200, "xMin": 0.0, "xMax": 1.0 },
    "boundaryConditions": { "xLow": "Outflow", "xHigh": "Outflow" },
    "timeLoop": { "endTime": 0.2, "outputInterval": 0.01 },
    "output": { "baseName": "my_case" },
    "initialConditions": {
        "default": { "rho": 1.0, "u": 0.0, "p": 1.0 },
        "patches": [
            {
                "geometry": { "type": "plane", "point": [0.5, 0, 0], "normal": [1, 0, 0] },
                "state": { "rho": 0.125, "p": 0.1 }
            }
        ]
    }
}
```

### JSON Schema Sections

| Section | Required | Description |
|---|---|---|
| `config` | Yes | Solver parameters: `dim`, `nGhost`, `RKOrder`, `reconOrder`, `semiImplicit`, `useIGR`, CFL/time-step params, multi-phase, viscosity, body forces, surface tension, IGR |
| `eos` | No | Equation of state: `"IdealGas"` (default) or `"StiffenedGas"` with `gamma`, `R`, `pInf` |
| `riemannSolver` | No | `"LF"`, `"Rusanov"`, or `"HLLC"` (default) |
| `pressureSolver` | No | `"GaussSeidel"` (default), `"Jacobi"`, or `"PETSc"` (requires `--petsc` build flag) |
| `mesh` | Yes | `nx`/`ny`/`nz` and `xMin`/`xMax`/`yMin`/`yMax`/`zMin`/`zMax` |
| `boundaryConditions` | No | Per-face: `"Outflow"`, `"Periodic"`, `"Symmetry"`, `"SlipWall"`, `"NoSlipWall"` |
| `timeLoop` | Yes | `endTime`, `outputInterval`, `printInterval`, `checkNaN` |
| `output` | No | `baseName`, `directory`, and `format` (`"VTKText"` or `"VTKRaw"`) for VTK output |
| `initialConditions` | Yes | `default` state + `patches` array with geometry and state overrides |
| `smoothing` | No | `iterations` for post-initialization field smoothing |
| `restart` | No | `checkpoint` (bool) and `file` (path) for checkpoint/restart |

### Initial Condition Geometries

- **`box`**: `min` [x,y,z], `max` [x,y,z]
- **`sphere`**: `center` [x,y,z], `radius`
- **`plane`**: `point` [x,y,z], `normal` [x,y,z] — positive side gets the patch state
- **`analytic`**: expressions evaluated per cell (with optional `region` sub-geometry)

### Compiled C++ Case (When Needed)

Use compiled cases only when you need logic not expressible in JSON — custom diagnostics, convergence studies, drag/lift computation, etc.

1. Create `cases/<name>/<name>.cpp`
2. Build and run: `./run_case.sh --compiled <name>`

Compiled cases use the C-style API: call `config_defaults()` to initialize a `SimulationConfig`, set fields, then use `runtime_init()` / `run_time_loop()` / `runtime_free()`.

### Code Generation

The `--case-optimization` flag generates optimized C++ from JSON and compiles it:

```bash
./run_case.sh --case-optimization <name>
```

This hardcodes all parameters at compile time for maximum performance.

## Code Style

- C++17, no namespaces (except `ExpressionEvaluator` class which uses pimpl)
- Headers use `#ifndef` include guards (not `#pragma once`)
- Plain C structs with `_init()` / `_free()` lifecycle functions for heap-allocated resources
- Free functions with `module_action()` naming (e.g. `config_validate()`, `eos_pressure()`, `mesh_index()`)
- Enums are plain `enum` (not `enum class`) with `UPPER_CASE` values (e.g. `EOS_IDEAL_GAS`, `RS_HLLC`, `BC_OUTFLOW`, `WENO5`)
- Flat `double*` arrays for field data; multi-phase uses stride-based access (`alpha[phase * totalCells + cell]`)
- `#define MAX_PHASES 8` for fixed-size phase arrays in structs
- No virtual dispatch — use enum + switch
- No `std::shared_ptr` or `std::unique_ptr` in data structs (except `ExpressionEvaluator` pimpl)

## Building and Running

`run_case.sh` handles configuring, building, and running automatically:

```bash
./run_case.sh <name>                         # Build and run
./run_case.sh --debug <name>                 # Debug build (AddressSanitizer)
./run_case.sh -n 4 <name>                    # Run with 4 MPI ranks
./run_case.sh --case-optimization <name>     # Codegen optimized build
./run_case.sh --list                         # List all cases
```

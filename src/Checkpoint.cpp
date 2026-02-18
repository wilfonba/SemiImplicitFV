#include "Checkpoint.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

namespace SemiImplicitFV {

void writeCheckpoint(
    const std::string& dir,
    const RectilinearMesh& mesh,
    const SolutionState& state,
    const SimulationConfig& config,
    int rank)
{
    // Ensure the output directory exists
    mkdir(dir.c_str(), 0755);

    // Build filename: dir/checkpoint.RRRR.bin
    std::ostringstream oss;
    oss << dir << "/checkpoint."
        << std::setw(4) << std::setfill('0') << rank << ".bin";
    std::string path = oss.str();

    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        throw std::runtime_error("Cannot open checkpoint file for writing: " + path);
    }

    // Write header
    uint64_t magic = CHECKPOINT_MAGIC;
    int32_t version = CHECKPOINT_VERSION;
    int32_t dim = config.dim;
    int32_t nPhases = config.multiPhaseParams.nPhases;
    int32_t nx = mesh.nx();
    int32_t ny = mesh.ny();
    int32_t nz = mesh.nz();
    int32_t nGhost = config.nGhost;
    int32_t step = config.step;
    double time = config.time;

    std::fwrite(&magic,   sizeof(magic),   1, fp);
    std::fwrite(&version, sizeof(version), 1, fp);
    std::fwrite(&dim,     sizeof(dim),     1, fp);
    std::fwrite(&nPhases, sizeof(nPhases), 1, fp);
    std::fwrite(&nx,      sizeof(nx),      1, fp);
    std::fwrite(&ny,      sizeof(ny),      1, fp);
    std::fwrite(&nz,      sizeof(nz),      1, fp);
    std::fwrite(&nGhost,  sizeof(nGhost),  1, fp);
    std::fwrite(&step,    sizeof(step),    1, fp);
    std::fwrite(&time,    sizeof(time),    1, fp);

    // Write conservative fields (entire arrays including ghosts)
    std::size_t n = state.rho.size();
    std::fwrite(state.rho.data(),  sizeof(double), n, fp);
    std::fwrite(state.rhoU.data(), sizeof(double), n, fp);
    if (dim >= 2) std::fwrite(state.rhoV.data(), sizeof(double), n, fp);
    if (dim >= 3) std::fwrite(state.rhoW.data(), sizeof(double), n, fp);
    std::fwrite(state.rhoE.data(), sizeof(double), n, fp);

    // Write multi-phase fields
    for (int ph = 0; ph < nPhases; ++ph) {
        std::fwrite(state.alpha[ph].data(), sizeof(double), n, fp);
    }
    for (int ph = 0; ph < nPhases; ++ph) {
        std::fwrite(state.alphaRho[ph].data(), sizeof(double), n, fp);
    }

    std::fclose(fp);
}

void loadCheckpoint(
    const std::string& filepath,
    const RectilinearMesh& mesh,
    SolutionState& state,
    SimulationConfig& config)
{
    FILE* fp = std::fopen(filepath.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("Cannot open checkpoint file for reading: " + filepath);
    }

    // Read and validate header
    uint64_t magic;
    int32_t version, dim, nPhases, nx, ny, nz, nGhost, step;
    double time;

    std::fread(&magic,   sizeof(magic),   1, fp);
    std::fread(&version, sizeof(version), 1, fp);
    std::fread(&dim,     sizeof(dim),     1, fp);
    std::fread(&nPhases, sizeof(nPhases), 1, fp);
    std::fread(&nx,      sizeof(nx),      1, fp);
    std::fread(&ny,      sizeof(ny),      1, fp);
    std::fread(&nz,      sizeof(nz),      1, fp);
    std::fread(&nGhost,  sizeof(nGhost),  1, fp);
    std::fread(&step,    sizeof(step),    1, fp);
    std::fread(&time,    sizeof(time),    1, fp);

    if (magic != CHECKPOINT_MAGIC) {
        std::fclose(fp);
        throw std::runtime_error("Invalid checkpoint file (bad magic): " + filepath);
    }
    if (version != CHECKPOINT_VERSION) {
        std::fclose(fp);
        throw std::runtime_error("Unsupported checkpoint version " + std::to_string(version)
                                 + " in file: " + filepath);
    }
    if (dim != config.dim) {
        std::fclose(fp);
        throw std::runtime_error("Checkpoint dim mismatch: file has " + std::to_string(dim)
                                 + ", config has " + std::to_string(config.dim));
    }
    if (nx != mesh.nx() || ny != mesh.ny() || nz != mesh.nz()) {
        std::fclose(fp);
        throw std::runtime_error("Checkpoint mesh size mismatch: file has ("
                                 + std::to_string(nx) + "," + std::to_string(ny) + "," + std::to_string(nz)
                                 + "), mesh has (" + std::to_string(mesh.nx()) + ","
                                 + std::to_string(mesh.ny()) + "," + std::to_string(mesh.nz()) + ")");
    }
    if (nGhost != config.nGhost) {
        std::fclose(fp);
        throw std::runtime_error("Checkpoint nGhost mismatch: file has " + std::to_string(nGhost)
                                 + ", config has " + std::to_string(config.nGhost));
    }
    if (nPhases != config.multiPhaseParams.nPhases) {
        std::fclose(fp);
        throw std::runtime_error("Checkpoint nPhases mismatch: file has " + std::to_string(nPhases)
                                 + ", config has " + std::to_string(config.multiPhaseParams.nPhases));
    }

    // Read conservative fields
    std::size_t n = state.rho.size();
    std::fread(state.rho.data(),  sizeof(double), n, fp);
    std::fread(state.rhoU.data(), sizeof(double), n, fp);
    if (dim >= 2) std::fread(state.rhoV.data(), sizeof(double), n, fp);
    if (dim >= 3) std::fread(state.rhoW.data(), sizeof(double), n, fp);
    std::fread(state.rhoE.data(), sizeof(double), n, fp);

    // Read multi-phase fields
    for (int ph = 0; ph < nPhases; ++ph) {
        std::fread(state.alpha[ph].data(), sizeof(double), n, fp);
    }
    for (int ph = 0; ph < nPhases; ++ph) {
        std::fread(state.alphaRho[ph].data(), sizeof(double), n, fp);
    }

    std::fclose(fp);

    // Set config time and step from checkpoint
    config.time = time;
    config.step = step;
}

} // namespace SemiImplicitFV

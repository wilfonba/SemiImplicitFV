#include "VTKWriter.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "SimulationConfig.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace SemiImplicitFV {

/// Ensure the parent directory of a file path exists, creating it if needed.
static void ensureParentDir(const std::string& filepath) {
    auto parent = std::filesystem::path(filepath).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

// ---------------------------------------------------------------------------
// Helper: collect interior cell values from a field into a contiguous buffer
// (skipping ghost cells via mesh.index).
// ---------------------------------------------------------------------------
static std::vector<double> gatherScalar(const RectilinearMesh& mesh,
                                        const std::vector<double>& field,
                                        int nx, int ny, int nz) {
    std::vector<double> buf(static_cast<std::size_t>(nx) * ny * nz);
    std::size_t pos = 0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                buf[pos++] = field[mesh.index(i, j, k)];
    return buf;
}

static std::vector<double> gatherVector(const RectilinearMesh& mesh,
                                        const std::vector<double>& fx,
                                        const std::vector<double>& fy,
                                        const std::vector<double>& fz,
                                        int dim, int nx, int ny, int nz) {
    std::size_t nCells = static_cast<std::size_t>(nx) * ny * nz;
    std::vector<double> buf(nCells * 3);
    std::size_t pos = 0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                std::size_t idx = mesh.index(i, j, k);
                buf[pos++] = fx[idx];
                buf[pos++] = (dim >= 2) ? fy[idx] : 0.0;
                buf[pos++] = (dim >= 3) ? fz[idx] : 0.0;
            }
    return buf;
}

// ---------------------------------------------------------------------------
// ASCII (text) VTR writer — original implementation
// ---------------------------------------------------------------------------
static void writeVTR_Text(std::ofstream& file,
                          const RectilinearMesh& mesh,
                          const SolutionState& state,
                          const SimulationConfig& config,
                          int nx, int ny, int nz,
                          int ei0, int ei1, int ej0, int ej1, int ek0, int ek1,
                          int rank)
{
    file << std::setprecision(15);

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"RectilinearGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    file << "  <RectilinearGrid WholeExtent=\""
         << ei0 << " " << ei1 << " "
         << ej0 << " " << ej1 << " "
         << ek0 << " " << ek1 << "\">\n";
    file << "    <Piece Extent=\""
         << ei0 << " " << ei1 << " "
         << ej0 << " " << ej1 << " "
         << ek0 << " " << ek1 << "\">\n";

    // Coordinates
    file << "      <Coordinates>\n";

    file << "        <DataArray type=\"Float64\" Name=\"X\" format=\"ascii\">\n";
    file << "         ";
    for (int i = 0; i <= nx; ++i) file << " " << mesh.nodeX(i);
    file << "\n        </DataArray>\n";

    file << "        <DataArray type=\"Float64\" Name=\"Y\" format=\"ascii\">\n";
    file << "         ";
    if (ny == 1) {
        file << " 0.0 " << mesh.dx(0);
    } else {
        for (int j = 0; j <= ny; ++j) file << " " << mesh.nodeY(j);
    }
    file << "\n        </DataArray>\n";

    file << "        <DataArray type=\"Float64\" Name=\"Z\" format=\"ascii\">\n";
    file << "         ";
    if (nz == 1) {
        file << " 0.0 " << std::min(mesh.dx(0), std::min(mesh.dy(0), 1.0));
    } else {
        for (int k = 0; k <= nz; ++k) file << " " << mesh.nodeZ(k);
    }
    file << "\n        </DataArray>\n";

    file << "      </Coordinates>\n";

    // Cell data
    file << "      <CellData>\n";

    auto writeScalar = [&](const std::string& name, const std::vector<double>& field) {
        file << "        <DataArray type=\"Float64\" Name=\"" << name
             << "\" format=\"ascii\">\n";
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j) {
                file << "         ";
                for (int i = 0; i < nx; ++i)
                    file << " " << field[mesh.index(i, j, k)];
                file << "\n";
            }
        file << "        </DataArray>\n";
    };

    int dim = state.dim();
    auto writeVector = [&](const std::string& name,
                           const std::vector<double>& fx,
                           const std::vector<double>& fy,
                           const std::vector<double>& fz) {
        file << "        <DataArray type=\"Float64\" Name=\"" << name
             << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j) {
                file << "         ";
                for (int i = 0; i < nx; ++i) {
                    std::size_t idx = mesh.index(i, j, k);
                    file << " " << fx[idx]
                         << " " << (dim >= 2 ? fy[idx] : 0.0)
                         << " " << (dim >= 3 ? fz[idx] : 0.0);
                }
                file << "\n";
            }
        file << "        </DataArray>\n";
    };

    writeScalar("Pressure", state.pres);
    if (config.useIGR) writeScalar("Sigma", state.sigma);
    writeScalar("TotalEnergy", state.rhoE);
    writeVector("Velocity", state.velU, state.velV, state.velW);
    writeVector("Momentum", state.rhoU, state.rhoV, state.rhoW);

    if (state.alphaRho.empty()) writeScalar("Density", state.rho);
    for (std::size_t ph = 0; ph < state.alphaRho.size(); ++ph)
        writeScalar("AlphaRho_" + std::to_string(ph), state.alphaRho[ph]);
    for (std::size_t ph = 0; ph < state.alpha.size(); ++ph)
        writeScalar("Alpha_" + std::to_string(ph), state.alpha[ph]);

    if (rank >= 0) {
        file << "        <DataArray type=\"Int32\" Name=\"Rank\" format=\"ascii\">\n";
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j) {
                file << "         ";
                for (int i = 0; i < nx; ++i) file << " " << rank;
                file << "\n";
            }
        file << "        </DataArray>\n";
    }

    file << "      </CellData>\n";
    file << "    </Piece>\n";
    file << "  </RectilinearGrid>\n";
    file << "</VTKFile>\n";
}

// ---------------------------------------------------------------------------
// Appended-raw binary VTR writer
// ---------------------------------------------------------------------------

/// A data block to be appended after the XML section.
struct AppendedBlock {
    std::vector<char> data;   // raw bytes (doubles or int32s)
};

static void writeVTR_Raw(std::ofstream& file,
                         const RectilinearMesh& mesh,
                         const SolutionState& state,
                         const SimulationConfig& config,
                         int nx, int ny, int nz,
                         int ei0, int ei1, int ej0, int ej1, int ek0, int ek1,
                         int rank)
{
    int dim = state.dim();
    std::size_t nCells = static_cast<std::size_t>(nx) * ny * nz;

    // ---- Phase 1: Build all data blocks ----
    std::vector<AppendedBlock> blocks;

    auto addDoubleBlock = [&](const std::vector<double>& buf) {
        AppendedBlock blk;
        blk.data.resize(buf.size() * sizeof(double));
        std::memcpy(blk.data.data(), buf.data(), blk.data.size());
        blocks.push_back(std::move(blk));
    };

    auto addInt32Block = [&](const std::vector<int32_t>& buf) {
        AppendedBlock blk;
        blk.data.resize(buf.size() * sizeof(int32_t));
        std::memcpy(blk.data.data(), buf.data(), blk.data.size());
        blocks.push_back(std::move(blk));
    };

    // Coordinate blocks (node arrays)
    {
        std::vector<double> xcoords(nx + 1);
        for (int i = 0; i <= nx; ++i) xcoords[i] = mesh.nodeX(i);
        addDoubleBlock(xcoords);
    }
    {
        std::vector<double> ycoords;
        if (ny == 1) {
            ycoords = {0.0, mesh.dx(0)};
        } else {
            ycoords.resize(ny + 1);
            for (int j = 0; j <= ny; ++j) ycoords[j] = mesh.nodeY(j);
        }
        addDoubleBlock(ycoords);
    }
    {
        std::vector<double> zcoords;
        if (nz == 1) {
            zcoords = {0.0, std::min(mesh.dx(0), std::min(mesh.dy(0), 1.0))};
        } else {
            zcoords.resize(nz + 1);
            for (int k = 0; k <= nz; ++k) zcoords[k] = mesh.nodeZ(k);
        }
        addDoubleBlock(zcoords);
    }

    // Cell-data blocks — track names, types, and component counts for XML tags
    struct FieldInfo {
        std::string name;
        std::string type;     // "Float64" or "Int32"
        int nComponents;      // 1 or 3
    };
    std::vector<FieldInfo> fields;

    auto addScalarField = [&](const std::string& name, const std::vector<double>& field) {
        addDoubleBlock(gatherScalar(mesh, field, nx, ny, nz));
        fields.push_back({name, "Float64", 1});
    };

    auto addVectorField = [&](const std::string& name,
                              const std::vector<double>& fx,
                              const std::vector<double>& fy,
                              const std::vector<double>& fz) {
        addDoubleBlock(gatherVector(mesh, fx, fy, fz, dim, nx, ny, nz));
        fields.push_back({name, "Float64", 3});
    };

    addScalarField("Pressure", state.pres);
    if (config.useIGR) addScalarField("Sigma", state.sigma);
    addScalarField("TotalEnergy", state.rhoE);
    addVectorField("Velocity", state.velU, state.velV, state.velW);
    addVectorField("Momentum", state.rhoU, state.rhoV, state.rhoW);

    if (state.alphaRho.empty()) addScalarField("Density", state.rho);
    for (std::size_t ph = 0; ph < state.alphaRho.size(); ++ph)
        addScalarField("AlphaRho_" + std::to_string(ph), state.alphaRho[ph]);
    for (std::size_t ph = 0; ph < state.alpha.size(); ++ph)
        addScalarField("Alpha_" + std::to_string(ph), state.alpha[ph]);

    if (rank >= 0) {
        std::vector<int32_t> rankBuf(nCells, rank);
        addInt32Block(rankBuf);
        fields.push_back({"Rank", "Int32", 1});
    }

    // ---- Phase 2: Compute byte offsets ----
    // Each block is preceded by a uint64_t header giving its byte count.
    // offset[i] = sum of (sizeof(uint64_t) + block[j].data.size()) for j < i.
    std::vector<uint64_t> offsets(blocks.size());
    uint64_t running = 0;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        offsets[i] = running;
        running += sizeof(uint64_t) + blocks[i].data.size();
    }

    // ---- Phase 3: Write XML header ----
    // Build XML into a string so we write it all at once to the binary stream.
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\"?>\n";
    xml << "<VTKFile type=\"RectilinearGrid\" version=\"1.0\""
        << " byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    xml << "  <RectilinearGrid WholeExtent=\""
        << ei0 << " " << ei1 << " "
        << ej0 << " " << ej1 << " "
        << ek0 << " " << ek1 << "\">\n";
    xml << "    <Piece Extent=\""
        << ei0 << " " << ei1 << " "
        << ej0 << " " << ej1 << " "
        << ek0 << " " << ek1 << "\">\n";

    // Coordinates — first 3 blocks
    xml << "      <Coordinates>\n";
    xml << "        <DataArray type=\"Float64\" Name=\"X\""
        << " format=\"appended\" offset=\"" << offsets[0] << "\"/>\n";
    xml << "        <DataArray type=\"Float64\" Name=\"Y\""
        << " format=\"appended\" offset=\"" << offsets[1] << "\"/>\n";
    xml << "        <DataArray type=\"Float64\" Name=\"Z\""
        << " format=\"appended\" offset=\"" << offsets[2] << "\"/>\n";
    xml << "      </Coordinates>\n";

    // Cell data — blocks starting at index 3
    xml << "      <CellData>\n";
    for (std::size_t f = 0; f < fields.size(); ++f) {
        std::size_t blockIdx = 3 + f;  // 3 coordinate blocks precede cell data
        xml << "        <DataArray type=\"" << fields[f].type
            << "\" Name=\"" << fields[f].name << "\"";
        if (fields[f].nComponents > 1)
            xml << " NumberOfComponents=\"" << fields[f].nComponents << "\"";
        xml << " format=\"appended\" offset=\"" << offsets[blockIdx] << "\"/>\n";
    }
    xml << "      </CellData>\n";

    xml << "    </Piece>\n";
    xml << "  </RectilinearGrid>\n";
    xml << "  <AppendedData encoding=\"raw\">\n_";

    // Write the XML portion
    std::string xmlStr = xml.str();
    file.write(xmlStr.data(), static_cast<std::streamsize>(xmlStr.size()));

    // ---- Phase 4: Write binary data blocks ----
    for (auto& blk : blocks) {
        uint64_t nbytes = blk.data.size();
        file.write(reinterpret_cast<const char*>(&nbytes), sizeof(uint64_t));
        file.write(blk.data.data(), static_cast<std::streamsize>(nbytes));
    }

    // Close XML
    std::string footer = "\n  </AppendedData>\n</VTKFile>\n";
    file.write(footer.data(), static_cast<std::streamsize>(footer.size()));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void VTKWriter::writeVTR(const std::string& filename,
                         const RectilinearMesh& mesh,
                         const SolutionState& state,
                         const SimulationConfig& config,
                         const std::array<int,6>& pieceExtent,
                         int rank,
                         VTKFormat format)
{
    int nx = mesh.nx(), ny = mesh.ny(), nz = mesh.nz();

    bool hasExtent = false;
    for (int d = 0; d < 6; ++d) {
        if (pieceExtent[d] != 0) { hasExtent = true; break; }
    }
    int ei0 = 0, ei1 = nx;
    int ej0 = 0, ej1 = ny;
    int ek0 = 0, ek1 = nz;
    if (hasExtent) {
        ei0 = pieceExtent[0]; ei1 = pieceExtent[1];
        ej0 = pieceExtent[2]; ej1 = pieceExtent[3];
        ek0 = pieceExtent[4]; ek1 = pieceExtent[5];
    }

    ensureParentDir(filename);

    if (format == VTKFormat::VTKRaw) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("VTKWriter::writeVTR: cannot open " + filename);
        writeVTR_Raw(file, mesh, state, config,
                     nx, ny, nz, ei0, ei1, ej0, ej1, ek0, ek1, rank);
        file.close();
    } else {
        std::ofstream file(filename);
        if (!file.is_open())
            throw std::runtime_error("VTKWriter::writeVTR: cannot open " + filename);
        writeVTR_Text(file, mesh, state, config,
                      nx, ny, nz, ei0, ei1, ej0, ej1, ek0, ek1, rank);
        file.close();
    }
}

void VTKWriter::writePVTR(const std::string& filename,
                          int globalNx, int globalNy, int globalNz,
                          const std::vector<std::array<int,6>>& pieceExtents,
                          const std::vector<std::string>& pieceFiles,
                          const SimulationConfig& config,
                          VTKFormat format)
{
    ensureParentDir(filename);
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("VTKWriter::writePVTR: cannot open " + filename);
    }

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PRectilinearGrid\" version=\"1.0\" byte_order=\"LittleEndian\"";
    if (format == VTKFormat::VTKRaw)
        file << " header_type=\"UInt64\"";
    file << ">\n";
    file << "  <PRectilinearGrid WholeExtent=\"0 " << globalNx
         << " 0 " << globalNy
         << " 0 " << globalNz << "\" GhostLevel=\"0\">\n";

    // Declare coordinate arrays
    file << "    <PCoordinates>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"X\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Y\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Z\"/>\n";
    file << "    </PCoordinates>\n";

    // Declare cell data arrays
    file << "    <PCellData>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Pressure\"/>\n";
    if (config.useIGR) {
        file << "      <PDataArray type=\"Float64\" Name=\"Sigma\"/>\n";
    }
    file << "      <PDataArray type=\"Float64\" Name=\"TotalEnergy\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Momentum\" NumberOfComponents=\"3\"/>\n";
    int nPhases = config.multiPhaseParams.nPhases;
    if (nPhases <= 0) {
        file << "      <PDataArray type=\"Float64\" Name=\"Density\"/>\n";
    }
    for (int ph = 0; ph < nPhases; ++ph) {
        file << "      <PDataArray type=\"Float64\" Name=\"AlphaRho_" << ph << "\"/>\n";
    }
    for (int ph = 0; ph < nPhases; ++ph) {
        file << "      <PDataArray type=\"Float64\" Name=\"Alpha_" << ph << "\"/>\n";
    }
    file << "      <PDataArray type=\"Int32\" Name=\"Rank\"/>\n";
    file << "    </PCellData>\n";

    // Reference each piece file with its extent
    for (std::size_t p = 0; p < pieceFiles.size(); ++p) {
        const auto& ext = pieceExtents[p];
        file << "    <Piece Extent=\""
             << ext[0] << " " << ext[1] << " "
             << ext[2] << " " << ext[3] << " "
             << ext[4] << " " << ext[5]
             << "\" Source=\"" << pieceFiles[p] << "\"/>\n";
    }

    file << "  </PRectilinearGrid>\n";
    file << "</VTKFile>\n";

    file.close();
}

// PVD footer written after every append so the file is always valid XML
// and can be opened in ParaView while the simulation is still running.
static const std::string pvdFooter = "  </Collection>\n</VTKFile>\n";

void VTKWriter::writePVD(const std::string& filename,
                         const std::string& mode,
                         double time,
                         const std::string& dataFile)
{
    ensureParentDir(filename);
    if (mode == "w") {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("VTKWriter::writePVD: cannot open " + filename);
        }
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
        file << "  <Collection>\n";
        file << pvdFooter;
        file.close();
    } else if (mode == "a") {
        // Truncate the closing tags, append new entry, re-write closing tags.
        auto fileSize = std::filesystem::file_size(filename);
        std::filesystem::resize_file(filename, fileSize - pvdFooter.size());

        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) {
            throw std::runtime_error("VTKWriter::writePVD: cannot open " + filename);
        }
        file << std::setprecision(15);
        file << "    <DataSet timestep=\"" << time
             << "\" file=\"" << dataFile << "\"/>\n";
        file << pvdFooter;
        file.close();
    }
    // "close" is a no-op — the file is always kept in a valid state.
}

} // namespace SemiImplicitFV

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
#include <string>
#include <vector>
#include <algorithm>

static void ensureParentDir(const std::string& filepath) {
    auto parent = std::filesystem::path(filepath).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

/* Collect interior cell values from a field into a contiguous buffer. */
static std::vector<double> gatherScalar(const RectilinearMesh* mesh,
                                        const double* field,
                                        int nx, int ny, int nz) {
    std::vector<double> buf((size_t)nx * ny * nz);
    size_t pos = 0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                buf[pos++] = field[mesh_index(mesh, i, j, k)];
    return buf;
}

static std::vector<double> gatherVector(const RectilinearMesh* mesh,
                                        const double* fx,
                                        const double* fy,
                                        const double* fz,
                                        int dim, int nx, int ny, int nz) {
    size_t nCells = (size_t)nx * ny * nz;
    std::vector<double> buf(nCells * 3);
    size_t pos = 0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);
                buf[pos++] = fx[idx];
                buf[pos++] = (dim >= 2 && fy) ? fy[idx] : 0.0;
                buf[pos++] = (dim >= 3 && fz) ? fz[idx] : 0.0;
            }
    return buf;
}

/* ASCII VTR writer */
static void writeVTR_Text(std::ofstream& file,
                          const RectilinearMesh* mesh,
                          const SolutionState* state,
                          const SimulationConfig* config,
                          int nx, int ny, int nz,
                          int ei0, int ei1, int ej0, int ej1, int ek0, int ek1,
                          int rank)
{
    int dim = state->dim;
    size_t tc = state->totalCells;

    file << std::setprecision(15);
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"RectilinearGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    file << "  <RectilinearGrid WholeExtent=\""
         << ei0 << " " << ei1 << " " << ej0 << " " << ej1 << " " << ek0 << " " << ek1 << "\">\n";
    file << "    <Piece Extent=\""
         << ei0 << " " << ei1 << " " << ej0 << " " << ej1 << " " << ek0 << " " << ek1 << "\">\n";

    file << "      <Coordinates>\n";
    file << "        <DataArray type=\"Float64\" Name=\"X\" format=\"ascii\">\n";
    file << "         ";
    for (int i = 0; i <= nx; ++i) file << " " << mesh_nodeX(mesh, i);
    file << "\n        </DataArray>\n";

    file << "        <DataArray type=\"Float64\" Name=\"Y\" format=\"ascii\">\n";
    file << "         ";
    if (ny == 1) {
        file << " 0.0 " << mesh_dx(mesh, 0);
    } else {
        for (int j = 0; j <= ny; ++j) file << " " << mesh_nodeY(mesh, j);
    }
    file << "\n        </DataArray>\n";

    file << "        <DataArray type=\"Float64\" Name=\"Z\" format=\"ascii\">\n";
    file << "         ";
    if (nz == 1) {
        file << " 0.0 " << std::min(mesh_dx(mesh, 0), std::min(mesh_dy(mesh, 0), 1.0));
    } else {
        for (int k = 0; k <= nz; ++k) file << " " << mesh_nodeZ(mesh, k);
    }
    file << "\n        </DataArray>\n";
    file << "      </Coordinates>\n";

    file << "      <CellData>\n";

    auto writeScalar = [&](const char* name, const double* field) {
        file << "        <DataArray type=\"Float64\" Name=\"" << name
             << "\" format=\"ascii\">\n";
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j) {
                file << "         ";
                for (int i = 0; i < nx; ++i)
                    file << " " << field[mesh_index(mesh, i, j, k)];
                file << "\n";
            }
        file << "        </DataArray>\n";
    };

    auto writeVector = [&](const char* name,
                           const double* fx, const double* fy, const double* fz) {
        file << "        <DataArray type=\"Float64\" Name=\"" << name
             << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j) {
                file << "         ";
                for (int i = 0; i < nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);
                    file << " " << fx[idx]
                         << " " << (dim >= 2 && fy ? fy[idx] : 0.0)
                         << " " << (dim >= 3 && fz ? fz[idx] : 0.0);
                }
                file << "\n";
            }
        file << "        </DataArray>\n";
    };

    writeScalar("Pressure", state->pres);
    if (config->useIGR) writeScalar("Sigma", state->sigma);
    writeScalar("TotalEnergy", state->rhoE);
    writeVector("Velocity", state->velU, state->velV, state->velW);
    writeVector("Momentum", state->rhoU, state->rhoV, state->rhoW);

    int nPhases = state->nPhases;
    if (nPhases <= 0) writeScalar("Density", state->rho);
    for (int ph = 0; ph < nPhases; ++ph) {
        std::string n = "AlphaRho_" + std::to_string(ph);
        writeScalar(n.c_str(), state->alphaRho + ph * tc);
    }
    for (int ph = 0; ph < nPhases; ++ph) {
        std::string n = "Alpha_" + std::to_string(ph);
        writeScalar(n.c_str(), state->alpha + ph * tc);
    }

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

/* Appended-raw binary VTR writer */
struct AppendedBlock {
    std::vector<char> data;
};

static void writeVTR_Raw(std::ofstream& file,
                         const RectilinearMesh* mesh,
                         const SolutionState* state,
                         const SimulationConfig* config,
                         int nx, int ny, int nz,
                         int ei0, int ei1, int ej0, int ej1, int ek0, int ek1,
                         int rank)
{
    int dim = state->dim;
    size_t nCells = (size_t)nx * ny * nz;
    size_t tc = state->totalCells;

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

    /* Coordinate blocks */
    {
        std::vector<double> xcoords(nx + 1);
        for (int i = 0; i <= nx; ++i) xcoords[i] = mesh_nodeX(mesh, i);
        addDoubleBlock(xcoords);
    }
    {
        std::vector<double> ycoords;
        if (ny == 1) {
            ycoords = {0.0, mesh_dx(mesh, 0)};
        } else {
            ycoords.resize(ny + 1);
            for (int j = 0; j <= ny; ++j) ycoords[j] = mesh_nodeY(mesh, j);
        }
        addDoubleBlock(ycoords);
    }
    {
        std::vector<double> zcoords;
        if (nz == 1) {
            zcoords = {0.0, std::min(mesh_dx(mesh, 0), std::min(mesh_dy(mesh, 0), 1.0))};
        } else {
            zcoords.resize(nz + 1);
            for (int k = 0; k <= nz; ++k) zcoords[k] = mesh_nodeZ(mesh, k);
        }
        addDoubleBlock(zcoords);
    }

    struct FieldInfo {
        std::string name;
        std::string type;
        int nComponents;
    };
    std::vector<FieldInfo> fields;

    auto addScalarField = [&](const char* name, const double* field) {
        addDoubleBlock(gatherScalar(mesh, field, nx, ny, nz));
        fields.push_back({name, "Float64", 1});
    };

    auto addVectorField = [&](const char* name,
                              const double* fx, const double* fy, const double* fz) {
        addDoubleBlock(gatherVector(mesh, fx, fy, fz, dim, nx, ny, nz));
        fields.push_back({name, "Float64", 3});
    };

    addScalarField("Pressure", state->pres);
    if (config->useIGR) addScalarField("Sigma", state->sigma);
    addScalarField("TotalEnergy", state->rhoE);
    addVectorField("Velocity", state->velU, state->velV, state->velW);
    addVectorField("Momentum", state->rhoU, state->rhoV, state->rhoW);

    int nPhases = state->nPhases;
    if (nPhases <= 0) addScalarField("Density", state->rho);
    for (int ph = 0; ph < nPhases; ++ph) {
        std::string n = "AlphaRho_" + std::to_string(ph);
        addScalarField(n.c_str(), state->alphaRho + ph * tc);
    }
    for (int ph = 0; ph < nPhases; ++ph) {
        std::string n = "Alpha_" + std::to_string(ph);
        addScalarField(n.c_str(), state->alpha + ph * tc);
    }

    if (rank >= 0) {
        std::vector<int32_t> rankBuf(nCells, rank);
        addInt32Block(rankBuf);
        fields.push_back({"Rank", "Int32", 1});
    }

    /* Compute byte offsets */
    std::vector<uint64_t> offsets(blocks.size());
    uint64_t running = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
        offsets[i] = running;
        running += sizeof(uint64_t) + blocks[i].data.size();
    }

    /* Write XML header */
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\"?>\n";
    xml << "<VTKFile type=\"RectilinearGrid\" version=\"1.0\""
        << " byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    xml << "  <RectilinearGrid WholeExtent=\""
        << ei0 << " " << ei1 << " " << ej0 << " " << ej1 << " " << ek0 << " " << ek1 << "\">\n";
    xml << "    <Piece Extent=\""
        << ei0 << " " << ei1 << " " << ej0 << " " << ej1 << " " << ek0 << " " << ek1 << "\">\n";

    xml << "      <Coordinates>\n";
    xml << "        <DataArray type=\"Float64\" Name=\"X\""
        << " format=\"appended\" offset=\"" << offsets[0] << "\"/>\n";
    xml << "        <DataArray type=\"Float64\" Name=\"Y\""
        << " format=\"appended\" offset=\"" << offsets[1] << "\"/>\n";
    xml << "        <DataArray type=\"Float64\" Name=\"Z\""
        << " format=\"appended\" offset=\"" << offsets[2] << "\"/>\n";
    xml << "      </Coordinates>\n";

    xml << "      <CellData>\n";
    for (size_t f = 0; f < fields.size(); ++f) {
        size_t blockIdx = 3 + f;
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

    std::string xmlStr = xml.str();
    file.write(xmlStr.data(), (std::streamsize)xmlStr.size());

    /* Write binary data blocks */
    for (auto& blk : blocks) {
        uint64_t nbytes = blk.data.size();
        file.write(reinterpret_cast<const char*>(&nbytes), sizeof(uint64_t));
        file.write(blk.data.data(), (std::streamsize)nbytes);
    }

    std::string footer = "\n  </AppendedData>\n</VTKFile>\n";
    file.write(footer.data(), (std::streamsize)footer.size());
}

/* ---------------------------------------------------------------------------
   Public API
   --------------------------------------------------------------------------- */

void vtk_write_vtr(const char* filename,
                   const RectilinearMesh* mesh,
                   const SolutionState* state,
                   const SimulationConfig* config,
                   const int* pieceExtent,
                   int rank,
                   VTKFormat format)
{
    int nx = mesh->nx, ny = mesh->ny, nz = mesh->nz;

    int hasExtent = 0;
    if (pieceExtent) {
        for (int d = 0; d < 6; ++d) {
            if (pieceExtent[d] != 0) { hasExtent = 1; break; }
        }
    }
    int ei0 = 0, ei1 = nx;
    int ej0 = 0, ej1 = ny;
    int ek0 = 0, ek1 = nz;
    if (hasExtent) {
        ei0 = pieceExtent[0]; ei1 = pieceExtent[1];
        ej0 = pieceExtent[2]; ej1 = pieceExtent[3];
        ek0 = pieceExtent[4]; ek1 = pieceExtent[5];
    }

    std::string fn(filename);
    ensureParentDir(fn);

    if (format == VTK_RAW) {
        std::ofstream file(fn, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("vtk_write_vtr: cannot open " + fn);
        writeVTR_Raw(file, mesh, state, config,
                     nx, ny, nz, ei0, ei1, ej0, ej1, ek0, ek1, rank);
        file.close();
    } else {
        std::ofstream file(fn);
        if (!file.is_open())
            throw std::runtime_error("vtk_write_vtr: cannot open " + fn);
        writeVTR_Text(file, mesh, state, config,
                      nx, ny, nz, ei0, ei1, ej0, ej1, ek0, ek1, rank);
        file.close();
    }
}

void vtk_write_pvtr(const char* filename,
                    int globalNx, int globalNy, int globalNz,
                    const int* pieceExtents,
                    const char* const* pieceFiles,
                    int nPieces,
                    const SimulationConfig* config,
                    VTKFormat format)
{
    std::string fn(filename);
    ensureParentDir(fn);
    std::ofstream file(fn);
    if (!file.is_open()) {
        throw std::runtime_error("vtk_write_pvtr: cannot open " + fn);
    }

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PRectilinearGrid\" version=\"1.0\" byte_order=\"LittleEndian\"";
    if (format == VTK_RAW)
        file << " header_type=\"UInt64\"";
    file << ">\n";
    file << "  <PRectilinearGrid WholeExtent=\"0 " << globalNx
         << " 0 " << globalNy
         << " 0 " << globalNz << "\" GhostLevel=\"0\">\n";

    file << "    <PCoordinates>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"X\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Y\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Z\"/>\n";
    file << "    </PCoordinates>\n";

    file << "    <PCellData>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Pressure\"/>\n";
    if (config->useIGR) {
        file << "      <PDataArray type=\"Float64\" Name=\"Sigma\"/>\n";
    }
    file << "      <PDataArray type=\"Float64\" Name=\"TotalEnergy\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Momentum\" NumberOfComponents=\"3\"/>\n";
    int nPhases = config->multiPhaseParams.nPhases;
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

    for (int p = 0; p < nPieces; ++p) {
        const int* ext = pieceExtents + p * 6;
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

static const std::string pvdFooter = "  </Collection>\n</VTKFile>\n";

void vtk_write_pvd(const char* filename,
                   char mode,
                   double time,
                   const char* dataFile)
{
    std::string fn(filename);
    ensureParentDir(fn);

    if (mode == 'w') {
        std::ofstream file(fn);
        if (!file.is_open()) {
            throw std::runtime_error("vtk_write_pvd: cannot open " + fn);
        }
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
        file << "  <Collection>\n";
        file << pvdFooter;
        file.close();
    } else if (mode == 'a') {
        auto fileSize = std::filesystem::file_size(fn);
        std::filesystem::resize_file(fn, fileSize - pvdFooter.size());

        std::ofstream file(fn, std::ios::app);
        if (!file.is_open()) {
            throw std::runtime_error("vtk_write_pvd: cannot open " + fn);
        }
        file << std::setprecision(15);
        file << "    <DataSet timestep=\"" << time
             << "\" file=\"" << (dataFile ? dataFile : "") << "\"/>\n";
        file << pvdFooter;
        file.close();
    }
    /* 'c' (close) is a no-op -- the file is always kept in a valid state */
}

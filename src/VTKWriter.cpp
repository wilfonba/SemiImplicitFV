#include "VTKWriter.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace SemiImplicitFV {

/// Ensure the parent directory of a file path exists, creating it if needed.
static void ensureParentDir(const std::string& filepath) {
    auto parent = std::filesystem::path(filepath).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void VTKWriter::writeVTR(const std::string& filename,
                         const RectilinearMesh& mesh,
                         const SolutionState& state,
                         const std::array<int,6>& pieceExtent,
                         int rank)
{
    // Local cell counts (always used for data access)
    int nx = mesh.nx(), ny = mesh.ny(), nz = mesh.nz();

    // Extent for XML header: use pieceExtent (global indices) if provided,
    // otherwise default to local 0-based extent.
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
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("VTKWriter::writeVTR: cannot open " + filename);
    }

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

    // Coordinates (always local 0-based access into mesh)
    file << "      <Coordinates>\n";

    // X coordinates
    file << "        <DataArray type=\"Float64\" Name=\"X\" format=\"ascii\">\n";
    file << "         ";
    for (int i = 0; i <= nx; ++i) {
        file << " " << mesh.nodeX(i);
    }
    file << "\n        </DataArray>\n";

    // Y coordinates
    file << "        <DataArray type=\"Float64\" Name=\"Y\" format=\"ascii\">\n";
    file << "         ";
    if (ny == 1) {
        // 1D case: write a small y width
        file << " 0.0 " << mesh.dx(0);
    } else {
        for (int j = 0; j <= ny; ++j) {
            file << " " << mesh.nodeY(j);
        }
    }
    file << "\n        </DataArray>\n";

    // Z coordinates
    file << "        <DataArray type=\"Float64\" Name=\"Z\" format=\"ascii\">\n";
    file << "         ";
    if (nz == 1) {
        // 2D or 1D case: write a small z depth
        file << " 0.0 " << std::min(mesh.dx(0), std::min(mesh.dy(0), 1.0));
    } else {
        for (int k = 0; k <= nz; ++k) {
            file << " " << mesh.nodeZ(k);
        }
    }
    file << "\n        </DataArray>\n";

    file << "      </Coordinates>\n";

    // Cell data (always local 0-based access)
    file << "      <CellData>\n";

    // Helper lambda: write a scalar field
    auto writeScalar = [&](const std::string& name, const std::vector<double>& field) {
        file << "        <DataArray type=\"Float64\" Name=\"" << name
             << "\" format=\"ascii\">\n";
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                file << "         ";
                for (int i = 0; i < nx; ++i) {
                    std::size_t idx = mesh.index(i, j, k);
                    file << " " << field[idx];
                }
                file << "\n";
            }
        }
        file << "        </DataArray>\n";
    };

    // Helper lambda: write a 3-component vector field, zero-padding inactive dims
    int dim = state.dim();
    auto writeVector = [&](const std::string& name,
                           const std::vector<double>& fx,
                           const std::vector<double>& fy,
                           const std::vector<double>& fz) {
        file << "        <DataArray type=\"Float64\" Name=\"" << name
             << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int k = 0; k < nz; ++k) {
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
        }
        file << "        </DataArray>\n";
    };

    // Scalar fields
    writeScalar("Pressure", state.pres);
    writeScalar("Sigma", state.sigma);
    writeScalar("TotalEnergy", state.rhoE);

    // Vector fields
    writeVector("Velocity", state.velU, state.velV, state.velW);
    writeVector("Momentum", state.rhoU, state.rhoV, state.rhoW);

    // Multi-phase fields
    for (std::size_t ph = 0; ph < state.alphaRho.size(); ++ph) {
        writeScalar("AlphaRho_" + std::to_string(ph), state.alphaRho[ph]);
    }
    for (std::size_t ph = 0; ph < state.alpha.size(); ++ph) {
        writeScalar("Alpha_" + std::to_string(ph), state.alpha[ph]);
    }

    // MPI rank field (only when rank >= 0)
    if (rank >= 0) {
        file << "        <DataArray type=\"Int32\" Name=\"Rank\" format=\"ascii\">\n";
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                file << "         ";
                for (int i = 0; i < nx; ++i) {
                    file << " " << rank;
                }
                file << "\n";
            }
        }
        file << "        </DataArray>\n";
    }

    file << "      </CellData>\n";
    file << "    </Piece>\n";
    file << "  </RectilinearGrid>\n";
    file << "</VTKFile>\n";

    file.close();
}

void VTKWriter::writePVTR(const std::string& filename,
                          int globalNx, int globalNy, int globalNz,
                          const std::vector<std::array<int,6>>& pieceExtents,
                          const std::vector<std::string>& pieceFiles,
                          int nPhases)
{
    ensureParentDir(filename);
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("VTKWriter::writePVTR: cannot open " + filename);
    }

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PRectilinearGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
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
    file << "      <PDataArray type=\"Float64\" Name=\"Sigma\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"TotalEnergy\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Momentum\" NumberOfComponents=\"3\"/>\n";
    for (int ph = 0; ph < nPhases; ++ph) {
        file << "      <PDataArray type=\"Float64\" Name=\"AlphaRho_" << ph << "\"/>\n";
    }
    for (int ph = 0; ph < nPhases - 1; ++ph) {
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

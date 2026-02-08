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
                         const std::array<int,6>& pieceExtent)
{
    // Determine extent: use full grid if pieceExtent is all zeros (serial mode)
    int i0 = 0, i1 = mesh.nx();
    int j0 = 0, j1 = mesh.ny();
    int k0 = 0, k1 = mesh.nz();

    bool hasExtent = false;
    for (int d = 0; d < 6; ++d) {
        if (pieceExtent[d] != 0) { hasExtent = true; break; }
    }
    if (hasExtent) {
        i0 = pieceExtent[0]; i1 = pieceExtent[1];
        j0 = pieceExtent[2]; j1 = pieceExtent[3];
        k0 = pieceExtent[4]; k1 = pieceExtent[5];
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
         << i0 << " " << i1 << " "
         << j0 << " " << j1 << " "
         << k0 << " " << k1 << "\">\n";
    file << "    <Piece Extent=\""
         << i0 << " " << i1 << " "
         << j0 << " " << j1 << " "
         << k0 << " " << k1 << "\">\n";

    // Coordinates
    file << "      <Coordinates>\n";

    // X coordinates
    file << "        <DataArray type=\"Float64\" Name=\"X\" format=\"ascii\">\n";
    file << "         ";
    for (int i = i0; i <= i1; ++i) {
        file << " " << mesh.nodeX(i);
    }
    file << "\n        </DataArray>\n";

    // Y coordinates
    file << "        <DataArray type=\"Float64\" Name=\"Y\" format=\"ascii\">\n";
    file << "         ";
    if (mesh.ny() == 1) {
        // 1D case: write a small y width
        file << " 0.0 " << mesh.dx(0);
    } else {
        for (int j = j0; j <= j1; ++j) {
            file << " " << mesh.nodeY(j);
        }
    }
    file << "\n        </DataArray>\n";

    // Z coordinates
    file << "        <DataArray type=\"Float64\" Name=\"Z\" format=\"ascii\">\n";
    file << "         ";
    if (mesh.nz() == 1) {
        // 2D or 1D case: write a small z depth
        file << " 0.0 " << std::min(mesh.dx(0), std::min(mesh.dy(0), 1.0));
    } else {
        for (int k = k0; k <= k1; ++k) {
            file << " " << mesh.nodeZ(k);
        }
    }
    file << "\n        </DataArray>\n";

    file << "      </Coordinates>\n";

    // Cell data
    file << "      <CellData>\n";

    // Helper lambda: write a scalar field
    auto writeScalar = [&](const std::string& name, const std::vector<double>& field) {
        file << "        <DataArray type=\"Float64\" Name=\"" << name
             << "\" format=\"ascii\">\n";
        for (int k = k0; k < k1; ++k) {
            for (int j = j0; j < j1; ++j) {
                file << "         ";
                for (int i = i0; i < i1; ++i) {
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
        for (int k = k0; k < k1; ++k) {
            for (int j = j0; j < j1; ++j) {
                file << "         ";
                for (int i = i0; i < i1; ++i) {
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
    writeScalar("Density", state.rho);
    writeScalar("Pressure", state.pres);
    writeScalar("Temperature", state.temp);
    writeScalar("Sigma", state.sigma);
    writeScalar("TotalEnergy", state.rhoE);

    // Vector fields
    writeVector("Velocity", state.velU, state.velV, state.velW);
    writeVector("Momentum", state.rhoU, state.rhoV, state.rhoW);

    file << "      </CellData>\n";
    file << "    </Piece>\n";
    file << "  </RectilinearGrid>\n";
    file << "</VTKFile>\n";

    file.close();
}

void VTKWriter::writePVTR(const std::string& filename,
                          const RectilinearMesh& mesh,
                          const SolutionState& /*state*/,
                          const std::vector<std::array<int,6>>& pieceExtents,
                          const std::vector<std::string>& pieceFiles)
{
    ensureParentDir(filename);
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("VTKWriter::writePVTR: cannot open " + filename);
    }

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PRectilinearGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    file << "  <PRectilinearGrid WholeExtent=\"0 " << mesh.nx()
         << " 0 " << mesh.ny()
         << " 0 " << mesh.nz() << "\" GhostLevel=\"0\">\n";

    // Declare coordinate arrays
    file << "    <PCoordinates>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"X\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Y\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Z\"/>\n";
    file << "    </PCoordinates>\n";

    // Declare cell data arrays
    file << "    <PCellData>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Density\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Pressure\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Temperature\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Sigma\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"TotalEnergy\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\"/>\n";
    file << "      <PDataArray type=\"Float64\" Name=\"Momentum\" NumberOfComponents=\"3\"/>\n";
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
        file.close();
    } else if (mode == "a") {
        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) {
            throw std::runtime_error("VTKWriter::writePVD: cannot open " + filename);
        }
        file << std::setprecision(15);
        file << "    <DataSet timestep=\"" << time
             << "\" file=\"" << dataFile << "\"/>\n";
        file.close();
    } else if (mode == "close") {
        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) {
            throw std::runtime_error("VTKWriter::writePVD: cannot open " + filename);
        }
        file << "  </Collection>\n";
        file << "</VTKFile>\n";
        file.close();
    }
}

} // namespace SemiImplicitFV

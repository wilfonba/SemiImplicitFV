#ifndef IMMERSED_BOUNDARY_HPP
#define IMMERSED_BOUNDARY_HPP

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include <vector>
#include <array>
#include <memory>
#include <cstdint>

namespace SemiImplicitFV {

// Cell classification for immersed boundary method
enum class CellType : uint8_t { Fluid, Ghost, Dead };

// Abstract base class for immersed body shapes
class IBBody {
public:
    enum class WallType { Slip, NoSlip };
    virtual ~IBBody() = default;

    // Negative inside body, positive outside
    virtual double signedDistance(double x, double y, double z) const = 0;
    // Nearest point on the body surface
    virtual std::array<double,3> closestPoint(double x, double y, double z) const = 0;
    // Outward unit normal at the closest surface point
    virtual std::array<double,3> outwardNormal(double x, double y, double z) const = 0;

    WallType wallType() const { return wallType_; }
    void setWallType(WallType wt) { wallType_ = wt; }

private:
    WallType wallType_ = WallType::NoSlip;
};

// 2D circle centered at (cx, cy) with radius r
class IBCircle : public IBBody {
public:
    IBCircle(double cx, double cy, double r);
    double signedDistance(double x, double y, double z) const override;
    std::array<double,3> closestPoint(double x, double y, double z) const override;
    std::array<double,3> outwardNormal(double x, double y, double z) const override;

private:
    double cx_, cy_, r_;
};

// 2D axis-aligned rectangle centered at (cx, cy) with half-widths (hw, hh)
class IBRectangle : public IBBody {
public:
    IBRectangle(double cx, double cy, double hw, double hh);
    double signedDistance(double x, double y, double z) const override;
    std::array<double,3> closestPoint(double x, double y, double z) const override;
    std::array<double,3> outwardNormal(double x, double y, double z) const override;

private:
    double cx_, cy_, hw_, hh_;
};

// 3D infinite cylinder along one axis (0=x, 1=y, 2=z)
// c0, c1 are the center coords in the cross-section plane
class IBCylinder : public IBBody {
public:
    IBCylinder(double c0, double c1, double r, int axis = 2);
    double signedDistance(double x, double y, double z) const override;
    std::array<double,3> closestPoint(double x, double y, double z) const override;
    std::array<double,3> outwardNormal(double x, double y, double z) const override;

private:
    double c0_, c1_, r_;
    int axis_;
};

// 3D axis-aligned rectangular prism centered at (cx, cy, cz) with half-widths (hw, hh, hd)
class IBRectangularPrism : public IBBody {
public:
    IBRectangularPrism(double cx, double cy, double cz,
                       double hw, double hh, double hd);
    double signedDistance(double x, double y, double z) const override;
    std::array<double,3> closestPoint(double x, double y, double z) const override;
    std::array<double,3> outwardNormal(double x, double y, double z) const override;

private:
    double cx_, cy_, cz_, hw_, hh_, hd_;
};

// Precomputed interpolation data for a single ghost cell
struct GhostCellInfo {
    std::size_t cellIdx;                   // flat index of the ghost cell
    int bodyIdx;                           // which body owns this ghost cell
    std::array<double,3> normal;           // outward normal at body intercept
    std::vector<std::size_t> interpCells;  // indices of surrounding fluid cells
    std::vector<double> interpWeights;     // corresponding weights
};

// Immersed boundary method: classifies cells, precomputes interpolation stencils,
// and fills ghost cell states to enforce wall BCs on embedded bodies.
class ImmersedBoundaryMethod {
public:
    void addBody(std::shared_ptr<IBBody> body);

    // Call once after mesh creation -- classifies cells and precomputes interpolation
    void classifyCells(const RectilinearMesh& mesh);

    // Fill ghost cell primitive states (velocity reflected, scalars extrapolated)
    void applyGhostCells(const RectilinearMesh& mesh, SolutionState& state, int dim) const;

    // Fill ghost cells for a single scalar field (pressure, sigma)
    void applyScalarGhostCells(const RectilinearMesh& mesh,
                               std::vector<double>& field) const;

    bool isSolid(std::size_t idx) const { return cellType_[idx] != CellType::Fluid; }
    CellType cellType(std::size_t idx) const { return cellType_[idx]; }
    const std::vector<CellType>& cellTypes() const { return cellType_; }
    bool hasIBM() const { return !bodies_.empty(); }

private:
    std::vector<std::shared_ptr<IBBody>> bodies_;
    std::vector<CellType> cellType_;
    std::vector<GhostCellInfo> ghostCells_;

    // Find the lower cell index such that centroid(i) <= coord < centroid(i+1)
    static int findCell(const RectilinearMesh& mesh, double coord, int dir);

    // Build bilinear/trilinear interpolation stencil for an image point
    void buildInterpStencil(const RectilinearMesh& mesh,
                            double xIP, double yIP, double zIP,
                            int dim,
                            std::vector<std::size_t>& cells,
                            std::vector<double>& weights) const;
};

} // namespace SemiImplicitFV

#endif // IMMERSED_BOUNDARY_HPP

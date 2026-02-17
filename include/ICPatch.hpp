#ifndef IC_PATCH_HPP
#define IC_PATCH_HPP

#include "SimulationConfig.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "EquationOfState.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace SemiImplicitFV {

class ExpressionEvaluator;

// -----------------------------------------------------------------
//  ICState: primitive state for initial conditions
// -----------------------------------------------------------------
struct ICState {
    double rho = 1.0;
    double u   = 0.0;
    double v   = 0.0;
    double w   = 0.0;
    double p   = 1.0;
    double sigma = 0.0;

    // Multi-phase fields (empty for single-phase)
    std::vector<double> alpha;     // volume fractions [nPhases]
    std::vector<double> alphaRho;  // partial densities [nPhases]
};

// -----------------------------------------------------------------
//  ICGeometry: abstract base for patch region definition
// -----------------------------------------------------------------
class ICGeometry {
public:
    virtual ~ICGeometry() = default;

    /// Returns true if the point (x,y,z) is inside this geometry.
    virtual bool contains(double x, double y, double z) const = 0;

    /// Returns signed distance to the boundary (negative inside, positive outside).
    /// Used for diffuse-interface blending.
    virtual double signedDistance(double x, double y, double z) const = 0;

    /// Clone for value semantics via unique_ptr.
    virtual std::unique_ptr<ICGeometry> clone() const = 0;
};

// -----------------------------------------------------------------
//  BoxGeometry
// -----------------------------------------------------------------
class BoxGeometry : public ICGeometry {
public:
    BoxGeometry(const std::array<double,3>& lo, const std::array<double,3>& hi)
        : lo_(lo), hi_(hi) {}

    bool contains(double x, double y, double z) const override {
        return x >= lo_[0] && x <= hi_[0]
            && y >= lo_[1] && y <= hi_[1]
            && z >= lo_[2] && z <= hi_[2];
    }

    double signedDistance(double x, double y, double z) const override {
        // Positive outside, negative inside
        double dx = std::max({lo_[0] - x, x - hi_[0], 0.0});
        double dy = std::max({lo_[1] - y, y - hi_[1], 0.0});
        double dz = std::max({lo_[2] - z, z - hi_[2], 0.0});
        double outsideDist = std::sqrt(dx*dx + dy*dy + dz*dz);

        // Inside distance (negative)
        double ix = std::min(x - lo_[0], hi_[0] - x);
        double iy = std::min(y - lo_[1], hi_[1] - y);
        double iz = std::min(z - lo_[2], hi_[2] - z);
        double insideDist = std::min({ix, iy, iz});

        return (outsideDist > 0.0) ? outsideDist : -insideDist;
    }

    std::unique_ptr<ICGeometry> clone() const override {
        return std::make_unique<BoxGeometry>(*this);
    }

private:
    std::array<double,3> lo_, hi_;
};

// -----------------------------------------------------------------
//  SphereGeometry
// -----------------------------------------------------------------
class SphereGeometry : public ICGeometry {
public:
    SphereGeometry(const std::array<double,3>& center, double radius)
        : center_(center), radius_(radius) {}

    bool contains(double x, double y, double z) const override {
        double dx = x - center_[0], dy = y - center_[1], dz = z - center_[2];
        return (dx*dx + dy*dy + dz*dz) <= radius_ * radius_;
    }

    double signedDistance(double x, double y, double z) const override {
        double dx = x - center_[0], dy = y - center_[1], dz = z - center_[2];
        return std::sqrt(dx*dx + dy*dy + dz*dz) - radius_;
    }

    std::unique_ptr<ICGeometry> clone() const override {
        return std::make_unique<SphereGeometry>(*this);
    }

private:
    std::array<double,3> center_;
    double radius_;
};

// -----------------------------------------------------------------
//  PlaneGeometry: inside if dot(cellCenter - point, normal) > 0
// -----------------------------------------------------------------
class PlaneGeometry : public ICGeometry {
public:
    PlaneGeometry(const std::array<double,3>& point, const std::array<double,3>& normal)
        : point_(point), normal_(normal) {
        // Normalize
        double mag = std::sqrt(normal_[0]*normal_[0] + normal_[1]*normal_[1] + normal_[2]*normal_[2]);
        if (mag > 0.0) {
            normal_[0] /= mag; normal_[1] /= mag; normal_[2] /= mag;
        }
    }

    bool contains(double x, double y, double z) const override {
        return dot(x, y, z) > 0.0;
    }

    double signedDistance(double x, double y, double z) const override {
        return -dot(x, y, z);  // positive outside (where dot < 0)
    }

    std::unique_ptr<ICGeometry> clone() const override {
        return std::make_unique<PlaneGeometry>(*this);
    }

private:
    double dot(double x, double y, double z) const {
        return (x - point_[0]) * normal_[0]
             + (y - point_[1]) * normal_[1]
             + (z - point_[2]) * normal_[2];
    }
    std::array<double,3> point_, normal_;
};

// -----------------------------------------------------------------
//  AnalyticGeometry: uses a sub-geometry for region + expressions
// -----------------------------------------------------------------
class AnalyticGeometry : public ICGeometry {
public:
    explicit AnalyticGeometry(std::unique_ptr<ICGeometry> region)
        : region_(std::move(region)) {}

    AnalyticGeometry(const AnalyticGeometry& other)
        : region_(other.region_ ? other.region_->clone() : nullptr) {}

    bool contains(double x, double y, double z) const override {
        return region_ ? region_->contains(x, y, z) : true;
    }

    double signedDistance(double x, double y, double z) const override {
        return region_ ? region_->signedDistance(x, y, z) : -1.0;
    }

    std::unique_ptr<ICGeometry> clone() const override {
        return std::make_unique<AnalyticGeometry>(*this);
    }

    const ICGeometry* region() const { return region_.get(); }

private:
    std::unique_ptr<ICGeometry> region_;
};

// -----------------------------------------------------------------
//  DiffuseInterface parameters
// -----------------------------------------------------------------
struct DiffuseInterface {
    double diffuseWidth = 0.0;   // in cells (0 = sharp)
    std::string profile = "tanh"; // "tanh" or "linear"
};

// -----------------------------------------------------------------
//  ICPatch: geometry + state + optional expressions + diffuse interface
// -----------------------------------------------------------------
struct ICPatch {
    std::unique_ptr<ICGeometry> geometry;
    ICState state;
    DiffuseInterface interface;

    // For analytic patches: map from field name to expression string
    // e.g., {"rho": "1.0 + 0.5*sin(2*pi*x)", "p": "1.0"}
    std::map<std::string, std::string> expressions;

    ICPatch() = default;
    ICPatch(ICPatch&& other) = default;
    ICPatch& operator=(ICPatch&& other) = default;

    // Copy constructor (deep copies geometry)
    ICPatch(const ICPatch& other)
        : geometry(other.geometry ? other.geometry->clone() : nullptr),
          state(other.state), interface(other.interface),
          expressions(other.expressions) {}
};

// -----------------------------------------------------------------
//  Apply initial conditions to the mesh
// -----------------------------------------------------------------

/// Fill all cells with defaultState, then apply patches in order.
/// For multi-phase: computes rhoE via MixtureEOS or single-phase EOS.
void applyInitialConditions(
    const RectilinearMesh& mesh,
    SolutionState& state,
    const EquationOfState& eos,
    const SimulationConfig& config,
    const ICState& defaultState,
    const std::vector<ICPatch>& patches);

} // namespace SemiImplicitFV

#endif // IC_PATCH_HPP

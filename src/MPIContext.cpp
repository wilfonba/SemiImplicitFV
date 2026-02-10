#include "MPIContext.hpp"
#include <stdexcept>
#include <algorithm>

namespace SemiImplicitFV {

MPIContext::~MPIContext() {
    if (cartComm_ != MPI_COMM_NULL && cartComm_ != MPI_COMM_WORLD) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Comm_free(&cartComm_);
        }
    }
}

MPIContext::MPIContext(MPIContext&& other) noexcept
    : cartComm_(other.cartComm_)
    , rank_(other.rank_)
    , size_(other.size_)
    , dims_(other.dims_)
    , coords_(other.coords_)
    , neighbors_(other.neighbors_)
    , localXNodes_(std::move(other.localXNodes_))
    , localYNodes_(std::move(other.localYNodes_))
    , localZNodes_(std::move(other.localZNodes_))
    , localNx_(other.localNx_)
    , localNy_(other.localNy_)
    , localNz_(other.localNz_)
    , localExtent_(other.localExtent_)
{
    other.cartComm_ = MPI_COMM_NULL;
}

MPIContext& MPIContext::operator=(MPIContext&& other) noexcept {
    if (this != &other) {
        if (cartComm_ != MPI_COMM_NULL && cartComm_ != MPI_COMM_WORLD) {
            MPI_Comm_free(&cartComm_);
        }
        cartComm_ = other.cartComm_;
        rank_ = other.rank_;
        size_ = other.size_;
        dims_ = other.dims_;
        coords_ = other.coords_;
        neighbors_ = other.neighbors_;
        localXNodes_ = std::move(other.localXNodes_);
        localYNodes_ = std::move(other.localYNodes_);
        localZNodes_ = std::move(other.localZNodes_);
        localNx_ = other.localNx_;
        localNy_ = other.localNy_;
        localNz_ = other.localNz_;
        localExtent_ = other.localExtent_;
        other.cartComm_ = MPI_COMM_NULL;
    }
    return *this;
}

std::vector<double> MPIContext::sliceNodes(const std::vector<double>& globalNodes,
                                            int globalN, int nProcs, int coord) {
    // Distribute globalN cells across nProcs processors.
    // Each proc gets base = globalN / nProcs cells, with the first
    // (globalN % nProcs) procs getting one extra cell.
    int base = globalN / nProcs;
    int remainder = globalN % nProcs;

    int startCell = 0;
    for (int p = 0; p < coord; ++p) {
        startCell += base + (p < remainder ? 1 : 0);
    }
    int localN = base + (coord < remainder ? 1 : 0);

    // Extract local node coordinates: localN+1 nodes
    std::vector<double> localNodes(localN + 1);
    for (int i = 0; i <= localN; ++i) {
        localNodes[i] = globalNodes[startCell + i];
    }
    return localNodes;
}

MPIContext MPIContext::create(int globalNx, int globalNy, int globalNz,
                              const std::vector<double>& xNodes,
                              const std::vector<double>& yNodes,
                              const std::vector<double>& zNodes,
                              int dim,
                              const std::array<int,3>& periods,
                              const std::array<int,3>& procsPerDir) {
    MPIContext ctx;

    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Set up dimensions for Cartesian topology
    ctx.dims_ = procsPerDir;

    // For inactive dimensions, force 1 process
    if (dim < 2) ctx.dims_[1] = 1;
    if (dim < 3) ctx.dims_[2] = 1;

    // Let MPI fill in zeros with automatic decomposition
    // Only auto-decompose active dimensions
    if (dim == 1) {
        if (ctx.dims_[0] == 0) ctx.dims_[0] = worldSize;
    } else if (dim == 2) {
        // Count how many active dims need auto-assignment
        int autoDims = 0;
        for (int d = 0; d < 2; ++d) {
            if (ctx.dims_[d] == 0) autoDims++;
        }
        if (autoDims > 0) {
            // Use MPI_Dims_create for the active dimensions
            std::array<int,2> activeDims = {ctx.dims_[0], ctx.dims_[1]};
            MPI_Dims_create(worldSize, 2, activeDims.data());
            ctx.dims_[0] = activeDims[0];
            ctx.dims_[1] = activeDims[1];
        }
    } else {
        // 3D: let MPI auto-decompose all non-specified dims
        MPI_Dims_create(worldSize, 3, ctx.dims_.data());
    }

    // Verify the product matches worldSize
    int product = ctx.dims_[0] * ctx.dims_[1] * ctx.dims_[2];
    if (product != worldSize) {
        throw std::runtime_error(
            "MPIContext::create: dims product (" + std::to_string(product) +
            ") != worldSize (" + std::to_string(worldSize) + ")");
    }

    // Create Cartesian communicator (allow reordering for topology optimization)
    int mpiPeriods[3] = {periods[0], periods[1], periods[2]};
    MPI_Cart_create(MPI_COMM_WORLD, 3, ctx.dims_.data(), mpiPeriods, 1, &ctx.cartComm_);

    MPI_Comm_rank(ctx.cartComm_, &ctx.rank_);
    MPI_Comm_size(ctx.cartComm_, &ctx.size_);
    MPI_Cart_coords(ctx.cartComm_, ctx.rank_, 3, ctx.coords_.data());

    // Get neighbors via MPI_Cart_shift (returns MPI_PROC_NULL for non-periodic boundaries)
    MPI_Cart_shift(ctx.cartComm_, 0, 1, &ctx.neighbors_[XLow],  &ctx.neighbors_[XHigh]);
    MPI_Cart_shift(ctx.cartComm_, 1, 1, &ctx.neighbors_[YLow],  &ctx.neighbors_[YHigh]);
    MPI_Cart_shift(ctx.cartComm_, 2, 1, &ctx.neighbors_[ZLow],  &ctx.neighbors_[ZHigh]);

    // Slice global node arrays into local pieces
    ctx.localXNodes_ = sliceNodes(xNodes, globalNx, ctx.dims_[0], ctx.coords_[0]);
    ctx.localYNodes_ = sliceNodes(yNodes, globalNy, ctx.dims_[1], ctx.coords_[1]);
    ctx.localZNodes_ = sliceNodes(zNodes, globalNz, ctx.dims_[2], ctx.coords_[2]);

    ctx.localNx_ = static_cast<int>(ctx.localXNodes_.size()) - 1;
    ctx.localNy_ = static_cast<int>(ctx.localYNodes_.size()) - 1;
    ctx.localNz_ = static_cast<int>(ctx.localZNodes_.size()) - 1;

    // Compute global cell extent for VTK output
    auto computeStart = [](int globalN, int nProcs, int coord) {
        int base = globalN / nProcs;
        int remainder = globalN % nProcs;
        int start = 0;
        for (int p = 0; p < coord; ++p) {
            start += base + (p < remainder ? 1 : 0);
        }
        return start;
    };

    int i0 = computeStart(globalNx, ctx.dims_[0], ctx.coords_[0]);
    int j0 = computeStart(globalNy, ctx.dims_[1], ctx.coords_[1]);
    int k0 = computeStart(globalNz, ctx.dims_[2], ctx.coords_[2]);
    ctx.localExtent_ = {i0, i0 + ctx.localNx_,
                        j0, j0 + ctx.localNy_,
                        k0, k0 + ctx.localNz_};

    return ctx;
}

} // namespace SemiImplicitFV

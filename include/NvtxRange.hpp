#ifndef SIFV_NVTX_RANGE_HPP
#define SIFV_NVTX_RANGE_HPP

#ifdef SIFV_HAS_NVTX
#include <nvtx3/nvToolsExt.h>

#define SIFV_NVTX_PUSH(name) nvtxRangePushA(name)
#define SIFV_NVTX_POP()      nvtxRangePop()

namespace SemiImplicitFV {

class NvtxRange {
public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

} // namespace SemiImplicitFV

#else // SIFV_HAS_NVTX not defined

#define SIFV_NVTX_PUSH(name) ((void)0)
#define SIFV_NVTX_POP()      ((void)0)

namespace SemiImplicitFV {

class NvtxRange {
public:
    explicit NvtxRange(const char*) {}
};

} // namespace SemiImplicitFV

#endif // SIFV_HAS_NVTX

#endif // SIFV_NVTX_RANGE_HPP

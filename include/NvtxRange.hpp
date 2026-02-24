#ifndef NVTX_RANGE_HPP
#define NVTX_RANGE_HPP

#ifdef SIFV_HAS_NVTX
#include <nvtx3/nvToolsExt.h>

#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP()      nvtxRangePop()

#else /* SIFV_HAS_NVTX not defined */

#define NVTX_PUSH(name) ((void)0)
#define NVTX_POP()      ((void)0)

#endif /* SIFV_HAS_NVTX */

#endif /* NVTX_RANGE_HPP */

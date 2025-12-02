/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <iostream>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#include <rocwmma/rocwmma_coop.hpp>

#include <cstdio>
#include "common.hpp"
// #include <stdio.h>
/* Motivation
*
* For this particular GEMM kernel, high performance can be
* achieved through two general principles:
* 1) Data re-use
* 2) Latency hiding
*
* From the simple_gemm implementation, we know that the GEMM
* equation takes the form:
*
* D = alpha * AxB + beta * C, where
*
* A, B = input tiles of MxK and KxN, respectively
* C = input tile of MxN and
* D = final output tile, MxN
* alpha, beta are scalar factors
* (M, N and K are block dimensions)
*
* In the simple_gemm sample, each warp is responsible for computing
* one output D tile of the final result. In the current sample, each
* warp is now responsible for computing multiple D tiles, what we
* might call a Warp Tile. Because Warp Tile blocks share data locality
* in either the same row or column direction, warps can re-use input
* data from A and B as they step through the K dimension for each block.
*
* Moreover, Warp Tiles processed by warps in a thread block
* have common locality in the larger Macro Tile. In the Global D layout
* shown below, data re-use opportunities await in D tiles aligned in the
* same rows / columns. These will pass over the same input A/B values as
* they march through the K dimension.
*
* Block size:      (BlockM x BlockN)
* Warp tile size:  (BlocksX * BlockSize.x) x (BlocksY * BlockSize.y)
* Macro Tile size: (TBlock.x * WarpTileSize.x) x (TBlock.y * WarpTileSize.y)
*
* Wave data share input A: same row
* Wave data share input B: same col
*
* Global D layout & warp assignment for BlocksX = BlocksY = 2, 2x2 Warps
*
* W (X, Y) = wave row X, col Y
*                                     |--------- Macro Tile Y-------------|
*                                     |-- Wave Tile Y --|
*                                     |-BlockN-|
*
*                                      BlockN x BlocksY   BlockN x BlocksY
*                                     |<--------------->|<--------------->|
*      _ _   _ _      _ _          ___  ________ ________ ________ ________
*       |     |        |            ^  |        |        |        |        |
*       | Wave| BlockM |   BlockM   |  |        W        |        W        |
*       | Tile|       _|_     x     |  |__   (0, 0)    __|__   (0, 1)    __|
*       |  X  |            BlocksX  |  |                 |                 |
* Macro |     |                     |  |                 |                 |
*  Tile |    _|_                   _v_ |________|________|________|________|
*   X   |                           ^  |        |        |        |        |
*       |                  BlockM   |  |        W        |        W        |
*       |                     x     |  |__   (1, 0)    __|__   (1, 1)    __|
*       |                  BlocksX  |  |                 |                 |
*       |                           |  |                 |                 |
*      _|_                         _v_ |________|________|________|________|
*
*
* From the above diagram, we can see that input A/B data can be shared within warps,
* as well as between warps in the same threadblock. This means that warps in the same
* thread block can share the input loading responsibilities if they synchronize stepping
* through the K dimension for tiles at the same time.
*
* rocWMMA Cooperative API allows thread blocks to collaboratively move data from
* one location to another. In this case, we will move data from global memory space to
* local storage such that inter-warp data sharing is possible. Maximizing data re-use
* in this way reduces costly access to global memory and improves performance.
*
* To maximize efficiency, we can structure the kernel to maximize bandwidth usage and keep
* the compute resources as busy as possible at the same time. Using a pre-fetch technique,
* we can fetch A/B inputs for the next K-step while keeping the compute resources busy
* processing the current K-step. This helps to hide memory fetching latency.
*
* In general, the process would flow like the following:
*
*       Start
*         |
*   Pre-Fetch Global A/B for K0
*         |
*   Store LDS buffer0
*         |
*         v
*   Loop: i = 1:K-1
*   ^         |
*   |    Fetch Global A/B i+1; store LDS Buffer 1
*   |         |
*   |    Load LDS buffer0; Accum A x B
*   |         |
*   |    Swap buffer0, buffer1
*   |         |
*   |         |
*   end_loop <-
*         |
*   Load LDS buffer0; Accum A x B
*         |
*   Load Global C Tile
*         |
*   D = alpha * AccumAB + beta * C
*         |
*   Write D Tile
*         |
*         v
*        End
*
* Lds Mapping
* Buffer Width = LDS Width = BlockK
* Matrix geometry for inputs A and B have a common dimension (BlockK).
* We can fix one of the LDS dimensions to BlockK (in this case the width),
* and insert blocks of different heights (BlockM, BlockN) to use the space
* without the need of extra padding.
*
* Fragments of B must be transposed to fit this geometry,
* and both fragments from A and B must accomodate LDS data layout.
*
* Local Layout (LDS):
*
* Non - transposed A fragments [A0 ... AX-1] are placed first and occupy a total height
* of Macro Tile X, where X = number of A blocks and Ck is the kth column of the A block.
*
* Transposed B fragments [B0 (T) ... BY-1 (T)] follow A fragments and occupy a total height of
* Macro Tile Y, where Y = number of B blocks, and Rk is the kth row of the B block.
*
*
*                        _____________BlockK_____________
*                       |                                |
*                       v                                v
*                  (0,0) ----------------------------------->
*          -->       -->  ______________    ...        ______
*          |         |   |    |    |                  |      |
*          |         |   |    |    |                  |      |
*  Macro   |  BlockM |   | C0 | C1 | C2               | Ck-1 |   A0
*  Tile X  |         |   |    |    |                  |      |
*          |         --> |___ |___ |____    ...       |______|
*          |         .
*          |         .          ...  ...  ...  ...          AX-1
*          -->
*          -->       -->  ______________    ...        ______
*          |         |   |    |    |                  |      |
*          |         |   |    |    |                  |      |
*  Macro   |  BlockN |   | R0 | R1 | R2               | Rk-1 |   B0 (T)
*  Tile Y  |         |   |    |    |                  |      |
*          |         --> |___ |___ |____    ...       |______|
*          |         .
*          |         .          ...  ...  ...  ...        BY-1 (T)
*          -->                                           (MacroTileX + MacroTileY - 1, BlockK -1)
*
* Depending on the locality of the block being processed, warps load the corresponding
* A and B inputs from LDS buffer and use them for the accumulation of AxB calculations.
*/

using namespace rocwmma;

///
/// Parameter configuration
///

/* Depending on the GPU architecture this sample is run on, the following kernel parameters need to
*  be modified in order to obtain high performance.
* _________________________________________________________________________________________
*|         |           |           |           |          |          |          |          |
*|         | ROCWMMA_M | ROCWMMA_N | ROCWMMA_K | BLOCKS_X | BLOCKS_Y | TBLOCK_X | TBLOCK_Y |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*|         |           |           |           |          |          |          |          |
*|  GFX_9  |    32     |    32     |    16     |    2     |    2     |   128    |    2     |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*|         |           |           |           |          |          |          |          |
*|  GFX_11 |    16     |    16     |    16     |    4     |    2     |    64    |    4     |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*
* __________________________________________
*|         |                                |
*|         |           WARP_SIZE            |
*|_________|________________________________|
*|         |                                |
*|  GFX_9  | Constants::AMDGCN_WAVE_SIZE_64 |
*|_________|________________________________|
*|         |                                |
*|  GFX_11 | Constants::AMDGCN_WAVE_SIZE_32 |
*|_________|________________________________|
*/

namespace gfx9Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 32u,
        BLOCKS_X  = 1u,
        BLOCKS_Y  = 4u,
        TBLOCK_X  = 256u,
        TBLOCK_Y  = 1u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64
    };
}

namespace gfx11Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 16u,
        BLOCKS_X  = 4u,
        BLOCKS_Y  = 2u,
        TBLOCK_X  = 64u,
        TBLOCK_Y  = 4u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_32
    };
}

enum class DataType {
    kHalf,
    kInt8,
    kInt4,
    kE4M3,
    kE5M2,
};

enum class QuantGranularity {
    kPerTensor = 0,
    kPerBlock = 1,
    kPerWarp = 2,
    kPerThread = 3,
};

enum class MaskMode {
    kNone = 0,
    kCausal = 1,
};

enum class ComputeUnit {
  kTensorCore,
  kCudaCore,
};

constexpr float log2e = 1.44269504088896340736f;
constexpr float log2e_recp = 1.0f / log2e;
#define div_ceil(M, N) (((M) + (N)-1) / (N))
#define TRAP_IF(cond) do { if (cond) asm volatile("s_trap 0"); } while(0)

#if(ROCWMMA_ARCH_GFX9)
using namespace gfx9Params;
#else
using namespace gfx11Params;
#endif // defined(ROCWMMA_ARCH_GFX9)

///
/// Types and Data Layouts
///

using InputT   = int8_t;
using InputTV   = float8_fnuz_t;
using OutputT  = int32_t;
using ComputeT = int32_t;
using ComputeTV = float32_t;
using LDST = float32_t;
using LDST_new = float8_fnuz_t;

using DataLayoutA   = row_major;
using DataLayoutB   = col_major;
using DataLayoutC   = row_major;
using DataLayoutLds = col_major;
using DataLayoutS   = row_major;
using DataLayoutV   = col_major;
// can't modify
using DataLayoutLds_new = row_major;
///
/// Fragment types
///

// #if (ROCWMMA_ARCH_GFX9 || ROCWMMA_ARCH_GFX11)
// Warp tile: computed by each warp
constexpr uint32_t WARP_TILE_X = BLOCKS_X * ROCWMMA_M;
constexpr uint32_t WARP_TILE_Y = BLOCKS_Y * ROCWMMA_N;

constexpr uint32_t els_per_thread = ROCWMMA_M * ROCWMMA_N / WARP_SIZE;
constexpr uint32_t threads_per_row = ROCWMMA_N / els_per_thread;

// Macro Tile: computed by each thread block (workgroup)
// Note: TBLOCK_X must be multiple of WARP_SIZE.
constexpr uint32_t WARPS_X      = TBLOCK_X / WARP_SIZE;
constexpr uint32_t WARPS_Y      = TBLOCK_Y;
constexpr uint32_t MACRO_TILE_X = WARPS_X * WARP_TILE_X;
constexpr uint32_t MACRO_TILE_Y = WARPS_Y * WARP_TILE_Y;
constexpr uint32_t MACRO_TILE_K = WARPS_Y * BLOCKS_Y * ROCWMMA_K;

// Mfma frags
using MfmaFragA   = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA>;
using MfmaFragB   = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutB>;
using MfmaFragC   = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, OutputT, DataLayoutC>;
using MfmaFragD   = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float, DataLayoutC>;
using MfmaFragT   = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, int, DataLayoutC>;
using MfmaFragAcc = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT,DataLayoutC>;
using MfmaFragAccf32 = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeTV,DataLayoutC>;

using MfmaFragF8 = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, MACRO_TILE_Y, hip_fp8_e4m3_fnuz,DataLayoutC>;

using MfmaFragAcc2 = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputTV,DataLayoutS>;
using MfmaFragAcc2_1d = fragment<accumulator, ROCWMMA_M,ROCWMMA_N, MACRO_TILE_Y, OutputT,DataLayoutC>;

using MfmaFragS   = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputTV, DataLayoutS>;
using MfmaFragV   = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputTV, DataLayoutV>;

// Global read (macro tile)
using GRBuffA = fragment<matrix_a, MACRO_TILE_X, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA>;
using GRBuffB = fragment<matrix_b, ROCWMMA_M, MACRO_TILE_Y, ROCWMMA_K, InputT, DataLayoutB>;

using GRBuffS = fragment<matrix_a, MACRO_TILE_X, ROCWMMA_N, ROCWMMA_K, InputTV, DataLayoutS>;
using GRBuffV = fragment<matrix_b, ROCWMMA_M, MACRO_TILE_Y, ROCWMMA_K, InputTV, DataLayoutV>;

// Local write of global buffers (macro tile)
// - Must match Lds data layout.
// - Lds has transposed B frags.
using LWBuffA = ApplyDataLayout_t<GRBuffA, DataLayoutLds>;
using LWBuffB = ApplyDataLayout_t<ApplyTranspose_t<GRBuffB>, DataLayoutLds>;

using LWBuffS = ApplyDataLayout_t<GRBuffS, DataLayoutLds_new>;
using LWBuffV = ApplyDataLayout_t<ApplyTranspose_t<GRBuffV>, DataLayoutLds_new>;

// Local read (mfma frags)
// - Must match Lds data layout.
// - Lds has transposed B frags.
using LRFragA = ApplyDataLayout_t<MfmaFragA, DataLayoutLds>;
using LRFragB = ApplyDataLayout_t<ApplyTranspose_t<MfmaFragB>, DataLayoutLds>;

using LRFragTmp_1d = ApplyDataLayout_t<MfmaFragAcc2_1d, DataLayoutLds_new>;
using LRFragS = ApplyDataLayout_t<MfmaFragS, DataLayoutLds_new>;
using LRFragV = ApplyDataLayout_t<ApplyTranspose_t<MfmaFragV>, DataLayoutLds_new>;

using LRFragAccf32 = ApplyDataLayout_t<MfmaFragAccf32, DataLayoutLds_new>;
// #endif // (ROCWMMA_ARCH_GFX9 || ROCWMMA_ARCH_GFX11)

///
/// Wrapper functions: repeat mfma tile operations across entire warp tile.
///

// Cooperative global read / local write (Macro tile data movement)
// Loads / stores a global data fragment cooperatively across warps. Each participating warp is
// responsible for only a portion of the whole fragment.
//
// The cooperative operation is split into work items (SplitCount). Work items are consumed in
// a round robin fashion by warps in the range of [0, WaveCount). The wave index determines the
// order of the current wave in the collaboration pool.
//
// WaveCount, SplitCount and waveIndex parameters must match successive coop load / store calls
// to ensure the entire fragment remains coherent.

// Global A reads in cooperative mode (macro tile)
template <uint32_t WaveCountA>
ROCWMMA_DEVICE static inline void
    globalReadCoopA(GRBuffA& grBuffA, InputT const* gAddrA, uint32_t lda, uint32_t waveIndexA)
{
    load_matrix_coop_sync<WaveCountA>(grBuffA, gAddrA, lda, waveIndexA);
}

// Global B reads in cooperative mode (macro tile)
template <uint32_t WaveCountB>
ROCWMMA_DEVICE static inline void
    globalReadCoopB(GRBuffB& grBuffB, InputT const* gAddrB, uint32_t ldb, uint32_t waveIndexB)
{
    load_matrix_coop_sync<WaveCountB>(grBuffB, gAddrB, ldb, waveIndexB);
}

template <uint32_t WaveCountV>
ROCWMMA_DEVICE static inline void
    globalReadCoopV(GRBuffV& grBuffV, InputTV const* gAddrV, uint32_t ldV, uint32_t waveIndexV)
{
    load_matrix_coop_sync<WaveCountV>(grBuffV, gAddrV, ldV, waveIndexV);
}

// Local A writes in cooperative mode (macro tile)
template <uint32_t WaveCountA>
ROCWMMA_DEVICE static inline void
    localWriteCoopA(InputT* ldsAddr, GRBuffA const& grBuffA, uint32_t ldsld, uint32_t waveIndexA)
{
    // No transpose, but apply the lds data layout
    store_matrix_coop_sync<WaveCountA>(
        ldsAddr, applyDataLayout<DataLayoutLds, WaveCountA>(grBuffA), ldsld, waveIndexA);
}

// Local B writes in cooperative mode (macro tile)
template <uint32_t WaveCountB>
ROCWMMA_DEVICE static inline void
    localWriteCoopB(InputT* ldsAddr, GRBuffB const& grBuffB, uint32_t ldsld, uint32_t waveIndexB)
{
    // Transpose B and then apply lds data layout
    store_matrix_coop_sync<WaveCountB>(
        ldsAddr, applyDataLayout<DataLayoutLds, WaveCountB>(applyTranspose(grBuffB)), ldsld, waveIndexB);
}

template <uint32_t WaveCountV>
ROCWMMA_DEVICE static inline void
    localWriteCoopV(InputTV* ldsAddr, GRBuffV const& grBuffV, uint32_t ldsld, uint32_t waveIndexV)
{
    // Transpose B and then apply lds data layout
    store_matrix_coop_sync<WaveCountV>(
        ldsAddr, applyDataLayout<DataLayoutLds_new, WaveCountV>(applyTranspose(grBuffV)), ldsld, waveIndexV);
}

// Local A reads for warp tile gemm, non-cooperative
ROCWMMA_DEVICE static inline void
    localReadA(MfmaFragA (&fragsA)[BLOCKS_X], InputT const* ldsAddrA, uint32_t ldsld)
{
    using FragShape = GetIOShape_t<LRFragA>;
    using Mapper1d  = GetDataLayout_t<LRFragA>;

    // Each A block is stacked vertically in LDS
    auto blockStep = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        LRFragA tmp;
        load_matrix_sync(tmp, ldsAddrA, ldsld);
        fragsA[i] = applyDataLayout<DataLayoutA>(tmp);

        ldsAddrA += blockStep;
    }
}

ROCWMMA_DEVICE static inline void
    localReadS(MfmaFragS (&fragsS)[BLOCKS_X], InputTV const* ldsAddrS, uint32_t ldsld)
{
    using FragShape = GetIOShape_t<LRFragS>;
    using Mapper1d  = GetDataLayout_t<LRFragS>;

    // Each A block is stacked vertically in LDS
    auto blockStep = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        LRFragS tmp;
        load_matrix_sync(tmp, ldsAddrS, ldsld);
        fragsS[i] = applyDataLayout<DataLayoutA>(tmp);

        ldsAddrS += blockStep;
    }
}

// Local B reads for warp tile gemm, non-cooperative
ROCWMMA_DEVICE static inline void
    localReadB(MfmaFragB (&fragsB)[BLOCKS_Y], InputT const* ldsAddrB, uint32_t ldsld)
{
    using FragShape = GetIOShape_t<LRFragB>;
    using Mapper1d  = GetDataLayout_t<LRFragB>;

    // Each B block is stacked vertically in LDS
    auto blockStep = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);

#pragma unroll
    for(int i = 0; i < BLOCKS_Y; i++)
    {
        LRFragB tmp;
        load_matrix_sync(tmp, ldsAddrB, ldsld);

        // Transform back to MFMA tile
        fragsB[i] = applyDataLayout<DataLayoutB>(applyTranspose(tmp));

        ldsAddrB += blockStep;
    }
}

ROCWMMA_DEVICE static inline void
    localReadV(MfmaFragV (&fragsV)[BLOCKS_Y], InputTV const* ldsAddrV, uint32_t ldsld)
{
    using FragShape = GetIOShape_t<LRFragV>;
    using Mapper1d  = GetDataLayout_t<LRFragV>;

    // Each B block is stacked vertically in LDS
    auto blockStep = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);

#pragma unroll
    for(int i = 0; i < BLOCKS_Y; i++)
    {
        LRFragV tmp;
        load_matrix_sync(tmp, ldsAddrV, ldsld);

        // Transform back to MFMA tile
        fragsV[i] = applyDataLayout<DataLayoutV>(applyTranspose(tmp));

        ldsAddrV += blockStep;
    }
}

// Global D reads for warp tile gemm, non-cooperative
ROCWMMA_DEVICE static inline void
    globalWriteD(float* gAddrD, MfmaFragAccf32 const (&fragsD)[BLOCKS_X][BLOCKS_Y], uint32_t ldd)
{
    using FragShape = GetIOShape_t<MfmaFragD>;
    using Mapper1d  = GetDataLayout_t<MfmaFragD>;

    // Iterative offsets for each D block in the warp tile
    auto blockStepX = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldd);
    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldd);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            store_matrix_sync(gAddrD + offsetY, fragsD[i][j], ldd);
            // printf("offsetY:%u\n",offsetY);
            offsetY += blockStepY;
        }
        gAddrD += blockStepX;
    }
}

ROCWMMA_DEVICE static inline void
    globalWriteT(int* gAddrT, MfmaFragAcc const (&fragsT)[BLOCKS_X][BLOCKS_Y], uint32_t ldd)
{
    using FragShape = GetIOShape_t<MfmaFragT>;
    using Mapper1d  = GetDataLayout_t<MfmaFragT>;

    // Iterative offsets for each D block in the warp tile
    auto blockStepX = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldd);
    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldd);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            store_matrix_sync(gAddrT + offsetY, fragsT[i][j], ldd);
            // printf("offsetY:%u\n",offsetY);
            offsetY += blockStepY;
        }
        gAddrT += blockStepX;
    }
}

// Broadcast value to fragments in warp tile
template <typename FragT>
ROCWMMA_DEVICE static inline void fill(FragT (&frags)[BLOCKS_X][BLOCKS_Y],
                                       GetDataType_t<FragT> value)
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            fill_fragment(frags[i][j], value);
        }
    }
}

ROCWMMA_DEVICE static inline void
    localWriteAcc(MfmaFragAccf32 (&fragsAcc)[BLOCKS_X][BLOCKS_Y], ComputeTV * ldsAddrAcc, uint32_t ldsld)
{
    using FragShape = GetIOShape_t<LRFragAccf32>;
    using Mapper1d  = GetDataLayout_t<LRFragAccf32>;

    auto blockStepX = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldsld);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            store_matrix_sync(ldsAddrAcc + offsetY, fragsAcc[i][j], ldsld);
            offsetY += blockStepY;
        }
        ldsAddrAcc += blockStepX;
    }
}

ROCWMMA_DEVICE static inline void
    localWriteOut(MfmaFragAccf32 (&fragsAcc)[BLOCKS_Y], ComputeTV * ldsAddrAcc, uint32_t ldsld)
{
    using FragShape = GetIOShape_t<LRFragAccf32>;
    using Mapper1d  = GetDataLayout_t<LRFragAccf32>;

    // auto blockStepX = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldsld);

    auto offsetY = 0u;
#pragma unroll
    for(int j = 0; j < BLOCKS_Y; j++)
    {
        store_matrix_sync(ldsAddrAcc + offsetY, fragsAcc[j], ldsld);
        offsetY += blockStepY;
    }
}


ROCWMMA_DEVICE static inline void
    localReadOut(MfmaFragAccf32 (&fragsAcc)[BLOCKS_Y], ComputeTV const* ldsAddrAcc, uint32_t ldsld)
{
    using FragShape = GetIOShape_t<LRFragAccf32>;
    using Mapper1d  = GetDataLayout_t<LRFragAccf32>;

    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldsld);

    auto offsetY = 0u;
#pragma unroll
    for(int j = 0; j < BLOCKS_Y; j++)
    {
        load_matrix_sync(fragsAcc[j], ldsAddrAcc + offsetY, ldsld);
        offsetY += blockStepY;
    }
}

ROCWMMA_DEVICE static inline void
    localReadAcc(MfmaFragAccf32 (&fragsAcc)[BLOCKS_X][BLOCKS_Y], float const* ldsAddrAcc, uint32_t ldsld)
{
    using FragShape = GetIOShape_t<LRFragAccf32>;
    using Mapper1d  = GetDataLayout_t<LRFragAccf32>;

    auto blockStepX = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldsld);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            load_matrix_sync(fragsAcc[i][j], ldsAddrAcc + offsetY, ldsld);
            offsetY += blockStepY;
        }
        ldsAddrAcc += blockStepX;
    }
}


ROCWMMA_DEVICE static inline void
convertS32toF32(MfmaFragAcc  (&frags_i32)[BLOCKS_X][BLOCKS_Y],
                   MfmaFragAccf32 (&frags_f32)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for (int i = 0; i < BLOCKS_X; ++i) {
#pragma unroll
        for (int j = 0; j < BLOCKS_Y; ++j) {
// #pragma unroll
            for (int k = 0; k < frags_i32[i][j].num_elements; ++k) {
                frags_f32[i][j].x[k] = __int2float_rz(frags_i32[i][j].x[k]);
            }
        }
    }
}

ROCWMMA_DEVICE static inline void svgemm(MfmaFragAccf32 (&fragsAccOut)[BLOCKS_X][BLOCKS_Y],
                                       MfmaFragS const (&fragsA)[BLOCKS_X],
                                       MfmaFragV const (&fragsB)[BLOCKS_Y],
                                       MfmaFragAccf32 const (&fragsAccIn)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            mma_sync(fragsAccOut[i][j], fragsA[i], fragsB[j], fragsAccIn[i][j]);
        }
    }
}

// Performs warp tile mfma
ROCWMMA_DEVICE static inline void mfma(MfmaFragAcc (&fragsAccOut)[BLOCKS_X][BLOCKS_Y],
                                       MfmaFragA const (&fragsA)[BLOCKS_X],
                                       MfmaFragB const (&fragsB)[BLOCKS_Y],
                                       MfmaFragAcc const (&fragsAccIn)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            mma_sync(fragsAccOut[i][j], fragsA[i], fragsB[j], fragsAccIn[i][j]);
        }
    }
}

__device__ inline void twoAccToA(const MfmaFragAccf32& fragAcc0,
                                 const MfmaFragAccf32& fragAcc1,
                                 MfmaFragS& fragA)
{
    constexpr int WAVE_SIZE   = 64;
    constexpr int TILE_M      = 16;   // rows
    constexpr int TILE_N_ACC  = 16;   // cols of each acc tile
    constexpr int TILE_N_A    = 32;   // cols of A tile
    constexpr int COLBLOCK    = 8;    // 每个 lane 持有 8 列

    int lane = threadIdx.x % WAVE_SIZE;
    int row  = lane % TILE_M;        // 这一 lane 对应哪一行
    int blk  = lane / TILE_M;        // 0..3 -> 哪一段 8 列

    // 本线程手上 fragAcc 的 4 个元素缓存一下
    float acc0_local[4];
    float acc1_local[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc0_local[i] = fragAcc0.x[i];
        acc1_local[i] = fragAcc1.x[i];
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int global_col = blk * COLBLOCK + j;   // 0..31

        bool use_right = (global_col >= TILE_N_ACC);   // >=16 用右边那块
        int  tile_col  = use_right ? (global_col - TILE_N_ACC)
                                   : global_col;       // 0..15 in each fragAcc

        // 这个 (row, tile_col) 在 QK acc tile 里是哪个线程算出来的？
        int group    = row / 4;               // 0..3
        int src_lane = group * TILE_N_ACC + tile_col;   // 0..63
        int idx      = row % 4;               // 在那个 lane 的第几个 x[]

        // 先在本地 lane 选出“候选值”
        float local_val = use_right ? acc1_local[idx]
                                    : acc0_local[idx];

        // 从 src_lane 那个 lane 把它 shfl 过来
        float v = __shfl(local_val, src_lane, WAVE_SIZE);

        // 写到 fragA：这里一个 lane 持有 8 个半精度
        fragA.x[j] = InputTV(v);
    }
}

template <uint32_t         CTA_K,
          uint32_t         WARP_K,
          QuantGranularity Q_GRAN,
          QuantGranularity K_GRAN,
          int              SV_ITERS>
ROCWMMA_KERNEL void qk_int_sv_f8_attn_kernel(uint32_t head_dim,
                                             uint32_t lda,
                                             uint32_t ldb,
                                             uint32_t ldv,
                                             uint32_t ldd,
                                             int8_t* __restrict__ Q,
                                             int8_t* __restrict__ K,
                                             InputTV* __restrict__ V,
                                             float* __restrict__ O,
                                             float* __restrict__ Lse,
                                             float* __restrict__ Q_scale,
                                             float* __restrict__ K_scale,
                                             float* __restrict__ V_scale,
                                             float* __restrict__ V_mean,
                                             const uint32_t qo_len,
                                             const uint32_t kv_len,
                                             const uint32_t true_len,
                                             const uint32_t num_kv_groups,
                                             const uint32_t stride_bz_q,
                                             const uint32_t stride_seq_q,
                                             const uint32_t stride_h_q,
                                             const uint32_t stride_bz_k,
                                             const uint32_t stride_seq_k,
                                             const uint32_t stride_h_k,
                                             const uint32_t stride_bz_v,
                                             const uint32_t stride_h_v,
                                             const uint32_t stride_d_v,
                                             const uint32_t stride_bz_o,
                                             const uint32_t stride_seq_o,
                                             const uint32_t stride_h_o,
                                             float          sm_scale)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        ///
        /// 2D matrix coordinate setup
        ///

        // Tile Sizes
        constexpr auto warpTileSize  = make_coord2d(WARP_TILE_X, WARP_TILE_Y); 
        constexpr auto macroTileSize = make_coord2d(MACRO_TILE_X, MACRO_TILE_Y);

        // Local warp coordinate relative to current threadblock (wg).
        constexpr auto warpDims        = make_coord2d(WARPS_X, WARPS_Y);       
        auto           localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
        auto           localWarpOffset = localWarpCoord * warpTileSize;
        constexpr auto warpCount = get<0>(warpDims) * get<1>(warpDims);
        const auto warpIndex = get<0>(localWarpCoord) * get<1>(warpDims) + get<1>(localWarpCoord);

        const uint32_t batch_id = blockIdx.z;
        const uint32_t bx = blockIdx.x;
        const uint32_t num_qo_heads = gridDim.y;
        const uint32_t head_id = blockIdx.y;

        using MfmaFragDMap1d = GetDataLayout_t<MfmaFragD>;
        // auto Out_macroTileCoord = make_coord2d(blockIdx.x, 0) * macroTileSize;
        // auto Out_warpTileCoord  = Out_macroTileCoord + localWarpOffset;

        const uint32_t iterations = div_ceil(
            kv_len,
            MACRO_TILE_Y);
        // const int sv_iterations = (k + MACRO_TILE_Y - 1) / MACRO_TILE_Y;

        MfmaFragAccf32 fragsOut[SV_ITERS][BLOCKS_X][BLOCKS_Y];
        for(int i = 0; i < SV_ITERS; i++){
            fill(fragsOut[i], 0.0f);
        }

        HIP_DYNAMIC_SHARED(void*, localMemPtr);

        constexpr int NUM = ROCWMMA_M * ROCWMMA_N / WARP_SIZE;
        float m[NUM], d[NUM];
        for(int i = 0; i < NUM; i++)
        {
            m[i] = -500000.f;
            d[i] = 1.f;
        }

        for(int iter = 0; iter < iterations; iter++)
        {
            MfmaFragAcc fragsAcc[BLOCKS_X][BLOCKS_Y];
            fill(fragsAcc, 0.0f);
            auto macroTileCoord = make_coord2d(blockIdx.x, iter) * macroTileSize;
            {
                // auto macroTileCoord = make_coord2d(blockIdx.x, iter) * macroTileSize;
                auto warpTileCoord  = macroTileCoord + localWarpOffset;

                // Bounds check
                auto warpTileBound = warpTileCoord + warpTileSize;
                if(get<0>(warpTileBound) > qo_len || get<1>(warpTileBound) > kv_len)
                {
                    continue;
                }

                ///
                /// 1D global read coordinate setup
                ///
                using GRBuffAMap1d = GetDataLayout_t<GRBuffA>;
                using GRBuffBMap1d = GetDataLayout_t<GRBuffB>;

                // Initial globa read address offsets
                auto globalReadOffsetA
                    = batch_id * stride_bz_q + head_id * stride_h_q
                    + GRBuffAMap1d::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), lda);
                auto globalReadOffsetB
                    = batch_id * stride_bz_k + (head_id / num_kv_groups) * stride_h_k
                    + GRBuffBMap1d::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldb);

                // Incremental global read address offsets
                auto kStepOffsetA = GRBuffAMap1d::fromMatrixCoord(make_coord2d(0u, ROCWMMA_K), lda);
                auto kStepOffsetB = GRBuffBMap1d::fromMatrixCoord(make_coord2d(ROCWMMA_K, 0u), ldb);

                ///
                /// Cooperative config for global read A / B
                ///

                // WorkItems will be split up by minimum IOCount to perform either global read or local write.
                // These are inputs to cooperative functions.
                // constexpr auto warpCount = get<0>(warpDims) * get<1>(warpDims);

                // // Scheduling warp order is analogous to row major priority.
                // // E.g. Wg = (128, 2) = 2x2 warps
                // // (0, 0)   (0, 1)   Share Schedule: w0 = (0, 0), w1 = (0, 1),
                // // (1, 0)   (1, 1)                   w2 = (1, 0), w3 = (1, 1), count = 4
                // const auto warpIndex = get<0>(localWarpCoord) * get<1>(warpDims) + get<1>(localWarpCoord);

                ///
                /// Perform initial global pre-fetch
                ///

                GRBuffA grBuffA;
                GRBuffB grBuffB;

                globalReadCoopA<warpCount>(grBuffA, Q + globalReadOffsetA, lda, warpIndex);
                globalReadCoopB<warpCount>(grBuffB, K + globalReadOffsetB, ldb, warpIndex);

                globalReadOffsetA += kStepOffsetA;
                globalReadOffsetB += kStepOffsetB;

                ///
                /// Setup LDS addressing
                /// This kernel will use 2 separate LDS blocks for pipelining
                /// the input prefetching during the accumulation loop
                ///

                // HIP_DYNAMIC_SHARED(void*, localMemPtr);
                using LWBuffAShape = GetIOShape_t<LWBuffA>;
                using LWBuffBShape = GetIOShape_t<LWBuffB>;
                using LWBuffAMap1d = GetDataLayout_t<LWBuffA>;
                using LWBuffBMap1d = GetDataLayout_t<LWBuffB>;

                constexpr uint32_t ldsWidth  = ROCWMMA_K;
                constexpr uint32_t ldsHeight = LWBuffAShape::BlockHeight + LWBuffBShape::BlockHeight;
                constexpr uint32_t sizeLds   = ldsHeight * ldsWidth;
                constexpr uint32_t ldsld = std::is_same_v<DataLayoutLds, row_major> ? ldsWidth : ldsHeight;

                auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
                auto* ldsPtrHi = ldsPtrLo + sizeLds;

                // Local write offsets to start of A / B data
                auto ldsWriteOffsetA = 0u;
                auto ldsWriteOffsetB
                    = LWBuffAMap1d::fromMatrixCoord(make_coord2d(LWBuffAShape::BlockHeight, 0u), ldsld);

                // Local read offsets for mfma frags
                auto ldsReadOffsetA
                    = ldsWriteOffsetA
                    + LWBuffAMap1d::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
                auto ldsReadOffsetB
                    = ldsWriteOffsetB
                    + LWBuffBMap1d::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

                ///
                /// Write prefetch to local
                ///
                localWriteCoopA<warpCount>(ldsPtrLo + ldsWriteOffsetA, grBuffA, ldsld, warpIndex);
                localWriteCoopB<warpCount>(ldsPtrLo + ldsWriteOffsetB, grBuffB, ldsld, warpIndex);

                ///
                /// Initialize accumulation frags
                ///
                // MfmaFragAcc fragsAcc[BLOCKS_X][BLOCKS_Y];
                // fill(fragsAcc, 0.0f);

                ///
                /// Synchronize warps and memory
                ///
                synchronize_workgroup();

                ///
                /// Accumulate A * B for all mfma frags in warp tile
                ///
                for(uint32_t currentK = ROCWMMA_K; currentK < head_dim; currentK += ROCWMMA_K)
                {
                    MfmaFragA fragsA[BLOCKS_X];
                    MfmaFragB fragsB[BLOCKS_Y];

                    // Local read mfma frags from first LDS buffer
                    localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
                    localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);

                    // Prefetch next round of global frags
                    globalReadCoopA<warpCount>(grBuffA, Q + globalReadOffsetA, lda, warpIndex);
                    globalReadCoopB<warpCount>(grBuffB, K + globalReadOffsetB, ldb, warpIndex);

                    // Advance offsets to next k step
                    globalReadOffsetA += kStepOffsetA;
                    globalReadOffsetB += kStepOffsetB;

                    // accum(A * B)
                    mfma(fragsAcc, fragsA, fragsB, fragsAcc);

                    // Write prefetch to second LDS buffer
                    localWriteCoopA<warpCount>(ldsPtrHi + ldsWriteOffsetA, grBuffA, ldsld, warpIndex);
                    localWriteCoopB<warpCount>(ldsPtrHi + ldsWriteOffsetB, grBuffB, ldsld, warpIndex);

                    // Make sure that all waves have finished reading / writing to lds for currentK.
                    synchronize_workgroup();

                    // Swap Lds buffers
                    auto* tmp = ldsPtrLo;
                    ldsPtrLo  = ldsPtrHi;
                    ldsPtrHi  = tmp;
                }
            

                ///
                /// Clean up tail A * B
                ///
                MfmaFragA fragsA[BLOCKS_X];
                MfmaFragB fragsB[BLOCKS_Y];

                // Local read mfma frags
                localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
                localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);
                mfma(fragsAcc, fragsA, fragsB, fragsAcc);
            }
            // SAVE one block tile for verify
            // if(blockIdx.x == 1 && iter == 0 && head_id == 0 && batch_id == 0)
            // {
            //     auto warpTileCoordT = localWarpOffset;
            //     globalWriteT(T + MfmaFragDMap1d::fromMatrixCoord(warpTileCoordT, 64), fragsAcc, 64);
            // }
            // TRAP_IF(blockIdx.x == 0 && head_id == 0 && threadIdx.x == 0 && threadIdx.y == 0);
            //  Here,we try use store/load_matrix_sync interface to get a row data by reset LDS height 
            //  and width logically.
            
            MfmaFragAccf32 fragsTmp[BLOCKS_X][BLOCKS_Y];
            // for(int i = 0; i < BLOCKS_X; i++)
            // {
            //     for(int j = 0; j < BLOCKS_Y; j++)
            //     {
            //         fragsTmp[i][j] = fragsAcc[i][j];
            //     }
            // }
            convertS32toF32(fragsAcc,fragsTmp);

            // auto* ldsPtr = reinterpret_cast<ComputeTV*>(localMemPtr);
            // constexpr uint32_t ldsWidth_new  = MACRO_TILE_Y;
            // constexpr uint32_t ldsHeight_new = 2 * MACRO_TILE_X;
            // constexpr uint32_t ldsld_new = std::is_same_v<DataLayoutLds_new, row_major> ? ldsWidth_new : ldsHeight_new;

            // auto ldsReadOffsetAcc = get<0>(localWarpOffset) * ldsld_new + get<1>(localWarpOffset);

            // localWriteAcc(fragsTmp,ldsPtr + ldsReadOffsetAcc,ldsld_new);
            // synchronize_workgroup();

            float original_sm_scale = sm_scale;
            uint32_t baseq_scale_idx, basek_scale_idx;
            if constexpr (Q_GRAN == QuantGranularity::kPerThread)
            {
                const uint32_t num_warp_block_q = gridDim.x * MACRO_TILE_X / 32;//only support 16*16
                baseq_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) + head_id * (num_warp_block_q * 8) 
                                + bx * (MACRO_TILE_X / 32 * 8) + get<0>(localWarpCoord) / 2 * 8;
                // const uint32_t num_warp_block_q = gridDim.x * 2;//only support 16*16
                // baseq_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) + head_id * (num_warp_block_q * 8) 
                //                 + bx * (2 * 8) + (get<0>(localWarpCoord)/2) * 8;
            }
            if constexpr (K_GRAN == QuantGranularity::kPerThread)
            {
                // Here,dequant phase's div_ceil(kv_len,CTA_K) != div_ceil(kv_len,CTA_K) in quant phase
                // dequant phase kv_len %128 == 0               quant phase kv_len % 64 == 0 
                const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
                basek_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * (num_warp_block_k * 4) 
                                + (head_id / num_kv_groups) * (num_warp_block_k * 4) + iter * 4;
            }
            //  transform data for validate
            //  actually do softmax ops here
            // float m_prev = m[fq][k];


            ////    Here,we do update mdo, and we'll make below codes to a func.
            ////    start
            for(int i = 0; i < BLOCKS_X; i++)
            {
                // auto m_tmp = -50000.f;
                float m_tmp[NUM];
                for(int k = 0; k < els_per_thread; k++)
                {
                    m_tmp[k] = -5000.f;
                }
                // dequant and get max
                for(int j = 0; j < BLOCKS_Y; j++)
                {
                    // auto baseoffset = ldsReadOffsetAcc + i * ROCWMMA_M * ldsld_new + j * ROCWMMA_N;
                    // auto threadoffset = threadIdx.x % threads_per_row * els_per_thread
                    //                     + (threadIdx.x) % WARP_SIZE / threads_per_row * ldsld_new;

                    // uint32_t row = (threadIdx.x % WARP_SIZE / threads_per_row) % 8; //q_scale offset
                    // uint32_t col = (threadIdx.x % threads_per_row * els_per_thread) % 8 / 2; //k_scale offset

                    for(int k = 0; k < els_per_thread; k++)
                    {
                        // // uint32_t th_off = threadIdx.x % WARP_SIZE * els_per_thread;
                        // if((iter * MACRO_TILE_Y + get<1>(localWarpOffset) + j * ROCWMMA_N + threadIdx.x % threads_per_row * els_per_thread + k) % kv_len < true_len)
                        // {
                        //     // fragsTmp[i][j].x[th_off + k] *= (log2e * sm_scale);
                        //     // fragsTmp[i][j].x[th_off + k] *= Q_scale[baseq_scale_idx + row];

                        //     ldsPtr[baseoffset + threadoffset + k] *= (Q_scale[baseq_scale_idx + row] * log2e * sm_scale);
                        //     if(k >= els_per_thread / 2){
                        //         // fragsTmp[i][j].x[th_off + k] *= K_scale[basek_scale_idx + col + 1];
                        //         ldsPtr[baseoffset + threadoffset + k] *= K_scale[basek_scale_idx + col + 1];
                        //     }
                        //     else {
                        //         // fragsTmp[i][j].x[th_off + k] *= K_scale[basek_scale_idx + col];
                        //         ldsPtr[baseoffset + threadoffset + k] *= K_scale[basek_scale_idx + col];
                        //     }
                        // }
                        // else{
                        //     // fragsTmp[i][j].x[th_off + k] = -5000.f;
                        //     ldsPtr[baseoffset + threadoffset + k] = -5000.f;
                        // }
                        // // m_tmp = fmaxf(m_tmp, fragsTmp[i][j].x[th_off + k]);
                        // m_tmp = fmaxf(m_tmp, ldsPtr[baseoffset + threadoffset + k]);

                        // dequant here
                        auto row = (threadIdx.x % 64) / ROCWMMA_N * 4 + k;
                        auto col = threadIdx.x % ROCWMMA_N;
                        
                        fragsTmp[i][j].x[k] *= (Q_scale[baseq_scale_idx + row % 8] * log2e * sm_scale);
                        fragsTmp[i][j].x[k] *= K_scale[basek_scale_idx + (col % 8) / 2];

                        m_tmp[k] = fmaxf(m_tmp[k], fragsTmp[i][j].x[k]);
                    }
                }

                for(int k = 0; k < els_per_thread; k++)
                {
                    float v = m_tmp[k];
                    for (int offset = ROCWMMA_N / 2; offset > 0; offset >>= 1)
                    {
                        float other = __shfl_xor(v, offset, ROCWMMA_N);
                        v = fmaxf(v, other);
                    }
                    m_tmp[k] = v;
                }

                // m_tmp -= 8.807f;
                // m_tmp = fmaxf(m_tmp, __shfl_xor(m_tmp, 1, 4));
                // m_tmp = fmaxf(m_tmp, __shfl_xor(m_tmp, 2, 4));
                // // inter-warp get max
                // auto rd_off = MACRO_TILE_X * MACRO_TILE_Y;
                // auto wp_off = get<0>(localWarpOffset) * ldsld_new + i * ROCWMMA_M * ldsld_new;
                // auto th_off = threadIdx.x % WARP_SIZE / threads_per_row * ldsld_new;
                // auto write_off = get<1>(localWarpCoord);
                
                // if(threadIdx.x % 4 == 0)
                // {
                //     ldsPtr[rd_off + wp_off + th_off + write_off] = m_tmp;
                // }
                // synchronize_workgroup();

                // m_tmp = fmaxf(ldsPtr[rd_off + wp_off + th_off], ldsPtr[rd_off + wp_off + th_off + 1]);

                // synchronize_workgroup();

                float o_scale[NUM];
                for(int k = 0; k < NUM; k++)
                {
                    float m_prev = m[k];
                    m[k] = fmaxf(m[k], m_tmp[k]);

                    o_scale[k] = exp2f(m_prev - m[k]);

                    d[k] *= o_scale[k];
                }
                // float m_prev = m[i];
                // m[i] = fmaxf(m[i], m_tmp);

                // float o_scale = exp2f(m_prev - m[i]);

                // update denominator
                // d[i] *= o_scale;

                // float neg_m = -m[i];
                for(int j = 0; j < BLOCKS_Y; j++)
                {
                    // auto baseoffset = ldsReadOffsetAcc + i * ROCWMMA_M * ldsld_new + j * ROCWMMA_N;
                    // auto threadoffset = threadIdx.x % threads_per_row * els_per_thread
                    //                     + (threadIdx.x) % WARP_SIZE / threads_per_row * ldsld_new;
                    
                    // uint32_t row = (threadIdx.x % WARP_SIZE / threads_per_row) % 8; //q_scale offset
                    // uint32_t col = (threadIdx.x % threads_per_row * els_per_thread) % 8 / 2; //k_scale offset
                    for(int k = 0; k < els_per_thread; k++)
                    {
                        // uint32_t th_off = threadIdx.x % WARP_SIZE * els_per_thread;
                        // fragsTmp[i][j].x[th_off + k] = exp2f(fragsTmp[i][j].x[th_off + k] + neg_m);
                        // d[i] += fragsTmp[i][j].x[th_off + k];
                        // ldsPtr[baseoffset + threadoffset + k] *= (Q_scale[baseq_scale_idx + row] * log2e * sm_scale);
                        // if(k >= els_per_thread / 2){
                        //     ldsPtr[baseoffset + threadoffset + k] *= K_scale[basek_scale_idx + col + 1];
                        // }
                        // else ldsPtr[baseoffset + threadoffset + k] *= K_scale[basek_scale_idx + col];

                        // ldsPtr[baseoffset + threadoffset + k] = exp2f(ldsPtr[baseoffset + threadoffset + k] + neg_m);
                        // if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
                        // {
                        //     d[i] += ldsPtr[baseoffset + threadoffset + k];
                        // }

                        float neg_m = -m[k];
                        ////    accumulate d
                        fragsTmp[i][j].x[k] = exp2f(fragsTmp[i][j].x[k] + neg_m);

                        // if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
                        // {
                            float d_local = fragsTmp[i][j].x[k];
                            for (int offset = ROCWMMA_N / 2; offset > 0; offset >>= 1) {
                                float other = __shfl_xor(d_local, offset, ROCWMMA_N);
                                d_local += other;
                            }
                            d[k] += d_local;
                        // }
                    }
                }
                //here require handle RO
                for(int k = 0; k < SV_ITERS; k++)
                {
                    // auto OutOffset = ldsReadOffsetAcc + MACRO_TILE_X * MACRO_TILE_Y;
                    // localWriteOut(fragsOut[k][i],ldsPtr + OutOffset + i * ROCWMMA_M * ldsld_new, ldsld_new);
                    
                    // synchronize_workgroup();

                    for(int j = 0; j < BLOCKS_Y; j++)
                    {
                        // auto baseoffset = OutOffset + i * ROCWMMA_M * ldsld_new + j * ROCWMMA_N;
                        // auto threadoffset = threadIdx.x % threads_per_row * els_per_thread
                        //                     + (threadIdx.x) % WARP_SIZE / threads_per_row * ldsld_new;

                        for(int z = 0; z < fragsOut[k][i][j].num_elements; z++)
                        {
                            // float o_true = __shfl(o_scale[z], (threadIdx.x/ROCWMMA_N * 4 + z) * 4, 64);
                            float o_true = o_scale[z];
                            fragsOut[k][i][j].x[z] *= o_true;
                            // fragsOut[k][i][j].x[ z] *= o_scale;
                            // ldsPtr[baseoffset + threadoffset + z] *= o_scale;
                        }
                    }

                    // localReadOut(fragsOut[k][i],ldsPtr + OutOffset + i * ROCWMMA_M * ldsld_new,ldsld_new);
                    // synchronize_workgroup();
                }

                // raise RS to exponent
                // float neg_m = -m[i];
                // for(int j = 0; j < BLOCKS_Y; j++)
                // {
                //     auto baseoffset = ldsReadOffsetAcc + i * ROCWMMA_M * ldsld_new + j * ROCWMMA_N;
                //     auto threadoffset = threadIdx.x % threads_per_row * els_per_thread
                //                         + (threadIdx.x) % WARP_SIZE / threads_per_row * ldsld_new;
                    
                //     // uint32_t row = (threadIdx.x % WARP_SIZE / threads_per_row) % 8; //q_scale offset
                //     // uint32_t col = (threadIdx.x % threads_per_row * els_per_thread) % 8 / 2; //k_scale offset
                //     for(int k = 0; k < els_per_thread; k++)
                //     {
                //         // uint32_t th_off = threadIdx.x % WARP_SIZE * els_per_thread;
                //         // fragsTmp[i][j].x[th_off + k] = exp2f(fragsTmp[i][j].x[th_off + k] + neg_m);
                //         // d[i] += fragsTmp[i][j].x[th_off + k];
                //         // ldsPtr[baseoffset + threadoffset + k] *= (Q_scale[baseq_scale_idx + row] * log2e * sm_scale);
                //         // if(k >= els_per_thread / 2){
                //         //     ldsPtr[baseoffset + threadoffset + k] *= K_scale[basek_scale_idx + col + 1];
                //         // }
                //         // else ldsPtr[baseoffset + threadoffset + k] *= K_scale[basek_scale_idx + col];

                //         ldsPtr[baseoffset + threadoffset + k] = exp2f(ldsPtr[baseoffset + threadoffset + k] + neg_m);
                //         if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
                //         {
                //             d[i] += ldsPtr[baseoffset + threadoffset + k];
                //         }
                //     }
                // }
            }
            ////    end.

            // accumlate d
            // accumulate_d(ldsPtr,d);
            // for(int i = 0; i < BLOCKS_X; i++)
            // {
            //     for(int j = 0; j < BLOCKS_Y; j++)
            //     {
            //         auto baseoffset = ldsReadOffsetAcc + i * ROCWMMA_M * ldsld_new + j * ROCWMMA_N;
            //         auto threadoffset = threadIdx.x % threads_per_row * els_per_thread
            //                             + (threadIdx.x) % WARP_SIZE / threads_per_row * ldsld_new;
                    
            //         for(int k = 0; k < els_per_thread; k++)
            //         {
            //             d[i] += ldsPtr[baseoffset + threadoffset + k];
            //             // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.x < 64)
            //             // {
            //             //     printf("after acc:d[%d][%d]:%f\n",i,k,ldsPtr[baseoffset + threadoffset + k]);
            //             // }
            //         }
            //     }
            //     // inter warp reduce again?
            //     synchronize_workgroup();
            
            //     // accumlate d for inter-warp exchange
            // }

            //// cast f32 to fp8

            // localWriteAcc(fragsTmp,ldsPtr + ldsReadOffsetAcc,ldsld_new);
            // synchronize_workgroup();

            // InputTV RS[BLOCKS_X][BLOCKS_Y][els_per_thread];
            // for(int i = 0; i < BLOCKS_X; i++)
            // {
            //     for(int j = 0; j < BLOCKS_Y; j++)
            //     {
            //         auto baseoffset = ldsReadOffsetAcc + i * ROCWMMA_M * ldsld_new + j * ROCWMMA_N;
            //         auto threadoffset = threadIdx.x % threads_per_row * els_per_thread
            //                             + (threadIdx.x) % WARP_SIZE / threads_per_row * ldsld_new;
                    
            //         for(int k = 0; k < els_per_thread; k++)
            //         {
            //             // RS[i][j][k].__x = __hip_cvt_float_to_fp8(ldsPtr[baseoffset + threadoffset + k],
            //                                                 // __HIP_SATFINITE, __HIP_E4M3_FNUZ);
            //             // uint32_t th_off = threadIdx.x % WARP_SIZE * els_per_thread;
            //             // RS[i][j][k] = InputTV(fragsTmp[i][j].x[th_off + k]);
            //             RS[i][j][k] = InputTV(ldsPtr[baseoffset + threadoffset + k]);
            //             // __hip_fp8_storage_t byte = RS[i][j][k];
            //             // __half_raw h = __hip_cvt_fp8_to_halfraw(byte, __HIP_E4M3_FNUZ);
            //             // float fp8_decoded = __half2float(*reinterpret_cast<__half*>(&h));
            //         }
            //     }
            // }
            
            //  do we need sync here?
            // synchronize_workgroup();

            auto* ldsPtrf8 = reinterpret_cast<InputTV*>(localMemPtr);
            constexpr uint32_t ldsWidth_fp8  = ROCWMMA_K;
            // constexpr uint32_t ldsHeight_fp8 = 2 * MACRO_TILE_X * sizeof(ComputeTV) / sizeof(InputTV);
            constexpr uint32_t ldsHeight_fp8 = MACRO_TILE_X + MACRO_TILE_Y;
            constexpr uint32_t ldsld_fp8 = std::is_same_v<DataLayoutLds_new, row_major> ? ldsWidth_fp8 : ldsHeight_fp8;

            // auto ldsReadOffsetsv = get<0>(localWarpOffset) * ldsld_fp8 + get<1>(localWarpOffset);
            // for(int i = 0; i < BLOCKS_X; i++)
            // {
            //     for(int j = 0; j < BLOCKS_Y; j++)
            //     {
            //         auto baseoffset = ldsReadOffsetsv + i * ROCWMMA_M * ldsld_fp8 + j * ROCWMMA_N;
            //         auto threadoffset = threadIdx.x % threads_per_row * els_per_thread
            //                             + (threadIdx.x) % WARP_SIZE / threads_per_row * ldsld_fp8;
                    
            //         for(int k = 0; k < els_per_thread; k++)
            //         {
            //             ldsPtrf8[baseoffset + threadoffset + k] = RS[i][j][k];
            //             // if(blockIdx.x == 0 && head_id == 0 && (iter < 3 || iter > iterations -4) && threadIdx.x < 4 && threadIdx.y == 0 && i == 0 && j == 0)
            //             // {
            //             //     __hip_fp8_storage_t byte = ldsPtrf8[baseoffset + threadoffset + k];
            //             //     __half_raw h = __hip_cvt_fp8_to_halfraw(byte, __HIP_E4M3_FNUZ);
            //             //     float fp8_decoded = __half2float(*reinterpret_cast<__half*>(&h));
            //             // }
            //         }
            //     }
            // }


            // //  do we need sync here?
            // synchronize_workgroup();
            constexpr size_t SV_CNT = MACRO_TILE_Y / ROCWMMA_K;
            // auto ldsReadOffsetS = get<0>(localWarpOffset) * ldsld_fp8;
            MfmaFragS fragsS[SV_CNT][BLOCKS_X];
            for(int i = 0; i < SV_CNT; i++)
            {
                // localReadS(fragsS[i], ldsPtrf8 + ldsReadOffsetS, ldsld_fp8);
                // ldsReadOffsetS += ROCWMMA_K;
                twoAccToA(fragsTmp[0][i * 2], fragsTmp[0][i * 2 + 1], fragsS[i][0]);
            }
            // //cause we let datalayout lds_new == row_major now!


            ////    loop
            for(int sv_iter = 0; sv_iter < SV_ITERS; sv_iter++)
            {
                auto sv_mToffset = sv_iter * get<1>(macroTileSize);
                auto sv_wpoffset = sv_mToffset + get<1>(localWarpOffset);

                // if(sv_wpoffset + get<1>(warpTileSize) > k)
                // {
                //     continue;
                // }

                ////  load v and calcualte sv  
                using GRBuffVMap1d = GetDataLayout_t<GRBuffV>;
                auto globalReadOffsetV
                    = GRBuffVMap1d::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), 1u)
                    + sv_iter * MACRO_TILE_Y * ldv + batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v;
                
                auto kStepOffsetV = GRBuffVMap1d::fromMatrixCoord(make_coord2d(ROCWMMA_K, 0u), ldv);

                GRBuffV grBuffV;

                using LWBuffVShape = GetIOShape_t<LWBuffV>;
                using LWBuffVMap1d = GetDataLayout_t<LWBuffV>;

                auto ldsWriteOffsetV = MACRO_TILE_X * MACRO_TILE_Y; 
                // auto ldsReadOffsetS = get<0>(localWarpOffset) * ldsld_fp8;
                    // =get<0>(localWarpOffset) * ldsld_new + get<1>(localWarpOffset);
                auto ldsReadOffsetV
                    = ldsWriteOffsetV
                    + LWBuffVMap1d::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld_fp8);

                for(uint32_t currentK = 0; currentK < SV_CNT; currentK++)
                {
                    globalReadCoopV<warpCount>(grBuffV, V + globalReadOffsetV, ldv, warpIndex);
                    globalReadOffsetV += kStepOffsetV;

                    localWriteCoopV<warpCount>(ldsPtrf8 + ldsWriteOffsetV, grBuffV, ldsld_fp8, warpIndex);
                    synchronize_workgroup();

                    //do gemm
                    // MfmaFragS fragsS[BLOCKS_X];
                    MfmaFragV fragsV[BLOCKS_Y];

                    // Local read mfma frags from first LDS buffer
                    // localReadS(fragsS, ldsPtrf8 + ldsReadOffsetS, ldsld_fp8);
                    // ldsReadOffsetS += ROCWMMA_K;

                    localReadV(fragsV, ldsPtrf8 + ldsReadOffsetV, ldsld_fp8);
                    // if(currentK == 0 && blockIdx.x == 0 && head_id == 0 && 
                    //     (iter == 0 ) && threadIdx.x < 64 && threadIdx.y < 1)
                    // {
                    //     for(int k = 0; k < fragsS[0].num_elements; k++)
                    //     {
                    //         __hip_fp8_storage_t bytes = fragsS[0].x[ k];
                    //         __half_raw hs = __hip_cvt_fp8_to_halfraw(bytes, __HIP_E4M3_FNUZ);
                    //         float fp8_decodeds = __half2float(*reinterpret_cast<__half*>(&hs));
                                    
                    //         __hip_fp8_storage_t bytev = fragsV[0].x[k];
                    //         __half_raw hv = __hip_cvt_fp8_to_halfraw(bytev, __HIP_E4M3_FNUZ);
                    //         float fp8_decodedv = __half2float(*reinterpret_cast<__half*>(&hv));

                    //         // printf("S threadIdx.x:%u, k:%d, raw value:%f, value:%f\n",threadIdx.x,k,static_cast<float>(fragsS[0].x[ k]),fp8_decodeds);
                    //         // printf("V threadIdx.x:%u, k:%d, raw value:%f, value:%f\n",threadIdx.x,k,static_cast<float>(fragsV[0].x[ k]),fp8_decodedv);
                    //     }
                    // }
                    // accum(S * V)
                    svgemm(fragsOut[sv_iter], fragsS[currentK], fragsV, fragsOut[sv_iter]);
                    // if(currentK == 0 && blockIdx.x == 0 && head_id == 0 && 
                    //     (iter == 0) && threadIdx.x < 64 && threadIdx.y == 0)
                    // {
                    //     for(int i = 0 ; i < BLOCKS_X; i++)
                    //     {
                    //         for(int j = 0; j < BLOCKS_Y; j++)
                    //         {
                    //             for(int k = 0; k < els_per_thread; k++)
                    //             {
                    //                 // printf("[sv res]: threadIdx.x:%u, iter:%d, i:%d, j:%d, k:%d, fp32:%f\n",
                    //                 //     threadIdx.x, iter, i, j, k, fragsOut[0][i][j].x[k + threadIdx.x * els_per_thread]);
                    //             }
                    //         }
                    //     }
                    // }
                    // Make sure that all waves have finished reading / writing to lds for currentK.
                    synchronize_workgroup();
                }
            }
        }

        //// normalize d

        float d_rcp[BLOCKS_X];

        auto* ldsPtrOut = reinterpret_cast<ComputeTV*>(localMemPtr);
        constexpr uint32_t ldsWidth_Out  = MACRO_TILE_Y;
        constexpr uint32_t ldsHeight_Out = 2 * MACRO_TILE_X;
        constexpr uint32_t ldsld_Out = std::is_same_v<DataLayoutLds_new, row_major> ? ldsWidth_Out : ldsHeight_Out;

        auto ldsReadOffsetAcc = get<0>(localWarpOffset) * ldsld_Out + get<1>(localWarpOffset);

        // for(int i = 0; i < BLOCKS_X; i++)
        // {
        //     // auto wp_off = get<0>(localWarpOffset) * ldsld_Out + i * ROCWMMA_M * ldsld_Out;
        //     // auto th_off = threadIdx.x % WARP_SIZE / threads_per_row * ldsld_Out;
        //     // auto write_off = get<1>(localWarpCoord);

        //     // ldsPtrOut[wp_off + th_off + threadIdx.x % threads_per_row + write_off * threads_per_row] = d[i];
            
        //     // synchronize_workgroup();

        //     // float d_sum = 0.f;
        //     // d_sum = ldsPtrOut[wp_off + th_off + threadIdx.x % threads_per_row]
        //     //         +  ldsPtrOut[wp_off + th_off + threadIdx.x % threads_per_row + 4];
        //     // d[i] = d_sum;

        //     // synchronize_workgroup();
            
        //     d[i] += __shfl_xor(d[i], 1, 4);
        //     d[i] += __shfl_xor(d[i], 2, 4);
        //     d_rcp[i] = 1.0f / d[i];
        // }

        //// normlize o

        for(int i = 0; i < NUM; i++)
        {
            d_rcp[i] = 1.0f / d[i];
        }
#pragma unroll
        for(int sv_iter = 0; sv_iter < SV_ITERS; sv_iter++)
        {
            // localWriteAcc(fragsOut[sv_iter],ldsPtrOut + ldsReadOffsetAcc,ldsld_Out);
            // synchronize_workgroup();

#pragma unroll
            for(int i = 0; i < BLOCKS_X; i++)
            {
#pragma unroll
                for(int j = 0; j < BLOCKS_Y; j++)
                {
                    // auto baseoffset = ldsReadOffsetAcc + i * ROCWMMA_M * ldsld_Out + j * ROCWMMA_N;
                    // auto threadoffset = threadIdx.x % threads_per_row * els_per_thread
                    //                     + (threadIdx.x) % WARP_SIZE / threads_per_row * ldsld_Out;
                    
                    auto base_offset = batch_id * (num_qo_heads / num_kv_groups) * head_dim + (head_id / num_kv_groups) * head_dim;
                    auto scale_idx = sv_iter * MACRO_TILE_Y + get<1>(localWarpOffset) + j * ROCWMMA_N;
//                     const int v_index = base_offset + scale_idx + threadIdx.x % threads_per_row * els_per_thread;
//                     float4* o_vec_ptr = reinterpret_cast<float4*>(&ldsPtrOut[baseoffset + threadoffset]);
//                     const float4* v_vec_ptr = reinterpret_cast<const float4*>(&V_scale[v_index]);

//                     float4 o4 = *o_vec_ptr;
//                     float4 v4 = *v_vec_ptr;

//                     o4.x *= (d_rcp[i] * v4.x);
//                     o4.y *= (d_rcp[i] * v4.y);
//                     o4.z *= (d_rcp[i] * v4.z);
//                     o4.w *= (d_rcp[i] * v4.w);

//                     *o_vec_ptr = o4;
#pragma unroll
                    for(int k = 0; k < fragsOut[sv_iter][i][j].num_elements; k++)
                    {
                        auto col = threadIdx.x % ROCWMMA_N;
                        // float d_true = __shfl(d_rcp[i], (threadIdx.x/16*4+k)*4, 64);
                        // fragsOut[sv_iter][i][j].x[k] *= d_true;
                        fragsOut[sv_iter][i][j].x[k] /= d[k];
                        fragsOut[sv_iter][i][j].x[k] *= V_scale[base_offset + scale_idx + col];
                        // ldsPtrOut[baseoffset + threadoffset + k] *= d_rcp[i];
                        
                        // auto id = threadIdx.x % WARP_SIZE % threads_per_row;

                        // ldsPtrOut[baseoffset + threadoffset + k] *= V_scale[base_offset + scale_idx + id * els_per_thread + k];
                    }
                }
            }

            // localReadAcc(fragsOut[sv_iter],ldsPtrOut + ldsReadOffsetAcc,ldsld_Out);
            // synchronize_workgroup();
        }

        for(int i = 0; i < SV_ITERS; i++){
            auto Out_macroTileCoord = make_coord2d(blockIdx.x, i) * macroTileSize;
            auto Out_warpTileCoord  = Out_macroTileCoord + localWarpOffset;
            
            globalWriteD(O + batch_id * stride_bz_o + head_id * stride_h_o +
                        MfmaFragDMap1d::fromMatrixCoord(Out_warpTileCoord, ldd), fragsOut[i], ldd);
        }
    }
}
// template <typename InputT,
//           typename InputTV,
//           typename OutputT,
//           typename ComputeT,
//           typename LayoutA,
//           typename LayoutB,
//           typename LayoutC,
//           typename LayoutD = LayoutC>
// ROCWMMA_HOST void sage_attn_cpu(uint32_t      qo_len,      // m
//                                 uint32_t      head_dim,    // n  
//                                 uint32_t      kv_len,      // k
//                                 int8_t const* Q,           // quantized int8
//                                 int8_t const* K,           // quantized int8
//                                 float8_fnuz_t const* V,    // fp8
//                                 float*        O,           // output
//                                 float*        Q_scale,     // quantization scales
//                                 float*        K_scale,     
//                                 float*        V_scale,
//                                 float*        V_mean,
//                                 uint32_t      lda,
//                                 uint32_t      ldb,
//                                 uint32_t      ldv,
//                                 uint32_t      ldo,
//                                 uint32_t      batch_size,
//                                 uint32_t      num_qo_heads,
//                                 uint32_t      num_kv_heads,
//                                 float         sm_scale,
//                                 bool          fuse_v_scale = false,
//                                 bool          fuse_v_mean = false)
// {
//     constexpr float log2e = 1.442695f;
    
//     std::cout << "CPU Implementation Debug:" << std::endl;
//     std::cout << "qo_len=" << qo_len << ", head_dim=" << head_dim << ", kv_len=" << kv_len << std::endl;
//     std::cout << "batch_size=" << batch_size << ", num_qo_heads=" << num_qo_heads 
//               << ", num_kv_heads=" << num_kv_heads << std::endl;
    
//     // ??num_kv_groups
//     uint32_t num_kv_groups = (num_qo_heads + num_kv_heads - 1) / num_kv_heads;
//     if (num_kv_groups == 0) num_kv_groups = 1;
    
//     std::cout << "num_kv_groups=" << num_kv_groups << std::endl;
    
//     // ?????? - ?GPU??????
//     auto rowMajorIndex = [](uint32_t row, uint32_t col, uint32_t ld) { 
//         return row * ld + col; 
//     };
    
//     auto colMajorIndex = [](uint32_t row, uint32_t col, uint32_t ld) { 
//         return col * ld + row; 
//     };
    
//     // ??GPU?????????
//     // GPU?:DataLayoutA = row_major, DataLayoutB = col_major, DataLayoutC = row_major
//     auto qIndex = rowMajorIndex;  // Q: row_major [batch, num_qo_heads, qo_len, head_dim]
//     auto kIndex = colMajorIndex; // K: col_major [batch, num_kv_heads, head_dim, kv_len] 
//     auto vIndex = colMajorIndex; // V: col_major [batch, num_kv_heads, kv_len, head_dim]
//     auto oIndex = rowMajorIndex; // O: row_major [batch, num_qo_heads, qo_len, head_dim]
    
 
//     // ??????????
//     uint32_t q_total_size = batch_size * num_qo_heads * qo_len * head_dim;
//     uint32_t k_total_size = batch_size * num_kv_heads * kv_len * head_dim; 
//     uint32_t v_total_size = batch_size * num_kv_heads * kv_len * head_dim;
//     uint32_t o_total_size = batch_size * num_qo_heads * qo_len * head_dim;
    
//     std::cout << "Matrix sizes - Q: " << q_total_size << ", K: " << k_total_size 
//               << ", V: " << v_total_size << ", O: " << o_total_size << std::endl;
    
//     // ????
//     if (q_total_size == 0 || k_total_size == 0 || v_total_size == 0 || o_total_size == 0) {
//         std::cout << "Error: Zero matrix size detected!" << std::endl;
//         return;
//     }
    
//     // ??????0
//     std::fill(O, O + o_total_size, 0.0f);
    
//     // ?????
//     for(uint32_t batch = 0; batch < batch_size; ++batch) {
//         std::cout << "Processing batch " << batch << "/" << batch_size << std::endl;
        
//         for(uint32_t head = 0; head < num_qo_heads; ++head) {
//             // ?????KV? - ??????
//             uint32_t kv_head = head % num_kv_heads;  // ??????,?????
            
//             if (kv_head >= num_kv_heads) {
//                 kv_head = num_kv_heads - 1;  // ????
//             }                    
//                 // ??attention scores?exp scores
//                 std::vector<float> attention_scores(kv_len, 0.0f);
//                 std::vector<float> exp_scores(kv_len, 0.0f);
//                 float max_score = -std::numeric_limits<float>::infinity();
                
//                 // ??1: ??Q*K^T attention scores
//                 for(uint32_t j = 0; j < kv_len; ++j) { // ??key??
//                     float score = 0.0f;
                    
//                     // ????(????)
//                     for(uint32_t h = 0; h < head_dim; ++h) { // head_dim??
//                         // ?????? - ?GPU??????
//                         uint32_t q_idx = qIndex(
//                             batch * num_qo_heads * qo_len + head * qo_len + i, 
//                             h, head_dim
//                         );
                        
//                         uint32_t k_idx = kIndex(
//                             batch * num_kv_heads * head_dim + kv_head * head_dim + h,
//                             j, kv_len
//                         );
                        
//                         // ????
//                         if (q_idx >= q_total_size || k_idx >= k_total_size) {
//                             continue;
//                         }
                        
//                         // ?????
//                         int8_t q_val = Q[q_idx];
//                         int8_t k_val = K[k_idx];
                        
//                         // ???????? - ?????????????
//                         uint32_t q_scale_idx = 0;  // ????????
//                         uint32_t k_scale_idx = 0;  // ????????
                        
//                         // ???
//                         float q_val_dequant = static_cast<float>(q_val) * (Q_scale ? Q_scale[q_scale_idx] : 1.0f);
//                         float k_val_dequant = static_cast<float>(k_val) * (K_scale ? K_scale[k_scale_idx] : 1.0f);
                        
//                         // ????
//                         score += q_val_dequant * k_val_dequant;
//                     }
                    
//                     // ??softmax??
//                     score *= sm_scale * log2e;
//                     attention_scores[j] = score;
                    
//                     // ?????(???????softmax)
//                     if (score > max_score) {
//                         max_score = score;
//                     }
//                 }
                
//                 // Softmax??
//                 float sum_exp = 0.0f;
//                 for(uint32_t j = 0; j < kv_len; ++j) {
//                     // ?????softmax
//                     float shifted_score = attention_scores[j] - max_score;
//                     exp_scores[j] = std::exp(shifted_score);
//                     sum_exp += exp_scores[j];
//                 }
                
//                 // ????
//                 if (sum_exp < 1e-20f) {
//                     // ??sum_exp??,??????
//                     float uniform_val = 1.0f / kv_len;
//                     for(uint32_t j = 0; j < kv_len; ++j) {
//                         exp_scores[j] = uniform_val;
//                     }
//                 } else {
//                     // ???
//                     float sum_rcp = 1.0f / sum_exp;
//                     for(uint32_t j = 0; j < kv_len; ++j) {
//                         exp_scores[j] *= sum_rcp;
//                     }
//                 }
                
//                 // ??2: ???? O = softmax(QK^T) * V
//                 for(uint32_t h = 0; h < head_dim; ++h) {
//                     float output_val = 0.0f;
                    
//                     for(uint32_t j = 0; j < kv_len; ++j) {
//                         // ??V????
//                         uint32_t v_idx = vIndex(
//                             batch * num_kv_heads * kv_len + kv_head * kv_len + j,
//                             h, head_dim
//                         );
                        
//                         if (v_idx >= v_total_size) {
//                             continue;
//                         }
                        
//                         // ??V?(fp8???float)
//                         float8_fnuz_t v_val_fp8 = V[v_idx];
//                         float v_val = static_cast<float>(v_val_fp8);
                        
//                         // ???V??
//                         if (fuse_v_scale && V_scale) {
//                             // ??????
//                             uint32_t v_scale_idx = 0;
//                             v_val *= V_scale[v_scale_idx];
//                         }
                        
//                         // ?? attention * V
//                         output_val += exp_scores[j] * v_val;
//                     }
                    
//                     // ???????????
//                     uint32_t o_idx = oIndex(
//                         batch * num_qo_heads * qo_len + head * qo_len + i,
//                         h, head_dim
//                     );
                    
//                     if (o_idx < o_total_size) {
//                         O[o_idx] = output_val;
//                     }
//                 }
//             }
//         }
//     }
    
//     std::cout << "CPU computation completed." << std::endl;
    
// }

ROCWMMA_HOST void sage_attn_test(const uint32_t qo_len, const uint32_t kv_len, const uint32_t head_dim_t)
{
    // Runtime checks for host parameters
    uint32_t hTBLOCK_X    = gfx9Params::TBLOCK_X;
    uint32_t hTBLOCK_Y    = gfx9Params::TBLOCK_Y;
    uint32_t hBLOCKS_X    = gfx9Params::BLOCKS_X;
    uint32_t hBLOCKS_Y    = gfx9Params::BLOCKS_Y;
    uint32_t hROCWMMA_M   = gfx9Params::ROCWMMA_M;
    uint32_t hROCWMMA_N   = gfx9Params::ROCWMMA_N;
    uint32_t hROCWMMA_K   = gfx9Params::ROCWMMA_K;
    uint32_t hWARP_TILE_X = hBLOCKS_X * hROCWMMA_M;
    uint32_t hWARP_TILE_Y = hBLOCKS_Y * hROCWMMA_N;

    // Runtime warp calculation
    auto warpSize = getWarpSize();
    auto macroTileSize = rocwmma::make_coord2d(hTBLOCK_X / warpSize * hWARP_TILE_X, hTBLOCK_Y * hWARP_TILE_Y);
    
    // Device check for supported block and wave sizes
    if((isGfx11() || isGfx12()) && (hROCWMMA_M != 16 || hROCWMMA_N != 16))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }

    if(isGfx9() && (hROCWMMA_M != hROCWMMA_N) || (hROCWMMA_M != 16 && hROCWMMA_M != 32))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }

    if((isGfx11() || isGfx12()) && getWarpSize() != Constants::AMDGCN_WAVE_SIZE_32)
    {
        std::cout << "Unsupported wave size!\n";
        return;
    }

    if(isGfx9() && getWarpSize() != Constants::AMDGCN_WAVE_SIZE_64)
    {
        std::cout << "Unsupported wave size!\n";
        return;
    }

    // Bounds check
    if(( qo_len < get<0>(macroTileSize) || kv_len < get<1>(macroTileSize) || head_dim_t < hROCWMMA_K)
       || (qo_len % hROCWMMA_M || kv_len % hROCWMMA_N || head_dim_t % hROCWMMA_K))
    {
        std::cout << "Unsupported matrix size!\n";
        return;
    }

    // params for sage-attn
    //Q:[batch_size, num_qo_heads, qo_len, head_dim]
    //K:[batch_size, num_kv_heads, head_dim, kv_len]
    //V:[batch_size, num_kv_heads, kv_len, head_dim]
    //O:[batch_size, num_qo_heads, qo_len, head_dim]
    const uint32_t batch_size = 2;
    const uint32_t num_qo_heads = 30;
    const uint32_t num_kv_heads = 30;
    const float sm_scale = 0.125f;
    uint32_t head_dim = head_dim_t;

    const bool apply_silu = false;
    const bool fuse_v_scale = false;
    const bool fuse_v_mean = false;

    // Layouts leading dims
    int lda = std::is_same_v<DataLayoutA, row_major> ? head_dim : qo_len; // head_dim
    int ldb = std::is_same_v<DataLayoutB, row_major> ? kv_len : head_dim; // head_dim
    int ldv = std::is_same_v<DataLayoutV, row_major> ? head_dim : kv_len; // kv_len
    int ldd = std::is_same_v<DataLayoutC, row_major> ? head_dim : qo_len; // head_dim
    

 // Strides for batch and head dimensions
    const uint32_t stride_bz_q = qo_len * head_dim * num_qo_heads;
    const uint32_t stride_seq_q = head_dim;
    const uint32_t stride_h_q = qo_len * head_dim;
    
    const uint32_t stride_bz_k = num_kv_heads * kv_len * head_dim;
    const uint32_t stride_seq_k = head_dim;
    const uint32_t stride_h_k = kv_len * head_dim;
    
    const uint32_t stride_bz_v = num_kv_heads * kv_len * head_dim;
    const uint32_t stride_h_v = head_dim;
    const uint32_t stride_d_v = kv_len * head_dim;
    
    const uint32_t stride_bz_o = qo_len * head_dim * num_qo_heads;
    const uint32_t stride_seq_o = head_dim;
    const uint32_t stride_h_o = qo_len * head_dim;
    
    std::cout << "Initializing host data..." << std::endl;

    // Initialize input matrices
    std::vector<InputT> matrixQ(batch_size * num_qo_heads * qo_len * head_dim);
    std::vector<InputT> matrixK(batch_size * num_kv_heads * kv_len * head_dim); 
    std::vector<InputTV> matrixV(batch_size * num_kv_heads * kv_len * head_dim); 
    
    // Quantization scales
    uint32_t BLKQ=128;
    uint32_t WARPQ=32;
    uint32_t BLKK=64;
    uint32_t WARPK=64;

    std::vector<float> Q_scale(batch_size *num_qo_heads*(rocwmma::ceilDiv(qo_len, BLKQ) * (BLKQ / WARPQ) * 8));
    std::vector<float> K_scale(batch_size*num_kv_heads*(rocwmma::ceilDiv(kv_len, BLKK)  * (BLKK / WARPK) * 4));
    std::vector<float> V_scale(batch_size*num_kv_heads*(rocwmma::ceilDiv(kv_len, BLKK)  * (BLKK / WARPK) * 4));

    // Fill outputs with NaN to catch contamination
    std::vector<float> matrixO(batch_size * num_qo_heads * qo_len * head_dim, std::numeric_limits<float>::signaling_NaN());
    std::vector<float> matrixLse(batch_size * num_qo_heads * qo_len, -1.0f); // LSE output


    fillRand(matrixQ.data(), batch_size * num_qo_heads * qo_len, head_dim);
    fillRand(matrixK.data(), batch_size * num_kv_heads * kv_len, head_dim);
    fillRand(matrixV.data(), batch_size * num_kv_heads * kv_len, head_dim);

    // Initialize scales with reasonable values
    for (auto& scale : Q_scale) scale = 0.1f + (rand() % 100) / 1000.0f;
    for (auto& scale : K_scale) scale = 0.1f + (rand() % 100) / 1000.0f;
    for (auto& scale : V_scale) scale = 1.0f; // Typically 1.0 for non-quantized V

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    // Allocate and copy device memory
    InputT* d_q;
    InputT* d_k;
    InputTV* d_v;
    float* d_o;
    float* d_lse;
    float* d_q_scale;
    float* d_k_scale;
    float* d_v_scale;

    const size_t bytesQ = matrixQ.size() * sizeof(InputT);
    const size_t bytesK = matrixK.size() * sizeof(InputT);
    const size_t bytesV = matrixV.size() * sizeof(InputTV);
    const size_t bytesO = matrixO.size() * sizeof(float);
    const size_t bytesLse = matrixLse.size() * sizeof(float);
    const size_t bytesQScale = Q_scale.size() * sizeof(float);
    const size_t bytesKScale = K_scale.size() * sizeof(float);
    const size_t bytesVScale = V_scale.size() * sizeof(float);

    CHECK_HIP_ERROR(hipMalloc(&d_q, bytesQ));
    CHECK_HIP_ERROR(hipMalloc(&d_k, bytesK));
    CHECK_HIP_ERROR(hipMalloc(&d_v, bytesV));
    CHECK_HIP_ERROR(hipMalloc(&d_o, bytesO));
    CHECK_HIP_ERROR(hipMalloc(&d_lse, bytesLse));
    CHECK_HIP_ERROR(hipMalloc(&d_q_scale, bytesQScale));
    CHECK_HIP_ERROR(hipMalloc(&d_k_scale, bytesKScale));
    CHECK_HIP_ERROR(hipMalloc(&d_v_scale, bytesVScale));

    CHECK_HIP_ERROR(hipMemcpy(d_q, matrixQ.data(), bytesQ, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_k, matrixK.data(), bytesK, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_v, matrixV.data(), bytesV, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_q_scale, Q_scale.data(), bytesQScale, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_k_scale, K_scale.data(), bytesKScale, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_v_scale, V_scale.data(), bytesVScale, hipMemcpyHostToDevice));

    auto blockDim = dim3(hTBLOCK_X, hTBLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceilDiv(qo_len, get<0>(macroTileSize)), num_qo_heads, batch_size);

    std::cout << "Launching Attention kernel..." << std::endl;
    std::cout << "gridDim (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")"
              << " blockDim (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;
    constexpr int SV_ITERS = 1;
    // Uses 2 lds blocks for prefetch loop (A and B)
    size_t smem_max = std::max(2u * sizeof(int8_t) * (64 + 64) * 32, 2u * sizeof(float) * 1 * 64);

    using KernelFunc = void(*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                          int8_t*, int8_t*, InputTV*, float*, float*,
                          float*, float*, float*, float*,
                          uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                          uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                          uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,  float);

    auto kernel_ptr = qk_int_sv_f8_attn_kernel<64,
                                               64,
                                               QuantGranularity::kPerThread,
                                               QuantGranularity::kPerThread,
                                               SV_ITERS>;

    hipError_t attr_result = hipFuncSetAttribute(
                                                reinterpret_cast<const void*>(kernel_ptr),
                                                hipFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(smem_max));

    if(attr_result != hipSuccess)
    {
        std::cout << "hipFuncSetAttribute failed: " << hipGetErrorString(attr_result) << std::endl;
        return;
    }
    // Launch GPU kernel
    const int num_kv_groups = num_qo_heads / num_kv_heads;
    auto gpuKernel = [&]() {
        hipLaunchKernelGGL((qk_int_sv_f8_attn_kernel<
                            64,64,QuantGranularity::kPerThread,QuantGranularity::kPerThread,1>),
            dim3(gridDim), dim3(blockDim), smem_max, 0,
            head_dim, lda, ldb, ldv, ldd,
            d_q, d_k, d_v, d_o, d_lse,
            d_q_scale, d_k_scale, d_v_scale, nullptr,
            qo_len, kv_len, kv_len, num_kv_groups,
            stride_bz_q, stride_seq_q, stride_h_q,
            stride_bz_k, stride_seq_k, stride_h_k,
            stride_bz_v, stride_h_v, stride_d_v,
            stride_bz_o, stride_seq_o, stride_h_o,
            sm_scale);
    };

    constexpr uint32_t warmups = 2u;
    constexpr uint32_t recordRuns = 5u;

    // Warm-up runs, not recorded
    for(uint32_t i = 0; i < warmups; ++i)
    {
        gpuKernel();
    }

    // Actual recorded runs
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    for(uint32_t i = 0; i < recordRuns; ++i)
    {
        gpuKernel();
        CHECK_HIP_ERROR(hipGetLastError());
        CHECK_HIP_ERROR(hipStreamSynchronize(0));
    }
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));

    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // Echo performance
    std::cout << "TBlockX, TBlockY, " << "BlocksX, BlocksY, " << "BlkM, BlkN, BlkK, "
              << "MatM, MatN, MatK, " << "lda, ldb, " << "ldv, ldd, "
              << "elapsedMs" << "," << "sizeof(InputT)"
              << std::endl;

    std::cout << hTBLOCK_X << ", " << hTBLOCK_Y << ", " << hBLOCKS_X << ", " << hBLOCKS_Y << ", "
              << hROCWMMA_M << ", " << hROCWMMA_N << ", " << hROCWMMA_K << ", " << qo_len << ", " << head_dim
              << ", " << kv_len << ", " << lda << ", " << ldb << ", "
              << ldv << ", " << ldd << ", " << elapsedTimeMs << "," << sizeof(InputT) << std::endl;
// #if !NDEBUG
#if 0
    std::cout << "Validating result with reference..." << std::endl;

    if((uint64_t)qo_len * (uint64_t)kv_len * (uint64_t)head_dim > (2048ull * 2048ull * 2048ull))
    {
        std::cout << "Please wait. Large sizes can take a while!" << std::endl;
    }

    std::vector<float> cpu_results(matrixO.size());
    sage_attn_cpu<InputT, InputTV, OutputT, ComputeT, DataLayoutA, DataLayoutB, DataLayoutC>(
        qo_len, head_dim, kv_len,
        matrixQ.data(), matrixK.data(), matrixV.data(), cpu_results.data(),
        Q_scale.data(), K_scale.data(), V_scale.data(), nullptr,
        lda, ldb, ldv, ldd,
        batch_size, num_qo_heads, num_kv_heads, sm_scale,
        fuse_v_scale, fuse_v_mean);

    // Copy GPU results back
    std::vector<float> gpu_results(matrixO.size());
    CHECK_HIP_ERROR(hipMemcpy(gpu_results.data(), d_o, bytesO, hipMemcpyDeviceToHost));
    
    printf("gpu + cpu result:\n");
    for(int i= 0; i < matrixO.size(); i++)
        printf("i=%d, gpu_res=%f, cpu_res=%f\n",i, gpu_results[i], cpu_results[i]);  

    auto res = compareEqual(gpu_results.data(), cpu_results.data(), qo_len * head_dim);

    if(std::get<0>(res) == false)
    {
        std::cout << "FAILED\n";
    }
    else
    {
        std::cout << "PASSED\n";
    }

    std::cout << "Max relative error: " << std::get<1>(res) << std::endl;

#endif // !NDEBUG

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_q));
    CHECK_HIP_ERROR(hipFree(d_k));
    CHECK_HIP_ERROR(hipFree(d_v));
    CHECK_HIP_ERROR(hipFree(d_o));
    CHECK_HIP_ERROR(hipFree(d_lse));
    CHECK_HIP_ERROR(hipFree(d_q_scale));
    CHECK_HIP_ERROR(hipFree(d_k_scale));
    CHECK_HIP_ERROR(hipFree(d_v_scale));

    std::cout << "Finished!" << std::endl;
}

int main()
{
    if(isGfx9())
        sage_attn_test(17920, 17920, 64); // qo_len, kv_len, head_dim
    else
        std::cout << "This sample Not test on gfx11/gfx12 yet!" << std::endl;
    return 0;
}

// copy from:
// https://github.com/pytorch/pytorch/blob/5b815d980eb30a2711b7caecf2cf60dc96eb9b91/aten/src/THC/THCReduce.cuh

#pragma once
#include <stdio.h>

namespace amirstan{
namespace cuda{

#define MAX_GRID_SIZE 65535LL

template <typename T>
__host__ __device__ __forceinline__ T ceilDiv(T a, T b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ size_t getLinearBlockId() {
    return blockIdx.z * gridDim.y * gridDim.x +
      blockIdx.y * gridDim.x +
      blockIdx.x;
}

template <typename T>
struct divScalarOp {
    T N;
    divScalarOp(T N):N(N){}
    inline __device__ T operator()(const T& x){
        return x/N;
}
};

template <typename T>
struct squareOp {
inline __device__ T operator()(const T& x){
    return x*x;
}
};

template <typename T>
struct reduceSumOp {
inline __device__ T operator()(const T& x, const T &y){
    return x+y;
}
};

// Reduce N values concurrently, i.e. suppose N = 2, and there are 4 threads:
// (1, 2), (3, 4), (5, 6), (7, 8), then the return in threadVals for thread 0
// is (1 + 3 + 5 + 7, 2 + 4 + 6 + 8) = (16, 20)
//
// If smem is not used again, there is no need to __syncthreads before this
// call. However, if smem will be used, e.g., this function is called in a loop,
// then __syncthreads is needed either before or afterwards to prevent non-0
// threads overriding smem in the next loop before num-0 thread reads from it.
template <typename T, typename ReduceOp, int N>
__device__ void reduceNValuesInBlock(T *smem,
                             T threadVals[N],
                             const unsigned int numVals,
                             ReduceOp reduceOp,
                             T init) {
  if (numVals == 0) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      threadVals[i] = init;
    }
    return;
  }

  // We store each of the N values contiguously, so if N = 2, all values for
  // the first threadVal for each thread in the block are stored followed by
  // all of the values for the second threadVal for each thread in the block
  if (threadIdx.x < numVals) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      smem[i * numVals + threadIdx.x] = threadVals[i];
    }
  }
  __syncthreads();

  // Number of lanes in the final reduction --> this is used to determine
  // where to put the outputs of each of the n things we are reducing. If
  // nLP = 32, then we have the 32 outputs for the first threadVal,
  // followed by the 32 outputs for the second threadVal, etc.
  const unsigned int numLanesParticipating = min(numVals, warpSize);

  if (numVals > warpSize && ((threadIdx.x / warpSize) == 0 )) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      threadVals[i] = threadIdx.x < numVals ? threadVals[i] : init;
    }

    for (int i = warpSize + threadIdx.x; i < numVals; i += warpSize) {
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        threadVals[j] = reduceOp(threadVals[j], smem[j * numVals + i]);
      }
    }

    #pragma unroll
    for (int i = 0; i < N; ++i) {
      smem[i * numLanesParticipating + threadIdx.x] = threadVals[i];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (numLanesParticipating == 32) {
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int j = 1; j < 32; ++j) {
          threadVals[i] = reduceOp(threadVals[i], smem[i * 32 + j]);
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        for (int j = 1; j < numLanesParticipating; ++j) {
          threadVals[i] = reduceOp(threadVals[i], smem[i * numVals + j]);
        }
      }
    }
  }
}


template <typename T,
          typename AccT,
          typename ModifyOp,
          typename ReduceOp,
          typename FinalizeOp>
__global__ void reduceContigous2DKernel(T* out,const T* in,
                                        size_t reductionSize, size_t totalSlices,
                                        AccT init,
                                        ModifyOp modifyOp,
                                        ReduceOp reduceOp,
                                        FinalizeOp finalizeOp){
    const size_t sliceIndex = getLinearBlockId();
    if (sliceIndex >= totalSlices) {
      return;
    }

    const size_t outOffset = sliceIndex;
    const size_t inBaseOffset = sliceIndex * reductionSize;

    AccT r = init;
    
    for (size_t i = threadIdx.x; i < reductionSize; i += blockDim.x) {
        const AccT val = static_cast<AccT>(in[inBaseOffset + i]);
        r = reduceOp(r, modifyOp(val));
    }
    
    extern __shared__ char smemChar[];
    AccT* smem = (AccT*) smemChar;

    reduceNValuesInBlock<AccT, ReduceOp, 1>(smem, &r, blockDim.x, reduceOp, init);

    if (threadIdx.x == 0) {
        // Write out reduced value
        out[outOffset] = static_cast<T>(finalizeOp(r));
    }

}

bool getGridFromTiles(size_t gridTiles, dim3& grid) {
    if (gridTiles > MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE) {
        return false;
    }

    size_t gridX = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
    size_t gridY = 1;
    size_t gridZ = 1;

    if (gridTiles > MAX_GRID_SIZE) {
        gridTiles = ceilDiv(gridTiles, (size_t) MAX_GRID_SIZE);
        gridY = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;

        if (gridTiles > MAX_GRID_SIZE) {
        gridTiles = ceilDiv(gridTiles, (size_t) MAX_GRID_SIZE);
        gridZ = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
        }
    }

    grid = dim3(gridX, gridY, gridZ);
    return true;
}
    
inline dim3 getContigReduceBlock(size_t numSlices, size_t reductionSize) {
    // If the number of slices is low but the reduction dimension size
    // is high, then we should increase block size for greater parallelism.
    // Aim for at least 32 warps per SM (assume 15 SMs; don't bother
    // inquiring the real number for now).
    int maxWarps = 4; // better occupancy if many blocks are around
    // For numSlices > 15 * 8, there are > 32 warps active per SM.
    if (numSlices < 15 * 8) {
      maxWarps = 8;
      if (numSlices < 15 * 4) {
        maxWarps = 16;
        if (numSlices < 15 * 2) {
          maxWarps = 32;
        }
      }
    }

  // Scale up block size based on the reduction dimension size
  size_t warpsInReductionSize = ceilDiv(reductionSize, (size_t) 32);
  int numWarps = warpsInReductionSize > (size_t) maxWarps ?
    maxWarps : (int) warpsInReductionSize;

  return dim3(numWarps * 32);
}

template <
typename T,
typename AccT,
typename ModifyOp,
typename ReduceOp,
typename FinalizeOp>
bool reduce2DContigous(T* out,
                        const T* in,
                        AccT init,
                        size_t inElements,
                        size_t outElements,
                        const ModifyOp modifyOp,
                        const ReduceOp reduceOp,
                        const FinalizeOp finalizeOp,
                        cudaStream_t stream) {
    
    size_t reductionSize = inElements/outElements;
    dim3 block;
    dim3 grid;
    int smemSize = 0;

    getGridFromTiles(outElements, grid);
    block = getContigReduceBlock(outElements, reductionSize);
    smemSize = sizeof(AccT) * block.x;

    reduceContigous2DKernel<T,AccT, ModifyOp,ReduceOp, FinalizeOp>
    <<<grid, block, smemSize, stream>>>(
        out, in, 
        reductionSize, outElements, init,
        modifyOp, reduceOp, finalizeOp
    );

    return true;
}


}
}
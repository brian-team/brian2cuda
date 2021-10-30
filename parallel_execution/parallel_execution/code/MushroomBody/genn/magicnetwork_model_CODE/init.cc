#include "definitionsInternal.h"
#include <iostream>
#include <random>
#include <cstdint>

struct MergedNeuronInitGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    double* inSynInSyn0;
    double* inSynInSyn1;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup2
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    double* inSynInSyn0;
    unsigned int numNeurons;
    
}
;
struct MergedSynapseSparseInitGroup0
 {
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int* colLength;
    unsigned int* remap;
    unsigned int rowStride;
    unsigned int colStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
__device__ __constant__ MergedNeuronInitGroup0 d_mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int numNeurons) {
    MergedNeuronInitGroup0 group = {spkCnt, spk, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup0, &group, sizeof(MergedNeuronInitGroup0), idx * sizeof(MergedNeuronInitGroup0)));
}
__device__ __constant__ MergedNeuronInitGroup1 d_mergedNeuronInitGroup1[1];
void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, double* inSynInSyn0, double* inSynInSyn1, unsigned int numNeurons) {
    MergedNeuronInitGroup1 group = {spkCnt, spk, inSynInSyn0, inSynInSyn1, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup1, &group, sizeof(MergedNeuronInitGroup1), idx * sizeof(MergedNeuronInitGroup1)));
}
__device__ __constant__ MergedNeuronInitGroup2 d_mergedNeuronInitGroup2[1];
void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, double* inSynInSyn0, unsigned int numNeurons) {
    MergedNeuronInitGroup2 group = {spkCnt, spk, inSynInSyn0, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup2, &group, sizeof(MergedNeuronInitGroup2), idx * sizeof(MergedNeuronInitGroup2)));
}
__device__ __constant__ MergedSynapseSparseInitGroup0 d_mergedSynapseSparseInitGroup0[1];
void pushMergedSynapseSparseInitGroup0ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, unsigned int* colLength, unsigned int* remap, unsigned int rowStride, unsigned int colStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseSparseInitGroup0 group = {rowLength, ind, colLength, remap, rowStride, colStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseSparseInitGroup0, &group, sizeof(MergedSynapseSparseInitGroup0), idx * sizeof(MergedSynapseSparseInitGroup0)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ unsigned int d_mergedNeuronInitGroupStartID0[] = {0, };
__device__ unsigned int d_mergedNeuronInitGroupStartID1[] = {128, };
__device__ unsigned int d_mergedNeuronInitGroupStartID2[] = {256, };
__device__ unsigned int d_mergedSynapseSparseInitGroupStartID0[] = {0, };

extern "C" __global__ void initializeKernel(unsigned long long deviceRNGSeed) {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    if(id < 128) {
        struct MergedNeuronInitGroup0 *group = &d_mergedNeuronInitGroup0[0]; 
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
            // current source variables
        }
    }
    // merged1
    if(id >= 128 && id < 256) {
        struct MergedNeuronInitGroup1 *group = &d_mergedNeuronInitGroup1[0]; 
        const unsigned int lid = id - 128;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
             {
                group->inSynInSyn0[lid] = 0.000000;
            }
             {
                group->inSynInSyn1[lid] = 0.000000;
            }
            // current source variables
        }
    }
    // merged2
    if(id >= 256 && id < 2784) {
        struct MergedNeuronInitGroup2 *group = &d_mergedNeuronInitGroup2[0]; 
        const unsigned int lid = id - 256;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
             {
                group->inSynInSyn0[lid] = 0.000000;
            }
            // current source variables
        }
    }
    
    // ------------------------------------------------------------------------
    // Custom update groups
    
    // ------------------------------------------------------------------------
    // Custom WU update groups with dense connectivity
    
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
    
}
extern "C" __global__ void initializeSparseKernel() {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    __shared__ unsigned int shRowLength[32];
    // merged0
    if(id < 128) {
        struct MergedSynapseSparseInitGroup0 *group = &d_mergedSynapseSparseInitGroup0[0]; 
        const unsigned int lid = id - 0;
        const unsigned int numBlocks = (group->numSrcNeurons + 32 - 1) / 32;
        unsigned int idx = lid;
        for(unsigned int r = 0; r < numBlocks; r++) {
            const unsigned numRowsInBlock = (r == (numBlocks - 1)) ? ((group->numSrcNeurons - 1) % 32) + 1 : 32;
            __syncthreads();
            if (threadIdx.x < numRowsInBlock) {
                shRowLength[threadIdx.x] = group->rowLength[(r * 32) + threadIdx.x];
            }
            __syncthreads();
            for(unsigned int i = 0; i < numRowsInBlock; i++) {
                if(lid < shRowLength[i]) {
                     {
                        const unsigned int postIndex = group->ind[idx];
                        const unsigned int colLocation = atomicAdd(&group->colLength[postIndex], 1);
                        const unsigned int colMajorIndex = (postIndex * group->colStride) + colLocation;
                        group->remap[colMajorIndex] = idx;
                    }
                }
                idx += group->rowStride;
            }
        }
    }
}
void initialize() {
    unsigned long long deviceRNGSeed = 0;
    CHECK_CUDA_ERRORS(cudaMemset(d_colLengthsynapses_1, 0, 100 * sizeof(unsigned int)));
     {
        const dim3 threads(32, 1);
        const dim3 grid(87, 1);
        initializeKernel<<<grid, threads>>>(deviceRNGSeed);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}

void initializeSparse() {
    copyStateToDevice(true);
    copyConnectivityToDevice(true);
    
     {
        const dim3 threads(32, 1);
        const dim3 grid(4, 1);
        initializeSparseKernel<<<grid, threads>>>();
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}

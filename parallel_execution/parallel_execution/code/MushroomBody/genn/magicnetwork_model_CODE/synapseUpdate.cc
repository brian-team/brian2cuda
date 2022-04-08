#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedPresynapticUpdateGroup0
 {
    double* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedPresynapticUpdateGroup1
 {
    double* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    unsigned int* rowLength;
    uint32_t* ind;
    double* lastupdate;
    double* Apost;
    double* g_raw;
    double* Apre;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedPresynapticUpdateGroup2
 {
    double* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    unsigned int* rowLength;
    uint32_t* ind;
    double* weight;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedPostsynapticUpdateGroup0
 {
    unsigned int* trgSpkCnt;
    unsigned int* trgSpk;
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int* colLength;
    unsigned int* remap;
    double* lastupdate;
    double* Apost;
    double* g_raw;
    double* Apre;
    unsigned int rowStride;
    unsigned int colStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
__device__ __constant__ MergedPresynapticUpdateGroup0 d_mergedPresynapticUpdateGroup0[1];
void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, double* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedPresynapticUpdateGroup0 group = {inSyn, srcSpkCnt, srcSpk, rowLength, ind, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup0, &group, sizeof(MergedPresynapticUpdateGroup0), idx * sizeof(MergedPresynapticUpdateGroup0)));
}
__device__ __constant__ MergedPresynapticUpdateGroup1 d_mergedPresynapticUpdateGroup1[1];
void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, double* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* rowLength, uint32_t* ind, double* lastupdate, double* Apost, double* g_raw, double* Apre, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedPresynapticUpdateGroup1 group = {inSyn, srcSpkCnt, srcSpk, rowLength, ind, lastupdate, Apost, g_raw, Apre, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup1, &group, sizeof(MergedPresynapticUpdateGroup1), idx * sizeof(MergedPresynapticUpdateGroup1)));
}
__device__ __constant__ MergedPresynapticUpdateGroup2 d_mergedPresynapticUpdateGroup2[1];
void pushMergedPresynapticUpdateGroup2ToDevice(unsigned int idx, double* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* rowLength, uint32_t* ind, double* weight, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedPresynapticUpdateGroup2 group = {inSyn, srcSpkCnt, srcSpk, rowLength, ind, weight, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup2, &group, sizeof(MergedPresynapticUpdateGroup2), idx * sizeof(MergedPresynapticUpdateGroup2)));
}
__device__ __constant__ MergedPostsynapticUpdateGroup0 d_mergedPostsynapticUpdateGroup0[1];
void pushMergedPostsynapticUpdateGroup0ToDevice(unsigned int idx, unsigned int* trgSpkCnt, unsigned int* trgSpk, unsigned int* rowLength, uint32_t* ind, unsigned int* colLength, unsigned int* remap, double* lastupdate, double* Apost, double* g_raw, double* Apre, unsigned int rowStride, unsigned int colStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedPostsynapticUpdateGroup0 group = {trgSpkCnt, trgSpk, rowLength, ind, colLength, remap, lastupdate, Apost, g_raw, Apre, rowStride, colStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPostsynapticUpdateGroup0, &group, sizeof(MergedPostsynapticUpdateGroup0), idx * sizeof(MergedPostsynapticUpdateGroup0)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID1[] = {128, };
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID2[] = {256, };
__device__ __constant__ unsigned int d_mergedPostsynapticUpdateGroupStartID0[] = {0, };
extern "C" __global__ void updatePresynapticKernel(double t)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shRowLength[32];
    __shared__ unsigned int shSpk[32];
    // merged0
    if(id < 128) {
        struct MergedPresynapticUpdateGroup0 *group = &d_mergedPresynapticUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        using namespace PresynapticUpdateSupportCode0;
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            atomicAdd(&group->inSyn[ipost], ((6.75000000000000044e-01) * (7.50000000000000098e-08)));}
                    }
                }
            }
        }
        
    }
    // merged1
    if(id >= 128 && id < 256) {
        struct MergedPresynapticUpdateGroup1 *group = &d_mergedPresynapticUpdateGroup1[0]; 
        const unsigned int lid = id - 128;
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        using namespace PresynapticUpdateSupportCode1;
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            double _Apost = group->Apost[synAddress] * exp(1.0*(- (t - group->lastupdate[synAddress]))/(1.00000000000000002e-02));
                            double _Apre = group->Apre[synAddress] * exp(1.0*(- (t - group->lastupdate[synAddress]))/(1.00000000000000002e-02));
                            group->Apost[synAddress] = _Apost;
                            group->Apre[synAddress] = _Apre;
                            atomicAdd(&group->inSyn[ipost], group->g_raw[synAddress]);
                            group->Apre[synAddress] += (1.00000000000000017e-10);
                            group->g_raw[synAddress] = _clip(group->g_raw[synAddress] + group->Apost[synAddress], 0 * (1.00000000000000000e+00), (3.75000000000000049e-09));
                            group->lastupdate[synAddress] = t;}
                    }
                }
            }
        }
        
    }
    // merged2
    if(id >= 256 && id < 704) {
        struct MergedPresynapticUpdateGroup2 *group = &d_mergedPresynapticUpdateGroup2[0]; 
        const unsigned int lid = id - 256;
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        using namespace PresynapticUpdateSupportCode0;
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            atomicAdd(&group->inSyn[ipost], ((6.75000000000000044e-01) * group->weight[synAddress]));}
                    }
                }
            }
        }
        
    }
}
extern "C" __global__ void updatePostsynapticKernel(double t)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shSpk[32];
    __shared__ unsigned int shColLength[32];
    // merged0
    if(id < 2528) {
        struct MergedPostsynapticUpdateGroup0 *group = &d_mergedPostsynapticUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
        const unsigned int numSpikes = group->trgSpkCnt[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (threadIdx.x < numSpikesInBlock) {
                const unsigned int spk = group->trgSpk[(r * 32) + threadIdx.x];
                shSpk[threadIdx.x] = spk;
                shColLength[threadIdx.x] = group->colLength[spk];
            }
            __syncthreads();
            // only work on existing neurons
            if (lid < group->colStride) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    if (lid < shColLength[j]) {
                        const unsigned int synAddress = group->remap[(shSpk[j] * group->colStride) + lid];
                        const unsigned int ipre = synAddress / group->rowStride;
                        using namespace PostsynapticUpdateSupportCode0;
                        double _Apost = group->Apost[synAddress] * exp(1.0*(- (t - group->lastupdate[synAddress]))/(1.00000000000000002e-02));
                        double _Apre = group->Apre[synAddress] * exp(1.0*(- (t - group->lastupdate[synAddress]))/(1.00000000000000002e-02));
                        group->Apost[synAddress] = _Apost;
                        group->Apre[synAddress] = _Apre;
                        group->Apost[synAddress] += (-1.00000000000000017e-10);
                        group->g_raw[synAddress] = _clip(group->g_raw[synAddress] + group->Apre[synAddress], 0 * (1.00000000000000000e+00), (3.75000000000000049e-09));
                        group->lastupdate[synAddress] = t;}
                }
            }
        }
    }
}
void updateSynapses(double t) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(22, 1);
        updatePresynapticKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
     {
        const dim3 threads(32, 1);
        const dim3 grid(79, 1);
        updatePostsynapticKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}

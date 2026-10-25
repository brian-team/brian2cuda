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
    double* g_raw;
    double* Apost;
    double* Apre;
    double* lastupdate;
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
    double* g_raw;
    double* Apost;
    double* Apre;
    double* lastupdate;
    unsigned int rowStride;
    unsigned int colStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
static MergedPresynapticUpdateGroup0 mergedPresynapticUpdateGroup0[1];
void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, double* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    mergedPresynapticUpdateGroup0[idx].inSyn = inSyn;
    mergedPresynapticUpdateGroup0[idx].srcSpkCnt = srcSpkCnt;
    mergedPresynapticUpdateGroup0[idx].srcSpk = srcSpk;
    mergedPresynapticUpdateGroup0[idx].rowLength = rowLength;
    mergedPresynapticUpdateGroup0[idx].ind = ind;
    mergedPresynapticUpdateGroup0[idx].rowStride = rowStride;
    mergedPresynapticUpdateGroup0[idx].numSrcNeurons = numSrcNeurons;
    mergedPresynapticUpdateGroup0[idx].numTrgNeurons = numTrgNeurons;
}
static MergedPresynapticUpdateGroup1 mergedPresynapticUpdateGroup1[1];
void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, double* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* rowLength, uint32_t* ind, double* g_raw, double* Apost, double* Apre, double* lastupdate, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    mergedPresynapticUpdateGroup1[idx].inSyn = inSyn;
    mergedPresynapticUpdateGroup1[idx].srcSpkCnt = srcSpkCnt;
    mergedPresynapticUpdateGroup1[idx].srcSpk = srcSpk;
    mergedPresynapticUpdateGroup1[idx].rowLength = rowLength;
    mergedPresynapticUpdateGroup1[idx].ind = ind;
    mergedPresynapticUpdateGroup1[idx].g_raw = g_raw;
    mergedPresynapticUpdateGroup1[idx].Apost = Apost;
    mergedPresynapticUpdateGroup1[idx].Apre = Apre;
    mergedPresynapticUpdateGroup1[idx].lastupdate = lastupdate;
    mergedPresynapticUpdateGroup1[idx].rowStride = rowStride;
    mergedPresynapticUpdateGroup1[idx].numSrcNeurons = numSrcNeurons;
    mergedPresynapticUpdateGroup1[idx].numTrgNeurons = numTrgNeurons;
}
static MergedPresynapticUpdateGroup2 mergedPresynapticUpdateGroup2[1];
void pushMergedPresynapticUpdateGroup2ToDevice(unsigned int idx, double* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* rowLength, uint32_t* ind, double* weight, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    mergedPresynapticUpdateGroup2[idx].inSyn = inSyn;
    mergedPresynapticUpdateGroup2[idx].srcSpkCnt = srcSpkCnt;
    mergedPresynapticUpdateGroup2[idx].srcSpk = srcSpk;
    mergedPresynapticUpdateGroup2[idx].rowLength = rowLength;
    mergedPresynapticUpdateGroup2[idx].ind = ind;
    mergedPresynapticUpdateGroup2[idx].weight = weight;
    mergedPresynapticUpdateGroup2[idx].rowStride = rowStride;
    mergedPresynapticUpdateGroup2[idx].numSrcNeurons = numSrcNeurons;
    mergedPresynapticUpdateGroup2[idx].numTrgNeurons = numTrgNeurons;
}
static MergedPostsynapticUpdateGroup0 mergedPostsynapticUpdateGroup0[1];
void pushMergedPostsynapticUpdateGroup0ToDevice(unsigned int idx, unsigned int* trgSpkCnt, unsigned int* trgSpk, unsigned int* rowLength, uint32_t* ind, unsigned int* colLength, unsigned int* remap, double* g_raw, double* Apost, double* Apre, double* lastupdate, unsigned int rowStride, unsigned int colStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    mergedPostsynapticUpdateGroup0[idx].trgSpkCnt = trgSpkCnt;
    mergedPostsynapticUpdateGroup0[idx].trgSpk = trgSpk;
    mergedPostsynapticUpdateGroup0[idx].rowLength = rowLength;
    mergedPostsynapticUpdateGroup0[idx].ind = ind;
    mergedPostsynapticUpdateGroup0[idx].colLength = colLength;
    mergedPostsynapticUpdateGroup0[idx].remap = remap;
    mergedPostsynapticUpdateGroup0[idx].g_raw = g_raw;
    mergedPostsynapticUpdateGroup0[idx].Apost = Apost;
    mergedPostsynapticUpdateGroup0[idx].Apre = Apre;
    mergedPostsynapticUpdateGroup0[idx].lastupdate = lastupdate;
    mergedPostsynapticUpdateGroup0[idx].rowStride = rowStride;
    mergedPostsynapticUpdateGroup0[idx].colStride = colStride;
    mergedPostsynapticUpdateGroup0[idx].numSrcNeurons = numSrcNeurons;
    mergedPostsynapticUpdateGroup0[idx].numTrgNeurons = numTrgNeurons;
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void updateSynapses(double t) {
     {
        // merged presynaptic update group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedPresynapticUpdateGroup0[g]; 
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group->srcSpkCnt[0]; i++) {
                using namespace PresynapticUpdateSupportCode0;
                const unsigned int ipre = group->srcSpk[i];
                const unsigned int npost = group->rowLength[ipre];
                for (unsigned int j = 0; j < npost; j++) {
                    const unsigned int synAddress = (ipre * group->rowStride) + j;
                    const unsigned int ipost = group->ind[synAddress];
                    group->inSyn[ipost] += ((6.75000000000000044e-01) * (7.50000000000000098e-08));}
            }
            
        }
    }
     {
        // merged presynaptic update group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedPresynapticUpdateGroup1[g]; 
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group->srcSpkCnt[0]; i++) {
                using namespace PresynapticUpdateSupportCode1;
                const unsigned int ipre = group->srcSpk[i];
                const unsigned int npost = group->rowLength[ipre];
                for (unsigned int j = 0; j < npost; j++) {
                    const unsigned int synAddress = (ipre * group->rowStride) + j;
                    const unsigned int ipost = group->ind[synAddress];
                    double _Apost = group->Apost[synAddress] * exp(1.0*(- (t - group->lastupdate[synAddress]))/(1.00000000000000002e-02));
                    double _Apre = group->Apre[synAddress] * exp(1.0*(- (t - group->lastupdate[synAddress]))/(1.00000000000000002e-02));
                    group->Apost[synAddress] = _Apost;
                    group->Apre[synAddress] = _Apre;
                    group->inSyn[ipost] += group->g_raw[synAddress];
                    group->Apre[synAddress] += (1.00000000000000017e-10);
                    group->g_raw[synAddress] = _clip(group->g_raw[synAddress] + group->Apost[synAddress], 0 * (1.00000000000000000e+00), (3.75000000000000049e-09));
                    group->lastupdate[synAddress] = t;}
            }
            
        }
    }
     {
        // merged presynaptic update group 2
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedPresynapticUpdateGroup2[g]; 
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group->srcSpkCnt[0]; i++) {
                using namespace PresynapticUpdateSupportCode0;
                const unsigned int ipre = group->srcSpk[i];
                const unsigned int npost = group->rowLength[ipre];
                for (unsigned int j = 0; j < npost; j++) {
                    const unsigned int synAddress = (ipre * group->rowStride) + j;
                    const unsigned int ipost = group->ind[synAddress];
                    group->inSyn[ipost] += ((6.75000000000000044e-01) * group->weight[synAddress]);}
            }
            
        }
    }
     {
        // merged postsynaptic update group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedPostsynapticUpdateGroup0[g]; 
            const unsigned int numSpikes = group->trgSpkCnt[0];
            for (unsigned int j = 0; j < numSpikes; j++) {
                const unsigned int spike = group->trgSpk[j];
                const unsigned int npre = group->colLength[spike];
                for (unsigned int i = 0; i < npre; i++) {
                    const unsigned int colMajorIndex = (spike * group->colStride) + i;
                    const unsigned int rowMajorIndex = group->remap[colMajorIndex];
                    using namespace PostsynapticUpdateSupportCode0;
                    double _Apost = group->Apost[rowMajorIndex] * exp(1.0*(- (t - group->lastupdate[rowMajorIndex]))/(1.00000000000000002e-02));
                    double _Apre = group->Apre[rowMajorIndex] * exp(1.0*(- (t - group->lastupdate[rowMajorIndex]))/(1.00000000000000002e-02));
                    group->Apost[rowMajorIndex] = _Apost;
                    group->Apre[rowMajorIndex] = _Apre;
                    group->Apost[rowMajorIndex] += (-1.00000000000000017e-10);
                    group->g_raw[rowMajorIndex] = _clip(group->g_raw[rowMajorIndex] + group->Apre[rowMajorIndex], 0 * (1.00000000000000000e+00), (3.75000000000000049e-09));
                    group->lastupdate[rowMajorIndex] = t;}
            }
            
        }
    }
}

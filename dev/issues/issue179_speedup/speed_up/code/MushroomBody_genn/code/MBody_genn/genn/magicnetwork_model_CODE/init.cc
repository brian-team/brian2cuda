#include "definitionsInternal.h"
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
static MergedNeuronInitGroup0 mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int numNeurons) {
    mergedNeuronInitGroup0[idx].spkCnt = spkCnt;
    mergedNeuronInitGroup0[idx].spk = spk;
    mergedNeuronInitGroup0[idx].numNeurons = numNeurons;
}
static MergedNeuronInitGroup1 mergedNeuronInitGroup1[1];
void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, double* inSynInSyn0, double* inSynInSyn1, unsigned int numNeurons) {
    mergedNeuronInitGroup1[idx].spkCnt = spkCnt;
    mergedNeuronInitGroup1[idx].spk = spk;
    mergedNeuronInitGroup1[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronInitGroup1[idx].inSynInSyn1 = inSynInSyn1;
    mergedNeuronInitGroup1[idx].numNeurons = numNeurons;
}
static MergedNeuronInitGroup2 mergedNeuronInitGroup2[1];
void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, double* inSynInSyn0, unsigned int numNeurons) {
    mergedNeuronInitGroup2[idx].spkCnt = spkCnt;
    mergedNeuronInitGroup2[idx].spk = spk;
    mergedNeuronInitGroup2[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronInitGroup2[idx].numNeurons = numNeurons;
}
static MergedSynapseSparseInitGroup0 mergedSynapseSparseInitGroup0[1];
void pushMergedSynapseSparseInitGroup0ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, unsigned int* colLength, unsigned int* remap, unsigned int rowStride, unsigned int colStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    mergedSynapseSparseInitGroup0[idx].rowLength = rowLength;
    mergedSynapseSparseInitGroup0[idx].ind = ind;
    mergedSynapseSparseInitGroup0[idx].colLength = colLength;
    mergedSynapseSparseInitGroup0[idx].remap = remap;
    mergedSynapseSparseInitGroup0[idx].rowStride = rowStride;
    mergedSynapseSparseInitGroup0[idx].colStride = colStride;
    mergedSynapseSparseInitGroup0[idx].numSrcNeurons = numSrcNeurons;
    mergedSynapseSparseInitGroup0[idx].numTrgNeurons = numTrgNeurons;
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void initialize() {
    // ------------------------------------------------------------------------
    // Local neuron groups
     {
        // merged neuron init group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup0[g]; 
            group->spkCnt[0] = 0;
            for (unsigned i = 0; i < (group->numNeurons); i++) {
                group->spk[i] = 0;
            }
            // current source variables
        }
    }
     {
        // merged neuron init group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup1[g]; 
            group->spkCnt[0] = 0;
            for (unsigned i = 0; i < (group->numNeurons); i++) {
                group->spk[i] = 0;
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn0[i] = 0.000000;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn1[i] = 0.000000;
                }
            }
            // current source variables
        }
    }
     {
        // merged neuron init group 2
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup2[g]; 
            group->spkCnt[0] = 0;
            for (unsigned i = 0; i < (group->numNeurons); i++) {
                group->spk[i] = 0;
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn0[i] = 0.000000;
                }
            }
            // current source variables
        }
    }
    // ------------------------------------------------------------------------
    // Custom update groups
    // ------------------------------------------------------------------------
    // Custom dense WU update groups
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
}

void initializeSparse() {
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
     {
        // merged sparse synapse init group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedSynapseSparseInitGroup0[g]; 
            // Zero column lengths
            std::fill_n(group->colLength, group->numTrgNeurons, 0);
            // Loop through presynaptic neurons
            for (unsigned int i = 0; i < group->numSrcNeurons; i++)
             {
                // Loop through synapses in corresponding matrix row
                for(unsigned int j = 0; j < group->rowLength[i]; j++)
                 {
                    // Calculate index of this synapse in the row-major matrix
                    const unsigned int rowMajorIndex = (i * group->rowStride) + j;
                    // Using this, lookup postsynaptic target
                    const unsigned int postIndex = group->ind[rowMajorIndex];
                    // From this calculate index of this synapse in the column-major matrix
                    const unsigned int colMajorIndex = (postIndex * group->colStride) + group->colLength[postIndex];
                    // Increment column length corresponding to this postsynaptic neuron
                    group->colLength[postIndex]++;
                    // Add remapping entry
                    group->remap[colMajorIndex] = rowMajorIndex;
                }
            }
        }
    }
    // ------------------------------------------------------------------------
    // Custom sparse WU update groups
}

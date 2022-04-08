#include "definitionsInternal.h"

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
unsigned long long iT;
double t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double initTime = 0.0;
double initSparseTime = 0.0;
double neuronUpdateTime = 0.0;
double presynapticUpdateTime = 0.0;
double postsynapticUpdateTime = 0.0;
double synapseDynamicsTime = 0.0;
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntneurongroup;
unsigned int* d_glbSpkCntneurongroup;
unsigned int* glbSpkneurongroup;
unsigned int* d_glbSpkneurongroup;
int32_t* ineurongroup;
int32_t* d_ineurongroup;
double* Vneurongroup;
double* d_Vneurongroup;
double* g_PN_iKCneurongroup;
double* d_g_PN_iKCneurongroup;
double* hneurongroup;
double* d_hneurongroup;
double* mneurongroup;
double* d_mneurongroup;
double* nneurongroup;
double* d_nneurongroup;
double* lastspikeneurongroup;
double* d_lastspikeneurongroup;
char* not_refractoryneurongroup;
char* d_not_refractoryneurongroup;
unsigned int* glbSpkCntneurongroup_1;
unsigned int* d_glbSpkCntneurongroup_1;
unsigned int* glbSpkneurongroup_1;
unsigned int* d_glbSpkneurongroup_1;
int32_t* ineurongroup_1;
int32_t* d_ineurongroup_1;
double* Vneurongroup_1;
double* d_Vneurongroup_1;
double* g_eKC_eKCneurongroup_1;
double* d_g_eKC_eKCneurongroup_1;
double* g_iKC_eKCneurongroup_1;
double* d_g_iKC_eKCneurongroup_1;
double* hneurongroup_1;
double* d_hneurongroup_1;
double* mneurongroup_1;
double* d_mneurongroup_1;
double* nneurongroup_1;
double* d_nneurongroup_1;
double* lastspikeneurongroup_1;
double* d_lastspikeneurongroup_1;
char* not_refractoryneurongroup_1;
char* d_not_refractoryneurongroup_1;
unsigned int* glbSpkCntspikegeneratorgroup;
unsigned int* d_glbSpkCntspikegeneratorgroup;
unsigned int* glbSpkspikegeneratorgroup;
unsigned int* d_glbSpkspikegeneratorgroup;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
double* inSynsynapses;
double* d_inSynsynapses;
double* inSynsynapses_1;
double* d_inSynsynapses_1;
double* inSynsynapses_2;
double* d_inSynsynapses_2;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthsynapses = 421;
unsigned int* rowLengthsynapses;
unsigned int* d_rowLengthsynapses;
uint32_t* indsynapses;
uint32_t* d_indsynapses;
const unsigned int maxRowLengthsynapses_1 = 100;
unsigned int* rowLengthsynapses_1;
unsigned int* d_rowLengthsynapses_1;
uint32_t* indsynapses_1;
uint32_t* d_indsynapses_1;
unsigned int* d_colLengthsynapses_1;
unsigned int* d_remapsynapses_1;
const unsigned int maxRowLengthsynapses_2 = 100;
unsigned int* rowLengthsynapses_2;
unsigned int* d_rowLengthsynapses_2;
uint32_t* indsynapses_2;
uint32_t* d_indsynapses_2;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
double* weightsynapses;
double* d_weightsynapses;
double* lastupdatesynapses_1;
double* d_lastupdatesynapses_1;
double* Apostsynapses_1;
double* d_Apostsynapses_1;
double* g_rawsynapses_1;
double* d_g_rawsynapses_1;
double* Apresynapses_1;
double* d_Apresynapses_1;

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushneurongroupSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntneurongroup, glbSpkCntneurongroup, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkneurongroup, glbSpkneurongroup, 2500 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushneurongroupCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntneurongroup, glbSpkCntneurongroup, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkneurongroup, glbSpkneurongroup, glbSpkCntneurongroup[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushineurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ineurongroup, ineurongroup, 2500 * sizeof(int32_t), cudaMemcpyHostToDevice));
}

void pushCurrentineurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ineurongroup, ineurongroup, 2500 * sizeof(int32_t), cudaMemcpyHostToDevice));
}

void pushVneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vneurongroup, Vneurongroup, 2500 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrentVneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vneurongroup, Vneurongroup, 2500 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushg_PN_iKCneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_g_PN_iKCneurongroup, g_PN_iKCneurongroup, 2500 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrentg_PN_iKCneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_g_PN_iKCneurongroup, g_PN_iKCneurongroup, 2500 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushhneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_hneurongroup, hneurongroup, 2500 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrenthneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_hneurongroup, hneurongroup, 2500 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushmneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_mneurongroup, mneurongroup, 2500 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrentmneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_mneurongroup, mneurongroup, 2500 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushnneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_nneurongroup, nneurongroup, 2500 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrentnneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_nneurongroup, nneurongroup, 2500 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushlastspikeneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_lastspikeneurongroup, lastspikeneurongroup, 2500 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrentlastspikeneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_lastspikeneurongroup, lastspikeneurongroup, 2500 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushnot_refractoryneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_not_refractoryneurongroup, not_refractoryneurongroup, 2500 * sizeof(char), cudaMemcpyHostToDevice));
}

void pushCurrentnot_refractoryneurongroupToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_not_refractoryneurongroup, not_refractoryneurongroup, 2500 * sizeof(char), cudaMemcpyHostToDevice));
}

void pushneurongroupStateToDevice(bool uninitialisedOnly) {
    pushineurongroupToDevice(uninitialisedOnly);
    pushVneurongroupToDevice(uninitialisedOnly);
    pushg_PN_iKCneurongroupToDevice(uninitialisedOnly);
    pushhneurongroupToDevice(uninitialisedOnly);
    pushmneurongroupToDevice(uninitialisedOnly);
    pushnneurongroupToDevice(uninitialisedOnly);
    pushlastspikeneurongroupToDevice(uninitialisedOnly);
    pushnot_refractoryneurongroupToDevice(uninitialisedOnly);
}

void pushneurongroup_1SpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntneurongroup_1, glbSpkCntneurongroup_1, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkneurongroup_1, glbSpkneurongroup_1, 100 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushneurongroup_1CurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntneurongroup_1, glbSpkCntneurongroup_1, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkneurongroup_1, glbSpkneurongroup_1, glbSpkCntneurongroup_1[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushineurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ineurongroup_1, ineurongroup_1, 100 * sizeof(int32_t), cudaMemcpyHostToDevice));
}

void pushCurrentineurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ineurongroup_1, ineurongroup_1, 100 * sizeof(int32_t), cudaMemcpyHostToDevice));
}

void pushVneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vneurongroup_1, Vneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrentVneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vneurongroup_1, Vneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushg_eKC_eKCneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_g_eKC_eKCneurongroup_1, g_eKC_eKCneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrentg_eKC_eKCneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_g_eKC_eKCneurongroup_1, g_eKC_eKCneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushg_iKC_eKCneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_g_iKC_eKCneurongroup_1, g_iKC_eKCneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrentg_iKC_eKCneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_g_iKC_eKCneurongroup_1, g_iKC_eKCneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushhneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_hneurongroup_1, hneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrenthneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_hneurongroup_1, hneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushmneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_mneurongroup_1, mneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrentmneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_mneurongroup_1, mneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushnneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_nneurongroup_1, nneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrentnneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_nneurongroup_1, nneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushlastspikeneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_lastspikeneurongroup_1, lastspikeneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushCurrentlastspikeneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_lastspikeneurongroup_1, lastspikeneurongroup_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushnot_refractoryneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_not_refractoryneurongroup_1, not_refractoryneurongroup_1, 100 * sizeof(char), cudaMemcpyHostToDevice));
}

void pushCurrentnot_refractoryneurongroup_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_not_refractoryneurongroup_1, not_refractoryneurongroup_1, 100 * sizeof(char), cudaMemcpyHostToDevice));
}

void pushneurongroup_1StateToDevice(bool uninitialisedOnly) {
    pushineurongroup_1ToDevice(uninitialisedOnly);
    pushVneurongroup_1ToDevice(uninitialisedOnly);
    pushg_eKC_eKCneurongroup_1ToDevice(uninitialisedOnly);
    pushg_iKC_eKCneurongroup_1ToDevice(uninitialisedOnly);
    pushhneurongroup_1ToDevice(uninitialisedOnly);
    pushmneurongroup_1ToDevice(uninitialisedOnly);
    pushnneurongroup_1ToDevice(uninitialisedOnly);
    pushlastspikeneurongroup_1ToDevice(uninitialisedOnly);
    pushnot_refractoryneurongroup_1ToDevice(uninitialisedOnly);
}

void pushspikegeneratorgroupSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntspikegeneratorgroup, glbSpkCntspikegeneratorgroup, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkspikegeneratorgroup, glbSpkspikegeneratorgroup, 100 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushspikegeneratorgroupCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntspikegeneratorgroup, glbSpkCntspikegeneratorgroup, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkspikegeneratorgroup, glbSpkspikegeneratorgroup, glbSpkCntspikegeneratorgroup[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushspikegeneratorgroupStateToDevice(bool uninitialisedOnly) {
}

void pushsynapsesConnectivityToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthsynapses, rowLengthsynapses, 100 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_indsynapses, indsynapses, 42100 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushsynapses_1ConnectivityToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthsynapses_1, rowLengthsynapses_1, 2500 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_indsynapses_1, indsynapses_1, 250000 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushsynapses_2ConnectivityToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthsynapses_2, rowLengthsynapses_2, 100 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_indsynapses_2, indsynapses_2, 10000 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushweightsynapsesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_weightsynapses, weightsynapses, 42100 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushinSynsynapsesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynsynapses, inSynsynapses, 2500 * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void pushsynapsesStateToDevice(bool uninitialisedOnly) {
    pushweightsynapsesToDevice(uninitialisedOnly);
    pushinSynsynapsesToDevice(uninitialisedOnly);
}

void pushlastupdatesynapses_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_lastupdatesynapses_1, lastupdatesynapses_1, 250000 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushApostsynapses_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Apostsynapses_1, Apostsynapses_1, 250000 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushg_rawsynapses_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_g_rawsynapses_1, g_rawsynapses_1, 250000 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushApresynapses_1ToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Apresynapses_1, Apresynapses_1, 250000 * sizeof(double), cudaMemcpyHostToDevice));
}

void pushinSynsynapses_1ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynsynapses_1, inSynsynapses_1, 100 * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void pushsynapses_1StateToDevice(bool uninitialisedOnly) {
    pushlastupdatesynapses_1ToDevice(uninitialisedOnly);
    pushApostsynapses_1ToDevice(uninitialisedOnly);
    pushg_rawsynapses_1ToDevice(uninitialisedOnly);
    pushApresynapses_1ToDevice(uninitialisedOnly);
    pushinSynsynapses_1ToDevice(uninitialisedOnly);
}

void pushinSynsynapses_2ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynsynapses_2, inSynsynapses_2, 100 * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void pushsynapses_2StateToDevice(bool uninitialisedOnly) {
    pushinSynsynapses_2ToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullneurongroupSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntneurongroup, d_glbSpkCntneurongroup, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkneurongroup, d_glbSpkneurongroup, 2500 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullneurongroupCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntneurongroup, d_glbSpkCntneurongroup, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkneurongroup, d_glbSpkneurongroup, glbSpkCntneurongroup[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullineurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ineurongroup, d_ineurongroup, 2500 * sizeof(int32_t), cudaMemcpyDeviceToHost));
}

void pullCurrentineurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ineurongroup, d_ineurongroup, 2500 * sizeof(int32_t), cudaMemcpyDeviceToHost));
}

void pullVneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vneurongroup, d_Vneurongroup, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrentVneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vneurongroup, d_Vneurongroup, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullg_PN_iKCneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(g_PN_iKCneurongroup, d_g_PN_iKCneurongroup, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrentg_PN_iKCneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(g_PN_iKCneurongroup, d_g_PN_iKCneurongroup, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullhneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(hneurongroup, d_hneurongroup, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrenthneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(hneurongroup, d_hneurongroup, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullmneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mneurongroup, d_mneurongroup, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrentmneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mneurongroup, d_mneurongroup, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullnneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(nneurongroup, d_nneurongroup, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrentnneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(nneurongroup, d_nneurongroup, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pulllastspikeneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(lastspikeneurongroup, d_lastspikeneurongroup, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrentlastspikeneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(lastspikeneurongroup, d_lastspikeneurongroup, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullnot_refractoryneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(not_refractoryneurongroup, d_not_refractoryneurongroup, 2500 * sizeof(char), cudaMemcpyDeviceToHost));
}

void pullCurrentnot_refractoryneurongroupFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(not_refractoryneurongroup, d_not_refractoryneurongroup, 2500 * sizeof(char), cudaMemcpyDeviceToHost));
}

void pullneurongroupStateFromDevice() {
    pullineurongroupFromDevice();
    pullVneurongroupFromDevice();
    pullg_PN_iKCneurongroupFromDevice();
    pullhneurongroupFromDevice();
    pullmneurongroupFromDevice();
    pullnneurongroupFromDevice();
    pulllastspikeneurongroupFromDevice();
    pullnot_refractoryneurongroupFromDevice();
}

void pullneurongroup_1SpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntneurongroup_1, d_glbSpkCntneurongroup_1, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkneurongroup_1, d_glbSpkneurongroup_1, 100 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullneurongroup_1CurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntneurongroup_1, d_glbSpkCntneurongroup_1, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkneurongroup_1, d_glbSpkneurongroup_1, glbSpkCntneurongroup_1[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullineurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ineurongroup_1, d_ineurongroup_1, 100 * sizeof(int32_t), cudaMemcpyDeviceToHost));
}

void pullCurrentineurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ineurongroup_1, d_ineurongroup_1, 100 * sizeof(int32_t), cudaMemcpyDeviceToHost));
}

void pullVneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vneurongroup_1, d_Vneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrentVneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vneurongroup_1, d_Vneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullg_eKC_eKCneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(g_eKC_eKCneurongroup_1, d_g_eKC_eKCneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrentg_eKC_eKCneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(g_eKC_eKCneurongroup_1, d_g_eKC_eKCneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullg_iKC_eKCneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(g_iKC_eKCneurongroup_1, d_g_iKC_eKCneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrentg_iKC_eKCneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(g_iKC_eKCneurongroup_1, d_g_iKC_eKCneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullhneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(hneurongroup_1, d_hneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrenthneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(hneurongroup_1, d_hneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullmneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mneurongroup_1, d_mneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrentmneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mneurongroup_1, d_mneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullnneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(nneurongroup_1, d_nneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrentnneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(nneurongroup_1, d_nneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pulllastspikeneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(lastspikeneurongroup_1, d_lastspikeneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullCurrentlastspikeneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(lastspikeneurongroup_1, d_lastspikeneurongroup_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullnot_refractoryneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(not_refractoryneurongroup_1, d_not_refractoryneurongroup_1, 100 * sizeof(char), cudaMemcpyDeviceToHost));
}

void pullCurrentnot_refractoryneurongroup_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(not_refractoryneurongroup_1, d_not_refractoryneurongroup_1, 100 * sizeof(char), cudaMemcpyDeviceToHost));
}

void pullneurongroup_1StateFromDevice() {
    pullineurongroup_1FromDevice();
    pullVneurongroup_1FromDevice();
    pullg_eKC_eKCneurongroup_1FromDevice();
    pullg_iKC_eKCneurongroup_1FromDevice();
    pullhneurongroup_1FromDevice();
    pullmneurongroup_1FromDevice();
    pullnneurongroup_1FromDevice();
    pulllastspikeneurongroup_1FromDevice();
    pullnot_refractoryneurongroup_1FromDevice();
}

void pullspikegeneratorgroupSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntspikegeneratorgroup, d_glbSpkCntspikegeneratorgroup, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkspikegeneratorgroup, d_glbSpkspikegeneratorgroup, 100 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullspikegeneratorgroupCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntspikegeneratorgroup, d_glbSpkCntspikegeneratorgroup, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkspikegeneratorgroup, d_glbSpkspikegeneratorgroup, glbSpkCntspikegeneratorgroup[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullspikegeneratorgroupStateFromDevice() {
}

void pullsynapsesConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthsynapses, d_rowLengthsynapses, 100 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indsynapses, d_indsynapses, 42100 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullsynapses_1ConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthsynapses_1, d_rowLengthsynapses_1, 2500 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indsynapses_1, d_indsynapses_1, 250000 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullsynapses_2ConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthsynapses_2, d_rowLengthsynapses_2, 100 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indsynapses_2, d_indsynapses_2, 10000 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullweightsynapsesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(weightsynapses, d_weightsynapses, 42100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullinSynsynapsesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynsynapses, d_inSynsynapses, 2500 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullsynapsesStateFromDevice() {
    pullweightsynapsesFromDevice();
    pullinSynsynapsesFromDevice();
}

void pulllastupdatesynapses_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(lastupdatesynapses_1, d_lastupdatesynapses_1, 250000 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullApostsynapses_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Apostsynapses_1, d_Apostsynapses_1, 250000 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullg_rawsynapses_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(g_rawsynapses_1, d_g_rawsynapses_1, 250000 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullApresynapses_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Apresynapses_1, d_Apresynapses_1, 250000 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullinSynsynapses_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynsynapses_1, d_inSynsynapses_1, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullsynapses_1StateFromDevice() {
    pulllastupdatesynapses_1FromDevice();
    pullApostsynapses_1FromDevice();
    pullg_rawsynapses_1FromDevice();
    pullApresynapses_1FromDevice();
    pullinSynsynapses_1FromDevice();
}

void pullinSynsynapses_2FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynsynapses_2, d_inSynsynapses_2, 100 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullsynapses_2StateFromDevice() {
    pullinSynsynapses_2FromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getneurongroupCurrentSpikes(unsigned int batch) {
    return (glbSpkneurongroup);
}

unsigned int& getneurongroupCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntneurongroup[0];
}

int32_t* getCurrentineurongroup(unsigned int batch) {
    return ineurongroup;
}

double* getCurrentVneurongroup(unsigned int batch) {
    return Vneurongroup;
}

double* getCurrentg_PN_iKCneurongroup(unsigned int batch) {
    return g_PN_iKCneurongroup;
}

double* getCurrenthneurongroup(unsigned int batch) {
    return hneurongroup;
}

double* getCurrentmneurongroup(unsigned int batch) {
    return mneurongroup;
}

double* getCurrentnneurongroup(unsigned int batch) {
    return nneurongroup;
}

double* getCurrentlastspikeneurongroup(unsigned int batch) {
    return lastspikeneurongroup;
}

char* getCurrentnot_refractoryneurongroup(unsigned int batch) {
    return not_refractoryneurongroup;
}

unsigned int* getneurongroup_1CurrentSpikes(unsigned int batch) {
    return (glbSpkneurongroup_1);
}

unsigned int& getneurongroup_1CurrentSpikeCount(unsigned int batch) {
    return glbSpkCntneurongroup_1[0];
}

int32_t* getCurrentineurongroup_1(unsigned int batch) {
    return ineurongroup_1;
}

double* getCurrentVneurongroup_1(unsigned int batch) {
    return Vneurongroup_1;
}

double* getCurrentg_eKC_eKCneurongroup_1(unsigned int batch) {
    return g_eKC_eKCneurongroup_1;
}

double* getCurrentg_iKC_eKCneurongroup_1(unsigned int batch) {
    return g_iKC_eKCneurongroup_1;
}

double* getCurrenthneurongroup_1(unsigned int batch) {
    return hneurongroup_1;
}

double* getCurrentmneurongroup_1(unsigned int batch) {
    return mneurongroup_1;
}

double* getCurrentnneurongroup_1(unsigned int batch) {
    return nneurongroup_1;
}

double* getCurrentlastspikeneurongroup_1(unsigned int batch) {
    return lastspikeneurongroup_1;
}

char* getCurrentnot_refractoryneurongroup_1(unsigned int batch) {
    return not_refractoryneurongroup_1;
}

unsigned int* getspikegeneratorgroupCurrentSpikes(unsigned int batch) {
    return (glbSpkspikegeneratorgroup);
}

unsigned int& getspikegeneratorgroupCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntspikegeneratorgroup[0];
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushneurongroupStateToDevice(uninitialisedOnly);
    pushneurongroup_1StateToDevice(uninitialisedOnly);
    pushspikegeneratorgroupStateToDevice(uninitialisedOnly);
    pushsynapsesStateToDevice(uninitialisedOnly);
    pushsynapses_1StateToDevice(uninitialisedOnly);
    pushsynapses_2StateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushsynapsesConnectivityToDevice(uninitialisedOnly);
    pushsynapses_1ConnectivityToDevice(uninitialisedOnly);
    pushsynapses_2ConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pullneurongroupStateFromDevice();
    pullneurongroup_1StateFromDevice();
    pullspikegeneratorgroupStateFromDevice();
    pullsynapsesStateFromDevice();
    pullsynapses_1StateFromDevice();
    pullsynapses_2StateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullneurongroupCurrentSpikesFromDevice();
    pullneurongroup_1CurrentSpikesFromDevice();
    pullspikegeneratorgroupCurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
}

void allocateMem() {
    int deviceID;
    CHECK_CUDA_ERRORS(cudaDeviceGetByPCIBusId(&deviceID, "0000:3B:00.0"));
    CHECK_CUDA_ERRORS(cudaSetDevice(deviceID));
    
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntneurongroup, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntneurongroup, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkneurongroup, 2500 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkneurongroup, 2500 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ineurongroup, 2500 * sizeof(int32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ineurongroup, 2500 * sizeof(int32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vneurongroup, 2500 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vneurongroup, 2500 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&g_PN_iKCneurongroup, 2500 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_g_PN_iKCneurongroup, 2500 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&hneurongroup, 2500 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_hneurongroup, 2500 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&mneurongroup, 2500 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_mneurongroup, 2500 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&nneurongroup, 2500 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_nneurongroup, 2500 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&lastspikeneurongroup, 2500 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_lastspikeneurongroup, 2500 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&not_refractoryneurongroup, 2500 * sizeof(char), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_not_refractoryneurongroup, 2500 * sizeof(char)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntneurongroup_1, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntneurongroup_1, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkneurongroup_1, 100 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkneurongroup_1, 100 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ineurongroup_1, 100 * sizeof(int32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ineurongroup_1, 100 * sizeof(int32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vneurongroup_1, 100 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vneurongroup_1, 100 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&g_eKC_eKCneurongroup_1, 100 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_g_eKC_eKCneurongroup_1, 100 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&g_iKC_eKCneurongroup_1, 100 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_g_iKC_eKCneurongroup_1, 100 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&hneurongroup_1, 100 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_hneurongroup_1, 100 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&mneurongroup_1, 100 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_mneurongroup_1, 100 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&nneurongroup_1, 100 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_nneurongroup_1, 100 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&lastspikeneurongroup_1, 100 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_lastspikeneurongroup_1, 100 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&not_refractoryneurongroup_1, 100 * sizeof(char), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_not_refractoryneurongroup_1, 100 * sizeof(char)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntspikegeneratorgroup, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntspikegeneratorgroup, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkspikegeneratorgroup, 100 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkspikegeneratorgroup, 100 * sizeof(unsigned int)));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynsynapses, 2500 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynsynapses, 2500 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynsynapses_1, 100 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynsynapses_1, 100 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynsynapses_2, 100 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynsynapses_2, 100 * sizeof(double)));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthsynapses, 100 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthsynapses, 100 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indsynapses, 42100 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indsynapses, 42100 * sizeof(uint32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthsynapses_1, 2500 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthsynapses_1, 2500 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indsynapses_1, 250000 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indsynapses_1, 250000 * sizeof(uint32_t)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_colLengthsynapses_1, 100 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_remapsynapses_1, 250000 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthsynapses_2, 100 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthsynapses_2, 100 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indsynapses_2, 10000 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indsynapses_2, 10000 * sizeof(uint32_t)));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&weightsynapses, 42100 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_weightsynapses, 42100 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&lastupdatesynapses_1, 250000 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_lastupdatesynapses_1, 250000 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Apostsynapses_1, 250000 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Apostsynapses_1, 250000 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&g_rawsynapses_1, 250000 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_g_rawsynapses_1, 250000 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Apresynapses_1, 250000 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Apresynapses_1, 250000 * sizeof(double)));
    
    pushMergedNeuronInitGroup0ToDevice(0, d_glbSpkCntspikegeneratorgroup, d_glbSpkspikegeneratorgroup, 100);
    pushMergedNeuronInitGroup1ToDevice(0, d_glbSpkCntneurongroup_1, d_glbSpkneurongroup_1, d_inSynsynapses_1, d_inSynsynapses_2, 100);
    pushMergedNeuronInitGroup2ToDevice(0, d_glbSpkCntneurongroup, d_glbSpkneurongroup, d_inSynsynapses, 2500);
    pushMergedSynapseSparseInitGroup0ToDevice(0, d_rowLengthsynapses_1, d_indsynapses_1, d_colLengthsynapses_1, d_remapsynapses_1, 100, 2500, 2500, 100);
    pushMergedNeuronUpdateGroup0ToDevice(0, d_glbSpkCntspikegeneratorgroup, d_glbSpkspikegeneratorgroup, 100);
    pushMergedNeuronUpdateGroup1ToDevice(0, d_glbSpkCntneurongroup_1, d_glbSpkneurongroup_1, d_ineurongroup_1, d_Vneurongroup_1, d_g_eKC_eKCneurongroup_1, d_g_iKC_eKCneurongroup_1, d_hneurongroup_1, d_mneurongroup_1, d_nneurongroup_1, d_lastspikeneurongroup_1, d_not_refractoryneurongroup_1, d_inSynsynapses_2, d_inSynsynapses_1, 100);
    pushMergedNeuronUpdateGroup2ToDevice(0, d_glbSpkCntneurongroup, d_glbSpkneurongroup, d_ineurongroup, d_Vneurongroup, d_g_PN_iKCneurongroup, d_hneurongroup, d_mneurongroup, d_nneurongroup, d_lastspikeneurongroup, d_not_refractoryneurongroup, d_inSynsynapses, 2500);
    pushMergedPresynapticUpdateGroup0ToDevice(0, d_inSynsynapses_2, d_glbSpkCntneurongroup_1, d_glbSpkneurongroup_1, d_rowLengthsynapses_2, d_indsynapses_2, 100, 100, 100);
    pushMergedPresynapticUpdateGroup1ToDevice(0, d_inSynsynapses_1, d_glbSpkCntneurongroup, d_glbSpkneurongroup, d_rowLengthsynapses_1, d_indsynapses_1, d_lastupdatesynapses_1, d_Apostsynapses_1, d_g_rawsynapses_1, d_Apresynapses_1, 100, 2500, 100);
    pushMergedPresynapticUpdateGroup2ToDevice(0, d_inSynsynapses, d_glbSpkCntspikegeneratorgroup, d_glbSpkspikegeneratorgroup, d_rowLengthsynapses, d_indsynapses, d_weightsynapses, 421, 100, 2500);
    pushMergedPostsynapticUpdateGroup0ToDevice(0, d_glbSpkCntneurongroup_1, d_glbSpkneurongroup_1, d_rowLengthsynapses_1, d_indsynapses_1, d_colLengthsynapses_1, d_remapsynapses_1, d_lastupdatesynapses_1, d_Apostsynapses_1, d_g_rawsynapses_1, d_Apresynapses_1, 100, 2500, 2500, 100);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, d_glbSpkCntneurongroup);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(1, d_glbSpkCntneurongroup_1);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(2, d_glbSpkCntspikegeneratorgroup);
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntneurongroup));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntneurongroup));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkneurongroup));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkneurongroup));
    CHECK_CUDA_ERRORS(cudaFreeHost(ineurongroup));
    CHECK_CUDA_ERRORS(cudaFree(d_ineurongroup));
    CHECK_CUDA_ERRORS(cudaFreeHost(Vneurongroup));
    CHECK_CUDA_ERRORS(cudaFree(d_Vneurongroup));
    CHECK_CUDA_ERRORS(cudaFreeHost(g_PN_iKCneurongroup));
    CHECK_CUDA_ERRORS(cudaFree(d_g_PN_iKCneurongroup));
    CHECK_CUDA_ERRORS(cudaFreeHost(hneurongroup));
    CHECK_CUDA_ERRORS(cudaFree(d_hneurongroup));
    CHECK_CUDA_ERRORS(cudaFreeHost(mneurongroup));
    CHECK_CUDA_ERRORS(cudaFree(d_mneurongroup));
    CHECK_CUDA_ERRORS(cudaFreeHost(nneurongroup));
    CHECK_CUDA_ERRORS(cudaFree(d_nneurongroup));
    CHECK_CUDA_ERRORS(cudaFreeHost(lastspikeneurongroup));
    CHECK_CUDA_ERRORS(cudaFree(d_lastspikeneurongroup));
    CHECK_CUDA_ERRORS(cudaFreeHost(not_refractoryneurongroup));
    CHECK_CUDA_ERRORS(cudaFree(d_not_refractoryneurongroup));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(ineurongroup_1));
    CHECK_CUDA_ERRORS(cudaFree(d_ineurongroup_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(Vneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFree(d_Vneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(g_eKC_eKCneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFree(d_g_eKC_eKCneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(g_iKC_eKCneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFree(d_g_iKC_eKCneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(hneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFree(d_hneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(mneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFree(d_mneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(nneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFree(d_nneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(lastspikeneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFree(d_lastspikeneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(not_refractoryneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFree(d_not_refractoryneurongroup_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntspikegeneratorgroup));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntspikegeneratorgroup));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkspikegeneratorgroup));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkspikegeneratorgroup));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynsynapses));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynsynapses));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynsynapses_1));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynsynapses_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynsynapses_2));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynsynapses_2));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthsynapses));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthsynapses));
    CHECK_CUDA_ERRORS(cudaFreeHost(indsynapses));
    CHECK_CUDA_ERRORS(cudaFree(d_indsynapses));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthsynapses_1));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthsynapses_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(indsynapses_1));
    CHECK_CUDA_ERRORS(cudaFree(d_indsynapses_1));
    CHECK_CUDA_ERRORS(cudaFree(d_colLengthsynapses_1));
    CHECK_CUDA_ERRORS(cudaFree(d_remapsynapses_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthsynapses_2));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthsynapses_2));
    CHECK_CUDA_ERRORS(cudaFreeHost(indsynapses_2));
    CHECK_CUDA_ERRORS(cudaFree(d_indsynapses_2));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(weightsynapses));
    CHECK_CUDA_ERRORS(cudaFree(d_weightsynapses));
    CHECK_CUDA_ERRORS(cudaFreeHost(lastupdatesynapses_1));
    CHECK_CUDA_ERRORS(cudaFree(d_lastupdatesynapses_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(Apostsynapses_1));
    CHECK_CUDA_ERRORS(cudaFree(d_Apostsynapses_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(g_rawsynapses_1));
    CHECK_CUDA_ERRORS(cudaFree(d_g_rawsynapses_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(Apresynapses_1));
    CHECK_CUDA_ERRORS(cudaFree(d_Apresynapses_1));
    
}

size_t getFreeDeviceMemBytes() {
    size_t free;
    size_t total;
    CHECK_CUDA_ERRORS(cudaMemGetInfo(&free, &total));
    return free;
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t); 
    iT++;
    t = iT*DT;
}


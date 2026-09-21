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
unsigned int* glbSpkneurongroup;
int32_t* ineurongroup;
double* Vneurongroup;
double* g_PN_iKCneurongroup;
double* hneurongroup;
double* mneurongroup;
double* nneurongroup;
double* lastspikeneurongroup;
char* not_refractoryneurongroup;
unsigned int* glbSpkCntneurongroup_1;
unsigned int* glbSpkneurongroup_1;
int32_t* ineurongroup_1;
double* Vneurongroup_1;
double* g_eKC_eKCneurongroup_1;
double* g_iKC_eKCneurongroup_1;
double* hneurongroup_1;
double* mneurongroup_1;
double* nneurongroup_1;
double* lastspikeneurongroup_1;
char* not_refractoryneurongroup_1;
unsigned int* glbSpkCntspikegeneratorgroup;
unsigned int* glbSpkspikegeneratorgroup;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
double* inSynsynapses;
double* inSynsynapses_1;
double* inSynsynapses_2;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthsynapses = 424;
unsigned int* rowLengthsynapses;
uint32_t* indsynapses;
const unsigned int maxRowLengthsynapses_1 = 100;
unsigned int* rowLengthsynapses_1;
uint32_t* indsynapses_1;
unsigned int* colLengthsynapses_1;
unsigned int* remapsynapses_1;
const unsigned int maxRowLengthsynapses_2 = 100;
unsigned int* rowLengthsynapses_2;
uint32_t* indsynapses_2;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
double* weightsynapses;
double* g_rawsynapses_1;
double* Apostsynapses_1;
double* Apresynapses_1;
double* lastupdatesynapses_1;

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushneurongroupSpikesToDevice(bool uninitialisedOnly) {
}

void pushneurongroupCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushineurongroupToDevice(bool uninitialisedOnly) {
}

void pushCurrentineurongroupToDevice(bool uninitialisedOnly) {
}

void pushVneurongroupToDevice(bool uninitialisedOnly) {
}

void pushCurrentVneurongroupToDevice(bool uninitialisedOnly) {
}

void pushg_PN_iKCneurongroupToDevice(bool uninitialisedOnly) {
}

void pushCurrentg_PN_iKCneurongroupToDevice(bool uninitialisedOnly) {
}

void pushhneurongroupToDevice(bool uninitialisedOnly) {
}

void pushCurrenthneurongroupToDevice(bool uninitialisedOnly) {
}

void pushmneurongroupToDevice(bool uninitialisedOnly) {
}

void pushCurrentmneurongroupToDevice(bool uninitialisedOnly) {
}

void pushnneurongroupToDevice(bool uninitialisedOnly) {
}

void pushCurrentnneurongroupToDevice(bool uninitialisedOnly) {
}

void pushlastspikeneurongroupToDevice(bool uninitialisedOnly) {
}

void pushCurrentlastspikeneurongroupToDevice(bool uninitialisedOnly) {
}

void pushnot_refractoryneurongroupToDevice(bool uninitialisedOnly) {
}

void pushCurrentnot_refractoryneurongroupToDevice(bool uninitialisedOnly) {
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
}

void pushneurongroup_1CurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushineurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushCurrentineurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushVneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushCurrentVneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushg_eKC_eKCneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushCurrentg_eKC_eKCneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushg_iKC_eKCneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushCurrentg_iKC_eKCneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushhneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushCurrenthneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushmneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushCurrentmneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushnneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushCurrentnneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushlastspikeneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushCurrentlastspikeneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushnot_refractoryneurongroup_1ToDevice(bool uninitialisedOnly) {
}

void pushCurrentnot_refractoryneurongroup_1ToDevice(bool uninitialisedOnly) {
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
}

void pushspikegeneratorgroupCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushspikegeneratorgroupStateToDevice(bool uninitialisedOnly) {
}

void pushsynapsesConnectivityToDevice(bool uninitialisedOnly) {
}

void pushsynapses_1ConnectivityToDevice(bool uninitialisedOnly) {
}

void pushsynapses_2ConnectivityToDevice(bool uninitialisedOnly) {
}

void pushweightsynapsesToDevice(bool uninitialisedOnly) {
}

void pushinSynsynapsesToDevice(bool uninitialisedOnly) {
}

void pushsynapsesStateToDevice(bool uninitialisedOnly) {
    pushweightsynapsesToDevice(uninitialisedOnly);
    pushinSynsynapsesToDevice(uninitialisedOnly);
}

void pushg_rawsynapses_1ToDevice(bool uninitialisedOnly) {
}

void pushApostsynapses_1ToDevice(bool uninitialisedOnly) {
}

void pushApresynapses_1ToDevice(bool uninitialisedOnly) {
}

void pushlastupdatesynapses_1ToDevice(bool uninitialisedOnly) {
}

void pushinSynsynapses_1ToDevice(bool uninitialisedOnly) {
}

void pushsynapses_1StateToDevice(bool uninitialisedOnly) {
    pushg_rawsynapses_1ToDevice(uninitialisedOnly);
    pushApostsynapses_1ToDevice(uninitialisedOnly);
    pushApresynapses_1ToDevice(uninitialisedOnly);
    pushlastupdatesynapses_1ToDevice(uninitialisedOnly);
    pushinSynsynapses_1ToDevice(uninitialisedOnly);
}

void pushinSynsynapses_2ToDevice(bool uninitialisedOnly) {
}

void pushsynapses_2StateToDevice(bool uninitialisedOnly) {
    pushinSynsynapses_2ToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullneurongroupSpikesFromDevice() {
}

void pullneurongroupCurrentSpikesFromDevice() {
}

void pullineurongroupFromDevice() {
}

void pullCurrentineurongroupFromDevice() {
}

void pullVneurongroupFromDevice() {
}

void pullCurrentVneurongroupFromDevice() {
}

void pullg_PN_iKCneurongroupFromDevice() {
}

void pullCurrentg_PN_iKCneurongroupFromDevice() {
}

void pullhneurongroupFromDevice() {
}

void pullCurrenthneurongroupFromDevice() {
}

void pullmneurongroupFromDevice() {
}

void pullCurrentmneurongroupFromDevice() {
}

void pullnneurongroupFromDevice() {
}

void pullCurrentnneurongroupFromDevice() {
}

void pulllastspikeneurongroupFromDevice() {
}

void pullCurrentlastspikeneurongroupFromDevice() {
}

void pullnot_refractoryneurongroupFromDevice() {
}

void pullCurrentnot_refractoryneurongroupFromDevice() {
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
}

void pullneurongroup_1CurrentSpikesFromDevice() {
}

void pullineurongroup_1FromDevice() {
}

void pullCurrentineurongroup_1FromDevice() {
}

void pullVneurongroup_1FromDevice() {
}

void pullCurrentVneurongroup_1FromDevice() {
}

void pullg_eKC_eKCneurongroup_1FromDevice() {
}

void pullCurrentg_eKC_eKCneurongroup_1FromDevice() {
}

void pullg_iKC_eKCneurongroup_1FromDevice() {
}

void pullCurrentg_iKC_eKCneurongroup_1FromDevice() {
}

void pullhneurongroup_1FromDevice() {
}

void pullCurrenthneurongroup_1FromDevice() {
}

void pullmneurongroup_1FromDevice() {
}

void pullCurrentmneurongroup_1FromDevice() {
}

void pullnneurongroup_1FromDevice() {
}

void pullCurrentnneurongroup_1FromDevice() {
}

void pulllastspikeneurongroup_1FromDevice() {
}

void pullCurrentlastspikeneurongroup_1FromDevice() {
}

void pullnot_refractoryneurongroup_1FromDevice() {
}

void pullCurrentnot_refractoryneurongroup_1FromDevice() {
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
}

void pullspikegeneratorgroupCurrentSpikesFromDevice() {
}

void pullspikegeneratorgroupStateFromDevice() {
}

void pullsynapsesConnectivityFromDevice() {
}

void pullsynapses_1ConnectivityFromDevice() {
}

void pullsynapses_2ConnectivityFromDevice() {
}

void pullweightsynapsesFromDevice() {
}

void pullinSynsynapsesFromDevice() {
}

void pullsynapsesStateFromDevice() {
    pullweightsynapsesFromDevice();
    pullinSynsynapsesFromDevice();
}

void pullg_rawsynapses_1FromDevice() {
}

void pullApostsynapses_1FromDevice() {
}

void pullApresynapses_1FromDevice() {
}

void pulllastupdatesynapses_1FromDevice() {
}

void pullinSynsynapses_1FromDevice() {
}

void pullsynapses_1StateFromDevice() {
    pullg_rawsynapses_1FromDevice();
    pullApostsynapses_1FromDevice();
    pullApresynapses_1FromDevice();
    pulllastupdatesynapses_1FromDevice();
    pullinSynsynapses_1FromDevice();
}

void pullinSynsynapses_2FromDevice() {
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
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    glbSpkCntneurongroup = new unsigned int[1];
    glbSpkneurongroup = new unsigned int[2500];
    ineurongroup = new int32_t[2500];
    Vneurongroup = new double[2500];
    g_PN_iKCneurongroup = new double[2500];
    hneurongroup = new double[2500];
    mneurongroup = new double[2500];
    nneurongroup = new double[2500];
    lastspikeneurongroup = new double[2500];
    not_refractoryneurongroup = new char[2500];
    glbSpkCntneurongroup_1 = new unsigned int[1];
    glbSpkneurongroup_1 = new unsigned int[100];
    ineurongroup_1 = new int32_t[100];
    Vneurongroup_1 = new double[100];
    g_eKC_eKCneurongroup_1 = new double[100];
    g_iKC_eKCneurongroup_1 = new double[100];
    hneurongroup_1 = new double[100];
    mneurongroup_1 = new double[100];
    nneurongroup_1 = new double[100];
    lastspikeneurongroup_1 = new double[100];
    not_refractoryneurongroup_1 = new char[100];
    glbSpkCntspikegeneratorgroup = new unsigned int[1];
    glbSpkspikegeneratorgroup = new unsigned int[100];
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    inSynsynapses = new double[2500];
    inSynsynapses_1 = new double[100];
    inSynsynapses_2 = new double[100];
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    rowLengthsynapses = new unsigned int[100];
    indsynapses = new uint32_t[42400];
    rowLengthsynapses_1 = new unsigned int[2500];
    indsynapses_1 = new uint32_t[250000];
    colLengthsynapses_1 = new unsigned int[100];
    remapsynapses_1 = new unsigned int[250000];
    rowLengthsynapses_2 = new unsigned int[100];
    indsynapses_2 = new uint32_t[10000];
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    weightsynapses = new double[42400];
    g_rawsynapses_1 = new double[250000];
    Apostsynapses_1 = new double[250000];
    Apresynapses_1 = new double[250000];
    lastupdatesynapses_1 = new double[250000];
    
    pushMergedNeuronInitGroup0ToDevice(0, glbSpkCntspikegeneratorgroup, glbSpkspikegeneratorgroup, 100);
    pushMergedNeuronInitGroup1ToDevice(0, glbSpkCntneurongroup_1, glbSpkneurongroup_1, inSynsynapses_1, inSynsynapses_2, 100);
    pushMergedNeuronInitGroup2ToDevice(0, glbSpkCntneurongroup, glbSpkneurongroup, inSynsynapses, 2500);
    pushMergedSynapseSparseInitGroup0ToDevice(0, rowLengthsynapses_1, indsynapses_1, colLengthsynapses_1, remapsynapses_1, 100, 2500, 2500, 100);
    pushMergedNeuronUpdateGroup0ToDevice(0, glbSpkCntspikegeneratorgroup, glbSpkspikegeneratorgroup, 100);
    pushMergedNeuronUpdateGroup1ToDevice(0, glbSpkCntneurongroup_1, glbSpkneurongroup_1, ineurongroup_1, Vneurongroup_1, g_eKC_eKCneurongroup_1, g_iKC_eKCneurongroup_1, hneurongroup_1, mneurongroup_1, nneurongroup_1, lastspikeneurongroup_1, not_refractoryneurongroup_1, inSynsynapses_2, inSynsynapses_1, 100);
    pushMergedNeuronUpdateGroup2ToDevice(0, glbSpkCntneurongroup, glbSpkneurongroup, ineurongroup, Vneurongroup, g_PN_iKCneurongroup, hneurongroup, mneurongroup, nneurongroup, lastspikeneurongroup, not_refractoryneurongroup, inSynsynapses, 2500);
    pushMergedPresynapticUpdateGroup0ToDevice(0, inSynsynapses_2, glbSpkCntneurongroup_1, glbSpkneurongroup_1, rowLengthsynapses_2, indsynapses_2, 100, 100, 100);
    pushMergedPresynapticUpdateGroup1ToDevice(0, inSynsynapses_1, glbSpkCntneurongroup, glbSpkneurongroup, rowLengthsynapses_1, indsynapses_1, g_rawsynapses_1, Apostsynapses_1, Apresynapses_1, lastupdatesynapses_1, 100, 2500, 100);
    pushMergedPresynapticUpdateGroup2ToDevice(0, inSynsynapses, glbSpkCntspikegeneratorgroup, glbSpkspikegeneratorgroup, rowLengthsynapses, indsynapses, weightsynapses, 424, 100, 2500);
    pushMergedPostsynapticUpdateGroup0ToDevice(0, glbSpkCntneurongroup_1, glbSpkneurongroup_1, rowLengthsynapses_1, indsynapses_1, colLengthsynapses_1, remapsynapses_1, g_rawsynapses_1, Apostsynapses_1, Apresynapses_1, lastupdatesynapses_1, 100, 2500, 2500, 100);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, glbSpkCntneurongroup);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(1, glbSpkCntneurongroup_1);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(2, glbSpkCntspikegeneratorgroup);
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
    delete[] glbSpkCntneurongroup;
    delete[] glbSpkneurongroup;
    delete[] ineurongroup;
    delete[] Vneurongroup;
    delete[] g_PN_iKCneurongroup;
    delete[] hneurongroup;
    delete[] mneurongroup;
    delete[] nneurongroup;
    delete[] lastspikeneurongroup;
    delete[] not_refractoryneurongroup;
    delete[] glbSpkCntneurongroup_1;
    delete[] glbSpkneurongroup_1;
    delete[] ineurongroup_1;
    delete[] Vneurongroup_1;
    delete[] g_eKC_eKCneurongroup_1;
    delete[] g_iKC_eKCneurongroup_1;
    delete[] hneurongroup_1;
    delete[] mneurongroup_1;
    delete[] nneurongroup_1;
    delete[] lastspikeneurongroup_1;
    delete[] not_refractoryneurongroup_1;
    delete[] glbSpkCntspikegeneratorgroup;
    delete[] glbSpkspikegeneratorgroup;
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // postsynaptic variables
    // ------------------------------------------------------------------------
    delete[] inSynsynapses;
    delete[] inSynsynapses_1;
    delete[] inSynsynapses_2;
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    delete[] rowLengthsynapses;
    delete[] indsynapses;
    delete[] rowLengthsynapses_1;
    delete[] indsynapses_1;
    delete[] colLengthsynapses_1;
    delete[] remapsynapses_1;
    delete[] rowLengthsynapses_2;
    delete[] indsynapses_2;
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    delete[] weightsynapses;
    delete[] g_rawsynapses_1;
    delete[] Apostsynapses_1;
    delete[] Apresynapses_1;
    delete[] lastupdatesynapses_1;
    
}

size_t getFreeDeviceMemBytes() {
    return 0;
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t); 
    iT++;
    t = iT*DT;
}


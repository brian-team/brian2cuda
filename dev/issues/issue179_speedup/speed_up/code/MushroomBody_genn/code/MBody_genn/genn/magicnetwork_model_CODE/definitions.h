#pragma once
#define EXPORT_VAR extern
#define EXPORT_FUNC
// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

// Standard C includes
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#define DT 1.00000000000000005e-04
typedef double scalar;
#define SCALAR_MIN 2.22507385850720138e-308
#define SCALAR_MAX 1.79769313486231571e+308

#define TIME_MIN 2.22507385850720138e-308
#define TIME_MAX 1.79769313486231571e+308

// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR unsigned long long iT;
EXPORT_VAR double t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
EXPORT_VAR double initTime;
EXPORT_VAR double initSparseTime;
EXPORT_VAR double neuronUpdateTime;
EXPORT_VAR double presynapticUpdateTime;
EXPORT_VAR double postsynapticUpdateTime;
EXPORT_VAR double synapseDynamicsTime;
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
#define spikeCount_neurongroup glbSpkCntneurongroup[0]
#define spike_neurongroup glbSpkneurongroup
#define glbSpkShiftneurongroup 0

EXPORT_VAR unsigned int* glbSpkCntneurongroup;
EXPORT_VAR unsigned int* glbSpkneurongroup;
EXPORT_VAR int32_t* ineurongroup;
EXPORT_VAR double* Vneurongroup;
EXPORT_VAR double* g_PN_iKCneurongroup;
EXPORT_VAR double* hneurongroup;
EXPORT_VAR double* mneurongroup;
EXPORT_VAR double* nneurongroup;
EXPORT_VAR double* lastspikeneurongroup;
EXPORT_VAR char* not_refractoryneurongroup;
#define spikeCount_neurongroup_1 glbSpkCntneurongroup_1[0]
#define spike_neurongroup_1 glbSpkneurongroup_1
#define glbSpkShiftneurongroup_1 0

EXPORT_VAR unsigned int* glbSpkCntneurongroup_1;
EXPORT_VAR unsigned int* glbSpkneurongroup_1;
EXPORT_VAR int32_t* ineurongroup_1;
EXPORT_VAR double* Vneurongroup_1;
EXPORT_VAR double* g_eKC_eKCneurongroup_1;
EXPORT_VAR double* g_iKC_eKCneurongroup_1;
EXPORT_VAR double* hneurongroup_1;
EXPORT_VAR double* mneurongroup_1;
EXPORT_VAR double* nneurongroup_1;
EXPORT_VAR double* lastspikeneurongroup_1;
EXPORT_VAR char* not_refractoryneurongroup_1;
#define spikeCount_spikegeneratorgroup glbSpkCntspikegeneratorgroup[0]
#define spike_spikegeneratorgroup glbSpkspikegeneratorgroup
#define glbSpkShiftspikegeneratorgroup 0

EXPORT_VAR unsigned int* glbSpkCntspikegeneratorgroup;
EXPORT_VAR unsigned int* glbSpkspikegeneratorgroup;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR double* inSynsynapses;
EXPORT_VAR double* inSynsynapses_1;
EXPORT_VAR double* inSynsynapses_2;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
EXPORT_VAR const unsigned int maxRowLengthsynapses;
EXPORT_VAR unsigned int* rowLengthsynapses;
EXPORT_VAR uint32_t* indsynapses;
EXPORT_VAR const unsigned int maxRowLengthsynapses_1;
EXPORT_VAR unsigned int* rowLengthsynapses_1;
EXPORT_VAR uint32_t* indsynapses_1;
EXPORT_VAR unsigned int* colLengthsynapses_1;
EXPORT_VAR unsigned int* remapsynapses_1;
EXPORT_VAR const unsigned int maxRowLengthsynapses_2;
EXPORT_VAR unsigned int* rowLengthsynapses_2;
EXPORT_VAR uint32_t* indsynapses_2;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR double* weightsynapses;
EXPORT_VAR double* g_rawsynapses_1;
EXPORT_VAR double* Apostsynapses_1;
EXPORT_VAR double* Apresynapses_1;
EXPORT_VAR double* lastupdatesynapses_1;

EXPORT_FUNC void pushneurongroupSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullneurongroupSpikesFromDevice();
EXPORT_FUNC void pushneurongroupCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullneurongroupCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getneurongroupCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getneurongroupCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushineurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullineurongroupFromDevice();
EXPORT_FUNC void pushCurrentineurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentineurongroupFromDevice();
EXPORT_FUNC int32_t* getCurrentineurongroup(unsigned int batch = 0); 
EXPORT_FUNC void pushVneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVneurongroupFromDevice();
EXPORT_FUNC void pushCurrentVneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVneurongroupFromDevice();
EXPORT_FUNC double* getCurrentVneurongroup(unsigned int batch = 0); 
EXPORT_FUNC void pushg_PN_iKCneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullg_PN_iKCneurongroupFromDevice();
EXPORT_FUNC void pushCurrentg_PN_iKCneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentg_PN_iKCneurongroupFromDevice();
EXPORT_FUNC double* getCurrentg_PN_iKCneurongroup(unsigned int batch = 0); 
EXPORT_FUNC void pushhneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullhneurongroupFromDevice();
EXPORT_FUNC void pushCurrenthneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrenthneurongroupFromDevice();
EXPORT_FUNC double* getCurrenthneurongroup(unsigned int batch = 0); 
EXPORT_FUNC void pushmneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmneurongroupFromDevice();
EXPORT_FUNC void pushCurrentmneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentmneurongroupFromDevice();
EXPORT_FUNC double* getCurrentmneurongroup(unsigned int batch = 0); 
EXPORT_FUNC void pushnneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullnneurongroupFromDevice();
EXPORT_FUNC void pushCurrentnneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentnneurongroupFromDevice();
EXPORT_FUNC double* getCurrentnneurongroup(unsigned int batch = 0); 
EXPORT_FUNC void pushlastspikeneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulllastspikeneurongroupFromDevice();
EXPORT_FUNC void pushCurrentlastspikeneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentlastspikeneurongroupFromDevice();
EXPORT_FUNC double* getCurrentlastspikeneurongroup(unsigned int batch = 0); 
EXPORT_FUNC void pushnot_refractoryneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullnot_refractoryneurongroupFromDevice();
EXPORT_FUNC void pushCurrentnot_refractoryneurongroupToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentnot_refractoryneurongroupFromDevice();
EXPORT_FUNC char* getCurrentnot_refractoryneurongroup(unsigned int batch = 0); 
EXPORT_FUNC void pushneurongroupStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullneurongroupStateFromDevice();
EXPORT_FUNC void pushneurongroup_1SpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullneurongroup_1SpikesFromDevice();
EXPORT_FUNC void pushneurongroup_1CurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullneurongroup_1CurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getneurongroup_1CurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getneurongroup_1CurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushineurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullineurongroup_1FromDevice();
EXPORT_FUNC void pushCurrentineurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentineurongroup_1FromDevice();
EXPORT_FUNC int32_t* getCurrentineurongroup_1(unsigned int batch = 0); 
EXPORT_FUNC void pushVneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVneurongroup_1FromDevice();
EXPORT_FUNC void pushCurrentVneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVneurongroup_1FromDevice();
EXPORT_FUNC double* getCurrentVneurongroup_1(unsigned int batch = 0); 
EXPORT_FUNC void pushg_eKC_eKCneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullg_eKC_eKCneurongroup_1FromDevice();
EXPORT_FUNC void pushCurrentg_eKC_eKCneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentg_eKC_eKCneurongroup_1FromDevice();
EXPORT_FUNC double* getCurrentg_eKC_eKCneurongroup_1(unsigned int batch = 0); 
EXPORT_FUNC void pushg_iKC_eKCneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullg_iKC_eKCneurongroup_1FromDevice();
EXPORT_FUNC void pushCurrentg_iKC_eKCneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentg_iKC_eKCneurongroup_1FromDevice();
EXPORT_FUNC double* getCurrentg_iKC_eKCneurongroup_1(unsigned int batch = 0); 
EXPORT_FUNC void pushhneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullhneurongroup_1FromDevice();
EXPORT_FUNC void pushCurrenthneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrenthneurongroup_1FromDevice();
EXPORT_FUNC double* getCurrenthneurongroup_1(unsigned int batch = 0); 
EXPORT_FUNC void pushmneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmneurongroup_1FromDevice();
EXPORT_FUNC void pushCurrentmneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentmneurongroup_1FromDevice();
EXPORT_FUNC double* getCurrentmneurongroup_1(unsigned int batch = 0); 
EXPORT_FUNC void pushnneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullnneurongroup_1FromDevice();
EXPORT_FUNC void pushCurrentnneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentnneurongroup_1FromDevice();
EXPORT_FUNC double* getCurrentnneurongroup_1(unsigned int batch = 0); 
EXPORT_FUNC void pushlastspikeneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulllastspikeneurongroup_1FromDevice();
EXPORT_FUNC void pushCurrentlastspikeneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentlastspikeneurongroup_1FromDevice();
EXPORT_FUNC double* getCurrentlastspikeneurongroup_1(unsigned int batch = 0); 
EXPORT_FUNC void pushnot_refractoryneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullnot_refractoryneurongroup_1FromDevice();
EXPORT_FUNC void pushCurrentnot_refractoryneurongroup_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentnot_refractoryneurongroup_1FromDevice();
EXPORT_FUNC char* getCurrentnot_refractoryneurongroup_1(unsigned int batch = 0); 
EXPORT_FUNC void pushneurongroup_1StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullneurongroup_1StateFromDevice();
EXPORT_FUNC void pushspikegeneratorgroupSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullspikegeneratorgroupSpikesFromDevice();
EXPORT_FUNC void pushspikegeneratorgroupCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullspikegeneratorgroupCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getspikegeneratorgroupCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getspikegeneratorgroupCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushspikegeneratorgroupStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullspikegeneratorgroupStateFromDevice();
EXPORT_FUNC void pushsynapsesConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsynapsesConnectivityFromDevice();
EXPORT_FUNC void pushsynapses_1ConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsynapses_1ConnectivityFromDevice();
EXPORT_FUNC void pushsynapses_2ConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsynapses_2ConnectivityFromDevice();
EXPORT_FUNC void pushweightsynapsesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullweightsynapsesFromDevice();
EXPORT_FUNC void pushinSynsynapsesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsynapsesFromDevice();
EXPORT_FUNC void pushsynapsesStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsynapsesStateFromDevice();
EXPORT_FUNC void pushg_rawsynapses_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullg_rawsynapses_1FromDevice();
EXPORT_FUNC void pushApostsynapses_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullApostsynapses_1FromDevice();
EXPORT_FUNC void pushApresynapses_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullApresynapses_1FromDevice();
EXPORT_FUNC void pushlastupdatesynapses_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulllastupdatesynapses_1FromDevice();
EXPORT_FUNC void pushinSynsynapses_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsynapses_1FromDevice();
EXPORT_FUNC void pushsynapses_1StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsynapses_1StateFromDevice();
EXPORT_FUNC void pushinSynsynapses_2ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynsynapses_2FromDevice();
EXPORT_FUNC void pushsynapses_2StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsynapses_2StateFromDevice();
// Runner functions
EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyStateFromDevice();
EXPORT_FUNC void copyCurrentSpikesFromDevice();
EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();
EXPORT_FUNC void allocateMem();
EXPORT_FUNC void freeMem();
EXPORT_FUNC size_t getFreeDeviceMemBytes();
EXPORT_FUNC void stepTime();

// Functions generated by backend
EXPORT_FUNC void updateNeurons(double t); 
EXPORT_FUNC void updateSynapses(double t);
EXPORT_FUNC void initialize();
EXPORT_FUNC void initializeSparse();
}  // extern "C"

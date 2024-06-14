#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedNeuronUpdateGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    int32_t* i;
    double* V;
    double* g_eKC_eKC;
    double* g_iKC_eKC;
    double* h;
    double* m;
    double* n;
    double* lastspike;
    char* not_refractory;
    double* inSynInSyn0;
    double* inSynInSyn1;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup2
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    int32_t* i;
    double* V;
    double* g_PN_iKC;
    double* h;
    double* m;
    double* n;
    double* lastspike;
    char* not_refractory;
    double* inSynInSyn0;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup0
 {
    unsigned int* spkCnt;
    
}
;
static MergedNeuronUpdateGroup0 mergedNeuronUpdateGroup0[1];
void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int numNeurons) {
    mergedNeuronUpdateGroup0[idx].spkCnt = spkCnt;
    mergedNeuronUpdateGroup0[idx].spk = spk;
    mergedNeuronUpdateGroup0[idx].numNeurons = numNeurons;
}
static MergedNeuronUpdateGroup1 mergedNeuronUpdateGroup1[1];
void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, int32_t* i, double* V, double* g_eKC_eKC, double* g_iKC_eKC, double* h, double* m, double* n, double* lastspike, char* not_refractory, double* inSynInSyn0, double* inSynInSyn1, unsigned int numNeurons) {
    mergedNeuronUpdateGroup1[idx].spkCnt = spkCnt;
    mergedNeuronUpdateGroup1[idx].spk = spk;
    mergedNeuronUpdateGroup1[idx].i = i;
    mergedNeuronUpdateGroup1[idx].V = V;
    mergedNeuronUpdateGroup1[idx].g_eKC_eKC = g_eKC_eKC;
    mergedNeuronUpdateGroup1[idx].g_iKC_eKC = g_iKC_eKC;
    mergedNeuronUpdateGroup1[idx].h = h;
    mergedNeuronUpdateGroup1[idx].m = m;
    mergedNeuronUpdateGroup1[idx].n = n;
    mergedNeuronUpdateGroup1[idx].lastspike = lastspike;
    mergedNeuronUpdateGroup1[idx].not_refractory = not_refractory;
    mergedNeuronUpdateGroup1[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronUpdateGroup1[idx].inSynInSyn1 = inSynInSyn1;
    mergedNeuronUpdateGroup1[idx].numNeurons = numNeurons;
}
static MergedNeuronUpdateGroup2 mergedNeuronUpdateGroup2[1];
void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, int32_t* i, double* V, double* g_PN_iKC, double* h, double* m, double* n, double* lastspike, char* not_refractory, double* inSynInSyn0, unsigned int numNeurons) {
    mergedNeuronUpdateGroup2[idx].spkCnt = spkCnt;
    mergedNeuronUpdateGroup2[idx].spk = spk;
    mergedNeuronUpdateGroup2[idx].i = i;
    mergedNeuronUpdateGroup2[idx].V = V;
    mergedNeuronUpdateGroup2[idx].g_PN_iKC = g_PN_iKC;
    mergedNeuronUpdateGroup2[idx].h = h;
    mergedNeuronUpdateGroup2[idx].m = m;
    mergedNeuronUpdateGroup2[idx].n = n;
    mergedNeuronUpdateGroup2[idx].lastspike = lastspike;
    mergedNeuronUpdateGroup2[idx].not_refractory = not_refractory;
    mergedNeuronUpdateGroup2[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronUpdateGroup2[idx].numNeurons = numNeurons;
}
static MergedNeuronSpikeQueueUpdateGroup0 mergedNeuronSpikeQueueUpdateGroup0[3];
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt) {
    mergedNeuronSpikeQueueUpdateGroup0[idx].spkCnt = spkCnt;
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void updateNeurons(double t) {
     {
        // merged neuron spike queue update group 0
        for(unsigned int g = 0; g < 3; g++) {
            const auto *group = &mergedNeuronSpikeQueueUpdateGroup0[g]; 
            group->spkCnt[0] = 0;
        }
    }
     {
        // merged neuron update group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup0[g]; 
            
            for(unsigned int i = 0; i < group->numNeurons; i++) {
                
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                
                // test for and register a true spike
                if (0) {
                    group->spk[group->spkCnt[0]++] = i;
                }
            }
        }
    }
     {
        // merged neuron update group 1
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup1[g]; 
            
            for(unsigned int i = 0; i < group->numNeurons; i++) {
                int32_t li = group->i[i];
                double lV = group->V[i];
                double lg_eKC_eKC = group->g_eKC_eKC[i];
                double lg_iKC_eKC = group->g_iKC_eKC[i];
                double lh = group->h[i];
                double lm = group->m[i];
                double ln = group->n[i];
                double llastspike = group->lastspike[i];
                char lnot_refractory = group->not_refractory[i];
                
                double Isyn = 0;
                 {
                    // pull inSyn values in a coalesced access
                    double linSyn = group->inSynInSyn0[i];
                    Isyn += 0; lg_eKC_eKC += linSyn; linSyn= 0;
                    
                    group->inSynInSyn0[i] = linSyn;
                }
                 {
                    // pull inSyn values in a coalesced access
                    double linSyn = group->inSynInSyn1[i];
                    Isyn += 0; lg_iKC_eKC += linSyn; linSyn= 0;
                    
                    group->inSynInSyn1[i] = linSyn;
                }
                using namespace NeuronUpdateSupportCode0;
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                // Update "constant over DT" subexpressions (if any)
                
                
                
                // PoissonInputs targetting this group (if any)
                
                
                
                // Update state variables and the threshold condition
                
                lnot_refractory = lnot_refractory || (! (lV > (0 * (1.00000000000000002e-03))));
                double _BA_V = 1.0*(((((1.0*(((-9.50000000000000011e-02) * (1.42999999999999986e-06)) * (_brian_pow(ln, 4)))/(2.99999999999999998e-10)) + (1.0*((((5.00000000000000028e-02) * (7.15000000000000015e-06)) * lh) * (_brian_pow(lm, 3)))/(2.99999999999999998e-10))) + (1.0*((0.00000000000000000e+00) * lg_iKC_eKC)/(2.99999999999999998e-10))) + (1.0*((-9.19999999999999984e-02) * lg_eKC_eKC)/(2.99999999999999998e-10))) + (1.0*((-6.35600000000000054e-02) * (2.67000000000000009e-08))/(2.99999999999999998e-10)))/(((((1.0*((- (1.42999999999999986e-06)) * (_brian_pow(ln, 4)))/(2.99999999999999998e-10)) - (1.0*(((7.15000000000000015e-06) * lh) * (_brian_pow(lm, 3)))/(2.99999999999999998e-10))) - (1.0*lg_eKC_eKC/(2.99999999999999998e-10))) - (1.0*lg_iKC_eKC/(2.99999999999999998e-10))) - (1.0*(2.67000000000000009e-08)/(2.99999999999999998e-10)));
                double _V = (- _BA_V) + ((lV + _BA_V) * exp(DT * (((((1.0*((- (1.42999999999999986e-06)) * (_brian_pow(ln, 4)))/(2.99999999999999998e-10)) - (1.0*(((7.15000000000000015e-06) * lh) * (_brian_pow(lm, 3)))/(2.99999999999999998e-10))) - (1.0*lg_eKC_eKC/(2.99999999999999998e-10))) - (1.0*lg_iKC_eKC/(2.99999999999999998e-10))) - (1.0*(2.67000000000000009e-08)/(2.99999999999999998e-10)))));
                double _g_eKC_eKC = lg_eKC_eKC * exp(1.0*(- DT)/(5.00000000000000010e-03));
                double _g_iKC_eKC = lg_iKC_eKC * exp(1.0*(- DT)/(1.00000000000000002e-02));
                double _BA_h = 1.0*((0.128 * exp(1.0*(- 8)/3)) * exp(1.0*(- lV)/(18 * (1.00000000000000002e-03))))/((1.00000000000000002e-03) * ((1.0*(- 4)/((1.00000000000000002e-03) + (((1.00000000000000002e-03) * exp(- 5)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03)))))) - (1.0*(0.128 * exp(1.0*(- 8)/3))/((1.00000000000000002e-03) * (_brian_pow(exp(1.0*lV/(1.00000000000000002e-03)), 1.0*1/18))))));
                double _h = (- _BA_h) + ((_BA_h + lh) * exp(DT * ((1.0*(- 4)/((1.00000000000000002e-03) + (((1.00000000000000002e-03) * exp(- 5)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03)))))) - (1.0*(0.128 * exp(1.0*(- 8)/3))/((1.00000000000000002e-03) * (_brian_pow(exp(1.0*lV/(1.00000000000000002e-03)), 1.0*1/18)))))));
                double _BA_m = 1.0*((1.0*((- 0.32) * lV)/(((- (1.00000000000000002e-03)) * (1.00000000000000002e-03)) + ((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(- 13)) * exp(1.0*(- lV)/(4 * (1.00000000000000002e-03)))))) - (1.0*16.64/((- (1.00000000000000002e-03)) + (((1.00000000000000002e-03) * exp(- 13)) * exp(1.0*(- lV)/(4 * (1.00000000000000002e-03)))))))/((((1.0*((- 0.28) * lV)/(((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(5)) * exp(1.0*lV/(5 * (1.00000000000000002e-03)))) - ((1.00000000000000002e-03) * (1.00000000000000002e-03)))) + (1.0*(0.32 * lV)/(((- (1.00000000000000002e-03)) * (1.00000000000000002e-03)) + ((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(- 13)) * exp(1.0*(- lV)/(4 * (1.00000000000000002e-03))))))) - (1.0*7.0/((((1.00000000000000002e-03) * exp(5)) * exp(1.0*lV/(5 * (1.00000000000000002e-03)))) - (1.00000000000000002e-03)))) + (1.0*16.64/((- (1.00000000000000002e-03)) + (((1.00000000000000002e-03) * exp(- 13)) * exp(1.0*(- lV)/(4 * (1.00000000000000002e-03)))))));
                double _m = (- _BA_m) + ((_BA_m + lm) * exp(DT * ((((1.0*((- 0.28) * lV)/(((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(5)) * exp(1.0*lV/(5 * (1.00000000000000002e-03)))) - ((1.00000000000000002e-03) * (1.00000000000000002e-03)))) + (1.0*(0.32 * lV)/(((- (1.00000000000000002e-03)) * (1.00000000000000002e-03)) + ((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(- 13)) * exp(1.0*(- lV)/(4 * (1.00000000000000002e-03))))))) - (1.0*7.0/((((1.00000000000000002e-03) * exp(5)) * exp(1.0*lV/(5 * (1.00000000000000002e-03)))) - (1.00000000000000002e-03)))) + (1.0*16.64/((- (1.00000000000000002e-03)) + (((1.00000000000000002e-03) * exp(- 13)) * exp(1.0*(- lV)/(4 * (1.00000000000000002e-03)))))))));
                double _BA_n = 1.0*((1.0*((- 0.032) * lV)/(((- (1.00000000000000002e-03)) * (1.00000000000000002e-03)) + ((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(- 10)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03)))))) - (1.0*1.6/((- (1.00000000000000002e-03)) + (((1.00000000000000002e-03) * exp(- 10)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03)))))))/(((1.0*(0.032 * lV)/(((- (1.00000000000000002e-03)) * (1.00000000000000002e-03)) + ((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(- 10)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03)))))) + (1.0*1.6/((- (1.00000000000000002e-03)) + (((1.00000000000000002e-03) * exp(- 10)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03))))))) - (1.0*(0.5 * exp(1.0*(- 11)/8))/((1.00000000000000002e-03) * (_brian_pow(exp(1.0*lV/(1.00000000000000002e-03)), 1.0*1/40)))));
                double _n = (- _BA_n) + ((_BA_n + ln) * exp(DT * (((1.0*(0.032 * lV)/(((- (1.00000000000000002e-03)) * (1.00000000000000002e-03)) + ((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(- 10)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03)))))) + (1.0*1.6/((- (1.00000000000000002e-03)) + (((1.00000000000000002e-03) * exp(- 10)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03))))))) - (1.0*(0.5 * exp(1.0*(- 11)/8))/((1.00000000000000002e-03) * (_brian_pow(exp(1.0*lV/(1.00000000000000002e-03)), 1.0*1/40)))))));
                lV = _V;
                lg_eKC_eKC = _g_eKC_eKC;
                lg_iKC_eKC = _g_iKC_eKC;
                lh = _h;
                lm = _m;
                ln = _n;
                char _cond = (lV > (0 * (1.00000000000000002e-03))) && lnot_refractory;
                // test for and register a true spike
                if (_cond) {
                    group->spk[group->spkCnt[0]++] = i;
                    // spike reset code
                    llastspike = t;
                    lnot_refractory = false;
                }
                group->i[i] = li;
                group->V[i] = lV;
                group->g_eKC_eKC[i] = lg_eKC_eKC;
                group->g_iKC_eKC[i] = lg_iKC_eKC;
                group->h[i] = lh;
                group->m[i] = lm;
                group->n[i] = ln;
                group->lastspike[i] = llastspike;
                group->not_refractory[i] = lnot_refractory;
            }
        }
    }
     {
        // merged neuron update group 2
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup2[g]; 
            
            for(unsigned int i = 0; i < group->numNeurons; i++) {
                int32_t li = group->i[i];
                double lV = group->V[i];
                double lg_PN_iKC = group->g_PN_iKC[i];
                double lh = group->h[i];
                double lm = group->m[i];
                double ln = group->n[i];
                double llastspike = group->lastspike[i];
                char lnot_refractory = group->not_refractory[i];
                
                double Isyn = 0;
                 {
                    // pull inSyn values in a coalesced access
                    double linSyn = group->inSynInSyn0[i];
                    Isyn += 0; lg_PN_iKC += linSyn; linSyn= 0;
                    
                    group->inSynInSyn0[i] = linSyn;
                }
                using namespace NeuronUpdateSupportCode0;
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                // Update "constant over DT" subexpressions (if any)
                
                
                
                // PoissonInputs targetting this group (if any)
                
                
                
                // Update state variables and the threshold condition
                
                lnot_refractory = lnot_refractory || (! (lV > (0 * (1.00000000000000002e-03))));
                double _BA_V = 1.0*((((1.0*(((-9.50000000000000011e-02) * (1.42999999999999986e-06)) * (_brian_pow(ln, 4)))/(2.99999999999999998e-10)) + (1.0*((((5.00000000000000028e-02) * (7.15000000000000015e-06)) * lh) * (_brian_pow(lm, 3)))/(2.99999999999999998e-10))) + (1.0*((0.00000000000000000e+00) * lg_PN_iKC)/(2.99999999999999998e-10))) + (1.0*((-6.35600000000000054e-02) * (2.67000000000000009e-08))/(2.99999999999999998e-10)))/((((1.0*((- (1.42999999999999986e-06)) * (_brian_pow(ln, 4)))/(2.99999999999999998e-10)) - (1.0*(((7.15000000000000015e-06) * lh) * (_brian_pow(lm, 3)))/(2.99999999999999998e-10))) - (1.0*lg_PN_iKC/(2.99999999999999998e-10))) - (1.0*(2.67000000000000009e-08)/(2.99999999999999998e-10)));
                double _V = (- _BA_V) + ((lV + _BA_V) * exp(DT * ((((1.0*((- (1.42999999999999986e-06)) * (_brian_pow(ln, 4)))/(2.99999999999999998e-10)) - (1.0*(((7.15000000000000015e-06) * lh) * (_brian_pow(lm, 3)))/(2.99999999999999998e-10))) - (1.0*lg_PN_iKC/(2.99999999999999998e-10))) - (1.0*(2.67000000000000009e-08)/(2.99999999999999998e-10)))));
                double _g_PN_iKC = lg_PN_iKC * exp(1.0*(- DT)/(2.00000000000000004e-03));
                double _BA_h = 1.0*((0.128 * exp(1.0*(- 8)/3)) * exp(1.0*(- lV)/(18 * (1.00000000000000002e-03))))/((1.00000000000000002e-03) * ((1.0*(- 4)/((1.00000000000000002e-03) + (((1.00000000000000002e-03) * exp(- 5)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03)))))) - (1.0*(0.128 * exp(1.0*(- 8)/3))/((1.00000000000000002e-03) * (_brian_pow(exp(1.0*lV/(1.00000000000000002e-03)), 1.0*1/18))))));
                double _h = (- _BA_h) + ((_BA_h + lh) * exp(DT * ((1.0*(- 4)/((1.00000000000000002e-03) + (((1.00000000000000002e-03) * exp(- 5)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03)))))) - (1.0*(0.128 * exp(1.0*(- 8)/3))/((1.00000000000000002e-03) * (_brian_pow(exp(1.0*lV/(1.00000000000000002e-03)), 1.0*1/18)))))));
                double _BA_m = 1.0*((1.0*((- 0.32) * lV)/(((- (1.00000000000000002e-03)) * (1.00000000000000002e-03)) + ((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(- 13)) * exp(1.0*(- lV)/(4 * (1.00000000000000002e-03)))))) - (1.0*16.64/((- (1.00000000000000002e-03)) + (((1.00000000000000002e-03) * exp(- 13)) * exp(1.0*(- lV)/(4 * (1.00000000000000002e-03)))))))/((((1.0*((- 0.28) * lV)/(((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(5)) * exp(1.0*lV/(5 * (1.00000000000000002e-03)))) - ((1.00000000000000002e-03) * (1.00000000000000002e-03)))) + (1.0*(0.32 * lV)/(((- (1.00000000000000002e-03)) * (1.00000000000000002e-03)) + ((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(- 13)) * exp(1.0*(- lV)/(4 * (1.00000000000000002e-03))))))) - (1.0*7.0/((((1.00000000000000002e-03) * exp(5)) * exp(1.0*lV/(5 * (1.00000000000000002e-03)))) - (1.00000000000000002e-03)))) + (1.0*16.64/((- (1.00000000000000002e-03)) + (((1.00000000000000002e-03) * exp(- 13)) * exp(1.0*(- lV)/(4 * (1.00000000000000002e-03)))))));
                double _m = (- _BA_m) + ((_BA_m + lm) * exp(DT * ((((1.0*((- 0.28) * lV)/(((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(5)) * exp(1.0*lV/(5 * (1.00000000000000002e-03)))) - ((1.00000000000000002e-03) * (1.00000000000000002e-03)))) + (1.0*(0.32 * lV)/(((- (1.00000000000000002e-03)) * (1.00000000000000002e-03)) + ((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(- 13)) * exp(1.0*(- lV)/(4 * (1.00000000000000002e-03))))))) - (1.0*7.0/((((1.00000000000000002e-03) * exp(5)) * exp(1.0*lV/(5 * (1.00000000000000002e-03)))) - (1.00000000000000002e-03)))) + (1.0*16.64/((- (1.00000000000000002e-03)) + (((1.00000000000000002e-03) * exp(- 13)) * exp(1.0*(- lV)/(4 * (1.00000000000000002e-03)))))))));
                double _BA_n = 1.0*((1.0*((- 0.032) * lV)/(((- (1.00000000000000002e-03)) * (1.00000000000000002e-03)) + ((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(- 10)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03)))))) - (1.0*1.6/((- (1.00000000000000002e-03)) + (((1.00000000000000002e-03) * exp(- 10)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03)))))))/(((1.0*(0.032 * lV)/(((- (1.00000000000000002e-03)) * (1.00000000000000002e-03)) + ((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(- 10)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03)))))) + (1.0*1.6/((- (1.00000000000000002e-03)) + (((1.00000000000000002e-03) * exp(- 10)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03))))))) - (1.0*(0.5 * exp(1.0*(- 11)/8))/((1.00000000000000002e-03) * (_brian_pow(exp(1.0*lV/(1.00000000000000002e-03)), 1.0*1/40)))));
                double _n = (- _BA_n) + ((_BA_n + ln) * exp(DT * (((1.0*(0.032 * lV)/(((- (1.00000000000000002e-03)) * (1.00000000000000002e-03)) + ((((1.00000000000000002e-03) * (1.00000000000000002e-03)) * exp(- 10)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03)))))) + (1.0*1.6/((- (1.00000000000000002e-03)) + (((1.00000000000000002e-03) * exp(- 10)) * exp(1.0*(- lV)/(5 * (1.00000000000000002e-03))))))) - (1.0*(0.5 * exp(1.0*(- 11)/8))/((1.00000000000000002e-03) * (_brian_pow(exp(1.0*lV/(1.00000000000000002e-03)), 1.0*1/40)))))));
                lV = _V;
                lg_PN_iKC = _g_PN_iKC;
                lh = _h;
                lm = _m;
                ln = _n;
                char _cond = (lV > (0 * (1.00000000000000002e-03))) && lnot_refractory;
                // test for and register a true spike
                if (_cond) {
                    group->spk[group->spkCnt[0]++] = i;
                    // spike reset code
                    llastspike = t;
                    lnot_refractory = false;
                }
                group->i[i] = li;
                group->V[i] = lV;
                group->g_PN_iKC[i] = lg_PN_iKC;
                group->h[i] = lh;
                group->m[i] = lm;
                group->n[i] = ln;
                group->lastspike[i] = llastspike;
                group->not_refractory[i] = lnot_refractory;
            }
        }
    }
}

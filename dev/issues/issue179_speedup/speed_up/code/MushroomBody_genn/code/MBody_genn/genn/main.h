//--------------------------------------------------------------------------
/*! \file main.h

\brief Header file containing global variables and macros used in running the model.
*/
//--------------------------------------------------------------------------

using namespace std;
#include <cassert>
#include <cstdint>
#include <vector>

// we will hard-code some stuff ... because at the end of the day that is 
// what we will do for the CUDA version

#define DBG_SIZE 10000


//----------------------------------------------------------------------
// other stuff:
// These variables (if any) are needed to be able to efficiently copy brian
// synapse variables into genn SPARSE synaptic arrays (needed for run_regularly)

std::vector<size_t> sparseSynapseIndicessynapses;
std::vector<size_t> sparseSynapseIndicessynapses_2;
std::vector<size_t> sparseSynapseIndicessynapses_1;

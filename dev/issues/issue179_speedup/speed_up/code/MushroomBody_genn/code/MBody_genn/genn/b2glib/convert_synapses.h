#pragma once

// scalar can be any scalar type such as float, double
#include <stdint.h>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>

template<class scalar>
void convert_dynamic_arrays_2_dense_matrix(vector<int32_t> &source, vector<int32_t> &target, vector<scalar> &gvector, scalar *g, int srcNN, int trgNN)
{
    assert(source.size() == target.size());
    assert(source.size() == gvector.size());
    unsigned int size= source.size(); 
    for (int s= 0; s < srcNN; s++) {
        for (int t= 0; t < trgNN; t++) {
            g[s*trgNN+t]= (scalar)NAN;
        }
    }

    for (int i= 0; i < size; i++) {
        assert(source[i] < srcNN);
        assert(target[i] < trgNN);
        // Check for duplicate entries
        if (! std::isnan(g[source[i]*trgNN+target[i]])) {
            std::cerr << "*****" << std::endl;
            std::cerr << "ERROR  Cannot run GeNN simulation: More than one synapse for pair " << source[i] << " - " << target[i] << " and DENSE connectivity used." << std::endl;
            std::cerr << "*****" << std::endl;
            exit(222);
        }
        g[source[i]*trgNN+target[i]]= gvector[i];
    }
    for (int s= 0; s < srcNN; s++) {
        for (int t= 0; t < trgNN; t++) {
	  if (std::isnan(g[s*trgNN+t]))
                g[s*trgNN+t] = 0.0;
        }
    }
}

namespace b2g {
    unsigned int FULL_MONTY= 0;
    unsigned int COPY_ONLY= 1;
};

void initialize_sparse_synapses(const vector<int32_t> &source, const vector<int32_t> &target,
                                unsigned int *rowLength, unsigned int *ind, unsigned int maxRowLength,
                                int srcNN, int trgNN,
                                vector<size_t> &indices)
{
    // Initially zero row lengths
    std::fill_n(rowLength, srcNN, 0);

    const size_t size = source.size();

    // Reserve indices
    indices.clear();
    indices.reserve(size);

    // Loop through input arrays
    for (size_t i= 0; i < size; i++) {
        assert(source[i] < srcNN);
        assert(target[i] < trgNN);

        // Calculate index of synapse in ragged structure
        const size_t index = (source[i] * maxRowLength) + rowLength[source[i]];

        // Add index to vector and insert postsynaptic index into correct location
        // **TODO** insert in correct position to keep sorted
        indices.push_back(index);
        ind[index] = target[i];

        // Increment row length
        rowLength[source[i]]++;
    }
}


template<class scalar>
void convert_dynamic_arrays_2_sparse_synapses(const vector<scalar> &gvector, const vector<size_t> &indices,
                                              scalar *gv, int srcNN, int trgNN)
{
    const size_t size = indices.size();
    for (size_t i= 0; i < size; i++) {
        // Insert postsynaptic index in correct location
        gv[indices[i]] = gvector[i];
    }
}


template<class scalar>
void convert_dense_matrix_2_dynamic_arrays(scalar *g, int srcNN, int trgNN, vector<int32_t> &source, vector<int32_t> &target, vector<scalar> &gvector)
{
    assert(source.size() == target.size());
    assert(source.size() == gvector.size());
    unsigned int size= source.size(); 
    for (int i= 0; i < size; i++) {
        assert(source[i] < srcNN);
        assert(target[i] < trgNN);
        gvector[i]= g[source[i]*trgNN+target[i]];
    }
}

template<class scalar>
void convert_sparse_synapses_2_dynamic_arrays(unsigned int *rowLength, unsigned int *ind, unsigned int maxRowLength,
                                              scalar *gv, int srcNN, int trgNN, vector<int32_t> &source, vector<int32_t> &target, vector<scalar> &gvector, unsigned int mode)
{
// note: this does not preserve the original order of entries in the brian arrays - is that a problem?
    if (mode == b2g::FULL_MONTY) {
        assert(source.size() == target.size());
        assert(source.size() == gvector.size());
        size_t cnt= 0;
        for (int i= 0; i < srcNN; i++) {
            for (int j= 0; j < rowLength[i]; j++) {
                source[cnt]= i;
                target[cnt]= ind[(i * maxRowLength) + j];
                gvector[cnt]= gv[(i * maxRowLength) + j];
                cnt++;
            }
        }
    }
    else {
        size_t cnt= 0;
        for (int i= 0; i < srcNN; i++) {
            for (int j= 0; j < rowLength[i]; j++) {
                gvector[cnt++]= gv[(i * maxRowLength) + j];
            }
        }
    }
}

void create_hidden_weightmatrix(vector<int32_t> &source, vector<int32_t> &target, char* hwm, int srcNN, int trgNN)
{
    for (int s= 0; s < srcNN; s++) {
    for (int t= 0; t < trgNN; t++) {
        hwm[s*trgNN+t]= 0;
    }
    }
    for (int i= 0; i < source.size(); i++) {
    hwm[source[i]*trgNN+target[i]]= 1;
    }
}

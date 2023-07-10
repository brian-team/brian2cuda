{# USES_VARIABLES { Cm, dt, v, N, Ic, Ri,
                  _ab_star0, _ab_star1, _ab_star2, _b_plus, _b_minus,
                  _v_star, _u_plus, _u_minus,
                  _v_previous,
                  _gtot_all, _I0_all,
                  _c,
                  _P_diag, _P_parent, _P_children,
                  _B, _morph_parent_i, _starts, _ends,
                  _morph_children, _morph_children_num, _morph_idxchild,
                  _invr0, _invrn, _invr,
                  r_length_1, r_length_2, area } #}


{% extends 'common_group.cu' %}


{### BEFORE RUN ###}
{% block before_run_host_maincode %}

    ///// HOST_CONSTANTS ///////////
    %HOST_CONSTANTS%

    // Number of group units in stateupdate is always N (no subgroups)
    const int _N = N;

    ///// ADDITIONAL_HOST_CODE /////
    %ADDITIONAL_HOST_CODE%

    {# needed to translate _array... to _ptr_array... #}
    ///// pointers_lines /////
    {{pointers_lines|autoindent}}

    // The following code is simply copied from spatialneuron_prepare.cpp
    // of the cpp_standalone device (except for copying to GPU memory at bottom of file)

    const double _Ri = {{Ri}};  // Ri is a shared variable

    // Inverse axial resistance
    {# {{ openmp_pragma('parallel-static') }} #}
    for (int _i=1; _i<_N; _i++)
        {{_invr}}[_i] = 1.0/(_Ri*(1/{{r_length_2}}[_i-1] + 1/{{r_length_1}}[_i]));
    // Cut sections
    {# {{ openmp_pragma('parallel-static') }} #}
    for (int _i=0; _i<_num_starts; _i++)
        {{_invr}}[{{_starts}}[_i]] = 0;

    // Linear systems
    // The particular solution
    // a[i,j]=ab[u+i-j,j]   --  u is the number of upper diagonals = 1
    {# {{ openmp_pragma('parallel-static') }} #}
    for (int _i=0; _i<_N; _i++)
        {{_ab_star1}}[_i] = (-({{Cm}}[_i] / {{dt}}) - {{_invr}}[_i] / {{area}}[_i]);
    {# {{ openmp_pragma('parallel-static') }} #}
    for (int _i=1; _i<_N; _i++)
    {
        {{_ab_star0}}[_i] = {{_invr}}[_i] / {{area}}[_i-1];
        {{_ab_star2}}[_i-1] = {{_invr}}[_i] / {{area}}[_i];
        {{_ab_star1}}[_i-1] -= {{_invr}}[_i] / {{area}}[_i-1];
    }

    // Set the boundary conditions
    for (int _counter=0; _counter<_num_starts; _counter++)
    {
        const int _first = {{_starts}}[_counter];
        const int _last = {{_ends}}[_counter] - 1;  // the compartment indices are in the interval [starts, ends[
        // Inverse axial resistances at the ends: r0 and rn
        const double _invr0 = {{r_length_1}}[_first]/_Ri;
        const double _invrn = {{r_length_2}}[_last]/_Ri;
        {{_invr0}}[_counter] = _invr0;
        {{_invrn}}[_counter] = _invrn;
        // Correction for boundary conditions
        {{_ab_star1}}[_first] -= (_invr0 / {{area}}[_first]);
        {{_ab_star1}}[_last] -= (_invrn / {{area}}[_last]);
        // RHS for homogeneous solutions
        {{_b_plus}}[_last] = -(_invrn / {{area}}[_last]);
        {{_b_minus}}[_first] = -(_invr0 / {{area}}[_first]);
    }

    // Copy prepared arrays to GPU
    {% for var in ['_invr', 'Ri', 'Cm', 'dt', 'area', 'r_length_1',
                       'r_length_2', '_ab_star0', '_ab_star1', '_ab_star2',
                       '_starts', '_ends', '_invr0', '_invrn', '_b_plus',
                       '_b_minus'] %}
    {% set varname = get_array_name(variables[var], access_data=False) %}

    // {{var}}
    CUDA_SAFE_CALL(
            cudaMemcpy(dev{{varname}}, {{varname}},
                sizeof({{c_data_type(variables[var].dtype)}})*_num_{{varname}},
                cudaMemcpyHostToDevice)
            );
    {% endfor %}
{% endblock before_run_host_maincode %}


{### RUN ###}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
// FIRST: KERNEL DEFINITIONS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
// kernel 1: compute g_total and I_0
// (independent: everything, i.e., compartments and branches)
// remark: for this kernel we use the common_group.cu to have the machinery of optimal thread
//         no calculation machinery
{% block extra_vector_code %}
        {{_gtot_all}}[_idx] = _gtot;
        {{_I0_all}}[_idx] = _I0;

        {{_v_previous}}[_idx] = {{v}}[_idx];
{% endblock %}

// additional kernels (linear systems and solution combination)
{% block extra_device_helper %}

/////////////////////////////////////////////////////
// kernel 2: solve three tridiagonal system (one matrix of size compartment with three right hand sides)
// (independent: branches)
// remark: here we apply over the branches in parallel the Thomas algorithm
//         (i.e., Gaussian elimination for a tridiagonal system) which has a
//         runtime complexity O(compartments) but is inherently sequential
//         => run no as many blocks as branches with one thread each
//         => trivial optimization possible by using three threads (one per rhs)
//         => optimization possible e.g. by using cyclic reduction [more parallel]

__global__ void _tridiagsolve_kernel_{{codeobj_name}}(
    int _N,
    ///// KERNEL_PARAMETERS /////
    %KERNEL_PARAMETERS%
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * blockDim.x + tid;

    ///// KERNEL_CONSTANTS /////
    %KERNEL_CONSTANTS%

    ///// translate clock variables into pointers /////
    {{clock_pointers|autoindent}}

    // we need to run the kernel with 1 thread per block (to be changed by optimization)
    assert(tid == 0 && bid == _idx);

    // each thread processes the tridiagsystem of one branch
    const int _i = _idx;

    // below all the code is simply copied from spatialstateupdate.cpp

    // first and last index of the i-th section
    const int _j_start = {{_starts}}[_i];
    const int _j_end = {{_ends}}[_i];

    double _ai, _bi, _m; // helper variables

    // upper triangularization of tridiagonal system for _v_star, _u_plus, and _u_minus
    for(int _j=_j_start; _j<_j_end; _j++)
    {
        {{_v_star}}[_j]=-({{Cm}}[_j]/{{dt}}*{{v}}[_j])-{{_I0_all}}[_j]; // RHS -> _v_star (solution)
        {{_u_plus}}[_j]={{_b_plus}}[_j]; // RHS -> _u_plus (solution)
        {{_u_minus}}[_j]={{_b_minus}}[_j]; // RHS -> _u_minus (solution)
        _bi={{_ab_star1}}[_j]-{{_gtot_all}}[_j]; // main diagonal
        if (_j<_N-1)
            {{_c}}[_j]={{_ab_star0}}[_j+1]; // superdiagonal
        if (_j>0)
        {
            _ai={{_ab_star2}}[_j-1]; // subdiagonal
            _m=1.0/(_bi-_ai*{{_c}}[_j-1]);
            {{_c}}[_j]={{_c}}[_j]*_m;
            {{_v_star}}[_j]=({{_v_star}}[_j] - _ai*{{_v_star}}[_j-1])*_m;
            {{_u_plus}}[_j]=({{_u_plus}}[_j] - _ai*{{_u_plus}}[_j-1])*_m;
            {{_u_minus}}[_j]=({{_u_minus}}[_j] - _ai*{{_u_minus}}[_j-1])*_m;
        } else
        {
            {{_c}}[0]={{_c}}[0]/_bi;
            {{_v_star}}[0]={{_v_star}}[0]/_bi;
            {{_u_plus}}[0]={{_u_plus}}[0]/_bi;
            {{_u_minus}}[0]={{_u_minus}}[0]/_bi;
        }
    }
    // backwards substituation of the upper triangularized system for _v_star
    for(int _j=_j_end-2; _j>=_j_start; _j--)
    {
        {{_v_star}}[_j]={{_v_star}}[_j] - {{_c}}[_j]*{{_v_star}}[_j+1];
        {{_u_plus}}[_j]={{_u_plus}}[_j] - {{_c}}[_j]*{{_u_plus}}[_j+1];
        {{_u_minus}}[_j]={{_u_minus}}[_j] - {{_c}}[_j]*{{_u_minus}}[_j+1];
    }
}



/////////////////////////////////////////////////////
// kernel 3: solve the coupling system (one matrix of size branches)
// (no independence)
// remark: applies the Hines algorithm having O(branches) complexity
//         => run with one block one thread

__global__ void _coupling_kernel_{{codeobj_name}}(
    int _N,
    ///// KERNEL_PARAMETERS /////
    %KERNEL_PARAMETERS%
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * blockDim.x + tid;

    ///// KERNEL_CONSTANTS /////
    %KERNEL_CONSTANTS%

    ///// translate clock variables into pointers /////
    {{clock_pointers|autoindent}}

    // we need to run the kernel with 1 thread, 1 block
    assert(_idx == 0);

    // below all the code is simply copied from spatialstateupdate.cpp

        // indexing for _P_children which contains the elements above the diagonal of the coupling matrix _P
    const int _children_rowlength = _num_morph_children/_num_morph_children_num;
    #define _IDX_C(idx_row,idx_col) _children_rowlength * idx_row + idx_col

    // step a) construct the coupling system with matrix _P in sparse form. s.t.
    // _P_diag contains the diagonal elements
    // _P_children contains the super diagonal entries
    // _P_parent contains the single sub diagonal entry for each row
    // _B contains the right hand side
    for (int _i=0; _i<_num_B - 1; _i++)
    {
        const int _i_parent = {{_morph_parent_i}}[_i];
        const int _i_childind = {{_morph_idxchild}}[_i];
        const int _first = {{_starts}}[_i];
        const int _last = {{_ends}}[_i] - 1;  // the compartment indices are in the interval [starts, ends[
        const double _invr0 = {{_invr0}}[_i];
        const double _invrn = {{_invrn}}[_i];

        // Towards parent
        if (_i == 0) // first section, sealed end
        {
            // sparse matrix version
            {{_P_diag}}[0] = {{_u_minus}}[_first] - 1;
            {{_P_children}}[_IDX_C(0,0)] = {{_u_plus}}[_first];

            // RHS
            {{_B}}[0] = -{{_v_star}}[_first];
        }
        else
        {
            // sparse matrix version
            {{_P_diag}}[_i_parent] += (1 - {{_u_minus}}[_first]) * _invr0;
            {{_P_children}}[_IDX_C(_i_parent, _i_childind)] = -{{_u_plus}}[_first] * _invr0;

            // RHS
            {{_B}}[_i_parent] += {{_v_star}}[_first] * _invr0;
        }

        // Towards children

        // sparse matrix version
        {{_P_diag}}[_i+1] = (1 - {{_u_plus}}[_last]) * _invrn;
        {{_P_parent}}[_i] = -{{_u_minus}}[_last] * _invrn;

        // RHS
        {{_B}}[_i+1] = {{_v_star}}[_last] * _invrn;
    }


    // step b) solve the linear system (the result will be stored in the former rhs _B in the end)
    // use efficient O(n) solution of the sparse linear system (structure-specific Gaussian elemination)

    // part 1: lower triangularization
    for (int _i=_num_B-1; _i>=0; _i--) {
        const int _num_children = {{_morph_children_num}}[_i];

        // for every child eliminate the corresponding matrix element of row i
        for (int _k=0; _k<_num_children; _k++) {
            int _j = {{_morph_children}}[_IDX_C(_i,_k)]; // child index

            // subtracting _subfac times the j-th from the i-th row
            double _subfac = {{_P_children}}[_IDX_C(_i,_k)] / {{_P_diag}}[_j]; // element i,j appears only here

            // the following commented (superdiagonal) element is not used in the following anymore since
            // it is 0 by definition of (lower) triangularization; we keep it here for algorithmic clarity
            //{{_P_children}}[_IDX_C(_i,_k)] = {{_P_children}}[_IDX_C(_i,_k)]  - _subfac * {{_P_diag}}[_j]; // = 0;

            {{_P_diag}}[_i] = {{_P_diag}}[_i]  - _subfac * {{_P_parent}}[_j-1]; // note: element j,i is only used here
            {{_B}}[_i] = {{_B}}[_i] - _subfac * {{_B}}[_j];

        }
    }

    // part 2: forwards substitution
    {{_B}}[0] = {{_B}}[0] / {{_P_diag}}[0]; // the first section does not have a parent
    for (int _i=1; _i<_num_B; _i++) {
        const int _j = {{_morph_parent_i}}[_i-1]; // parent index
        {{_B}}[_i] = {{_B}}[_i] - {{_P_parent}}[_i-1] * {{_B}}[_j];
        {{_B}}[_i] = {{_B}}[_i] / {{_P_diag}}[_i];

    }

}


/////////////////////////////////////////////////////
// kernel 4: for each section compute the final solution by linear
//           combination of the general solution
// (independent: everything, i.e., compartments and branches)
// remark: branch granularity in implementation used since parents/children are combined for each branch

__global__ void _combine_kernel_{{codeobj_name}}(
    int _N,
    ///// KERNEL_PARAMETERS /////
    %KERNEL_PARAMETERS%
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * blockDim.x + tid;

    ///// KERNEL_CONSTANTS /////
    %KERNEL_CONSTANTS%

    ///// translate clock variables into pointers /////
    {{clock_pointers|autoindent}}

    // we need to run the kernel with 1 thread per block (to be changed by optimization)
    assert(tid == 0 && bid == _idx);

    // each thread combines the tridiagsystem of one branch
    const int _i = _idx;

    // below all the code is simply copied from spatialstateupdate.cpp

    const int _i_parent = {{_morph_parent_i}}[_i];
    const int _j_start = {{_starts}}[_i];
    const int _j_end = {{_ends}}[_i];
    for (int _j=_j_start; _j<_j_end; _j++)
        if (_j < _numv)  // don't go beyond the last element
            {{v}}[_j] = {{_v_star}}[_j] + {{_B}}[_i_parent] * {{_u_minus}}[_j]
                                       + {{_B}}[_i+1] * {{_u_plus}}[_j];


}



/////////////////////////////////////////////////////
// kernel 5: update currents
// (independent: everything, i.e., compartments and branches)

__global__ void _currents_kernel_{{codeobj_name}}(
    int _N,
    ///// KERNEL_PARAMETERS /////
    %KERNEL_PARAMETERS%
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * blockDim.x + tid;

    ///// KERNEL_CONSTANTS /////
    %KERNEL_CONSTANTS%

    ///// translate clock variables into pointers /////
    {{clock_pointers|autoindent}}

    if(_idx >= _N)
    {
        return;
    }

    // each thread processes the tridiagsystem of one branch
    const int _i = _idx;

    {{Ic}}[_i] = {{Cm}}[_i]*({{v}}[_i] - {{_v_previous}}[_i])/{{dt}};

}



{% endblock extra_device_helper %}


/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
// SECOND/LAST: KERNEL EXECUTIONS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

{% block extra_kernel_call_post %}

    // kernel 1 is automatically run (via common_group.cu), particularly with full occupancy
{#
    {% if profiled %}
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    {{codeobj_name}}_kernel_integration_profiling_info += (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    {% endif %}
#}

    // run kernel 2 (tridiag solve): branches many blocks with one thread each
{#
    {% if profiled %}
    std::clock_t _start_time_tridiagsolve = std::clock();
    {% endif %}
#}
    int num_blocks_tridiagsolve = _num_B-1;
    int num_threads_tridiagsolve = 1;
    _tridiagsolve_kernel_{{codeobj_name}}<<<num_blocks_tridiagsolve, num_threads_tridiagsolve>>>(
            _N,
            ///// HOST_PARAMETERS /////
            %HOST_PARAMETERS%
        );
    CUDA_CHECK_ERROR("_tridiagsolve_kernel_{{codeobj_name}}");
{#
    {% if profiled %}
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    {{codeobj_name}}_kernel_tridiagsolve_profiling_info += (double)(std::clock() -_start_time_tridiagsolve)/CLOCKS_PER_SEC;
    {% endif %}
#}

    // kernel 3 (coupling): one block one thread
{#
    {% if profiled %}
    std::clock_t _start_time_coupling = std::clock();
    {% endif %}
#}
    int num_blocks_coupling = 1;
    int num_threads_coupling = 1;
    _coupling_kernel_{{codeobj_name}}<<<num_blocks_coupling, num_threads_coupling>>>(
            _N,
            ///// HOST_PARAMETERS /////
            %HOST_PARAMETERS%
        );
    CUDA_CHECK_ERROR("_coupling_kernel_{{codeobj_name}}");
{#
    {% if profiled %}
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    {{codeobj_name}}_kernel_coupling_profiling_info += (double)(std::clock() -_start_time_coupling)/CLOCKS_PER_SEC;
    {% endif %}
#}

    // kernel 4 (combine): branches many blocks with one thread each
{#
    {% if profiled %}
    std::clock_t _start_time_combine = std::clock();
    {% endif %}
#}
    int num_blocks_combine = _num_B-1;
    int num_threads_combine = 1;
    _combine_kernel_{{codeobj_name}}<<<num_blocks_combine, num_threads_combine>>>(
            _N,
            ///// HOST_PARAMETERS /////
            %HOST_PARAMETERS%
        );
    CUDA_CHECK_ERROR("_combine_kernel_{{codeobj_name}}");
{#
    {% if profiled %}
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    {{codeobj_name}}_kernel_combine_profiling_info += (double)(std::clock() -_start_time_combine)/CLOCKS_PER_SEC;
    {% endif %}
#}

    // kernel 5 (final currents): max. occupancy

    // calculate max. occupancy => num_threads, num_blocks
    // first try to use it again

        static int num_threads_currents, num_blocks_currents;
        static bool first_run_custom = true;
        if (first_run_custom)
        {

            // calculate number of threads that maximize occupancy
            // and also the corresponding number of blocks
            // the code below is adapted from common_group.cu
            int min_num_threads_currents; // The minimum grid size needed to achieve the
                                 // maximum occupancy for a full device launch

            CUDA_SAFE_CALL(
                    cudaOccupancyMaxPotentialBlockSize(&min_num_threads_currents, &num_threads_currents,
                        _currents_kernel_{{codeobj_name}}, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                    );

            // Round up according to array size
            num_blocks_currents = (_N + num_threads_currents - 1) / num_threads_currents;
            // ensure our grid is executable
            struct cudaFuncAttributes funcAttrib_currents;
            CUDA_SAFE_CALL(
                    cudaFuncGetAttributes(&funcAttrib_currents, _currents_kernel_{{codeobj_name}})
                    );
            assert(num_threads_currents <= funcAttrib_currents.maxThreadsPerBlock);

            // kernel properties
            int max_active_blocks_currents;
            CUDA_SAFE_CALL(
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_currents,
                        _currents_kernel_{{codeobj_name}}, num_threads_currents, 0)
                    );

            float occupancy_currents = (max_active_blocks_currents * num_threads_currents / num_threads_per_warp) /
                              (float)(max_threads_per_sm / num_threads_per_warp);

            printf("INFO _currents\n_kernel_{{codeobj_name}}"
                       "\t%u blocks\n"
                       "\t%u threads\n"
                       "\t%i registers per thread\n"
                       "\t%i bytes statically-allocated shared memory per block\n"
                       "\t%i bytes local memory per thread\n"
                       "\t%i bytes user-allocated constant memory\n"
                       "\t%.3f theoretical occupancy\n",
                       num_blocks_currents, num_threads_currents, funcAttrib_currents.numRegs,
                       funcAttrib_currents.sharedSizeBytes, funcAttrib_currents.localSizeBytes,
                       funcAttrib_currents.constSizeBytes, occupancy_currents);


            first_run_custom = false; // now we have set up the grid

        }

{#
        {% if profiled %}
        std::clock_t _start_time_currents = std::clock();
        {% endif %}
#}
        // run kernel 5
        _currents_kernel_{{codeobj_name}}<<<num_blocks_currents, num_threads_currents>>>(
                _N,
                ///// HOST_PARAMETERS /////
                %HOST_PARAMETERS%
            );
        CUDA_CHECK_ERROR("_currents_kernel_{{codeobj_name}}");

{#
    {% if profiled %}
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    {{codeobj_name}}_kernel_currents_profiling_info += (double)(std::clock() -_start_time_currents)/CLOCKS_PER_SEC;
    {% endif %}
#}


{% endblock extra_kernel_call_post %}

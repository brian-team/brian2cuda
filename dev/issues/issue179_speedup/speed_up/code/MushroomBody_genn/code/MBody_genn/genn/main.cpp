//--------------------------------------------------------------------------
/*! \file main.cu

\brief Main entry point for the running a model simulation. 
*/
//--------------------------------------------------------------------------

#include "main.h"
#include "magicnetwork_model_CODE/definitions.h"

#include "b2glib/convert_synapses.h"
#include "code_objects/spikegeneratorgroup_codeobject.h"
#include "code_objects/spikemonitor_1_codeobject.h"
#include "code_objects/spikemonitor_2_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/synapses_1_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_1_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/synapses_1_post_push_spikes.h"
#include "code_objects/synapses_1_pre_push_spikes.h"
#include "code_objects/synapses_1_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_2_pre_push_spikes.h"
#include "code_objects/synapses_2_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/synapses_synapses_create_generator_codeobject.h"
#include "network.h"
#include "objects.h"

#include "engine.cpp"



//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the simulation of the MBody1 model network.
*/
//--------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    fprintf(stderr, "usage: main <basename> <time (s)> \n");
    return 1;
  }
  double totalTime= atof(argv[2]);
  string OutDir = std::string(argv[1]) +"_output";
  string cmd= std::string("mkdir ") +OutDir;
  system(cmd.c_str());
  string name;
  name= OutDir+ "/"+ argv[1] + ".time";
  FILE *timef= fopen(name.c_str(),"a");

  fprintf(stderr, "# DT %f \n", DT);
  fprintf(stderr, "# totalTime %f \n", totalTime);

    


  //-----------------------------------------------------------------
  // build the neuronal circuitery (calls initialize and allocateMem)
  engine eng;

  //-----------------------------------------------------------------
  // load variables and parameters and translate them from Brian to Genn
  _init_arrays();
  _load_arrays();
  rk_randomseed(brian::_mersenne_twister_states[0]);
    

  {
	  using namespace brian;
   	  
   _array_defaultclock_dt[0] = 0.0001;
   _array_defaultclock_dt[0] = 0.0001;
   _array_defaultclock_dt[0] = 0.0001;
   _dynamic_array_spikegeneratorgroup_spike_number.resize(1954);
   
                   for(int i=0; i<_dynamic_array_spikegeneratorgroup_spike_number.size(); i++)
                   {
                       _dynamic_array_spikegeneratorgroup_spike_number[i] = _static_array__dynamic_array_spikegeneratorgroup_spike_number[i];
                   }
                   
   _dynamic_array_spikegeneratorgroup_neuron_index.resize(1954);
   
                   for(int i=0; i<_dynamic_array_spikegeneratorgroup_neuron_index.size(); i++)
                   {
                       _dynamic_array_spikegeneratorgroup_neuron_index[i] = _static_array__dynamic_array_spikegeneratorgroup_neuron_index[i];
                   }
                   
   _dynamic_array_spikegeneratorgroup_spike_time.resize(1954);
   
                   for(int i=0; i<_dynamic_array_spikegeneratorgroup_spike_time.size(); i++)
                   {
                       _dynamic_array_spikegeneratorgroup_spike_time[i] = _static_array__dynamic_array_spikegeneratorgroup_spike_time[i];
                   }
                   
   _dynamic_array_spikegeneratorgroup__timebins.resize(1954);
   _array_spikegeneratorgroup__lastindex[0] = 0;
   _array_spikegeneratorgroup_period[0] = 0.0;
   
                   for(int i=0; i<_num__array_neurongroup_lastspike; i++)
                   {
                       _array_neurongroup_lastspike[i] = - 10000.0;
                   }
                   
   
                   for(int i=0; i<_num__array_neurongroup_not_refractory; i++)
                   {
                       _array_neurongroup_not_refractory[i] = true;
                   }
                   
   
                   for(int i=0; i<_num__array_neurongroup_V; i++)
                   {
                       _array_neurongroup_V[i] = - 0.06356;
                   }
                   
   
                   for(int i=0; i<_num__array_neurongroup_h; i++)
                   {
                       _array_neurongroup_h[i] = 1;
                   }
                   
   
                   for(int i=0; i<_num__array_neurongroup_m; i++)
                   {
                       _array_neurongroup_m[i] = 0;
                   }
                   
   
                   for(int i=0; i<_num__array_neurongroup_n; i++)
                   {
                       _array_neurongroup_n[i] = 0.5;
                   }
                   
   
                   for(int i=0; i<_num__array_neurongroup_1_lastspike; i++)
                   {
                       _array_neurongroup_1_lastspike[i] = - 10000.0;
                   }
                   
   
                   for(int i=0; i<_num__array_neurongroup_1_not_refractory; i++)
                   {
                       _array_neurongroup_1_not_refractory[i] = true;
                   }
                   
   
                   for(int i=0; i<_num__array_neurongroup_1_V; i++)
                   {
                       _array_neurongroup_1_V[i] = - 0.06356;
                   }
                   
   
                   for(int i=0; i<_num__array_neurongroup_1_h; i++)
                   {
                       _array_neurongroup_1_h[i] = 1;
                   }
                   
   
                   for(int i=0; i<_num__array_neurongroup_1_m; i++)
                   {
                       _array_neurongroup_1_m[i] = 0;
                   }
                   
   
                   for(int i=0; i<_num__array_neurongroup_1_n; i++)
                   {
                       _array_neurongroup_1_n[i] = 0.5;
                   }
                   
   _run_synapses_synapses_create_generator_codeobject();
   _run_synapses_group_variable_set_conditional_codeobject();
   _run_synapses_1_synapses_create_generator_codeobject();
   _run_synapses_1_group_variable_set_conditional_codeobject();
   _run_synapses_1_group_variable_set_conditional_codeobject_1();
   _run_synapses_2_synapses_create_generator_codeobject();
   _array_defaultclock_timestep[0] = 0;
   _array_defaultclock_t[0] = 0.0;
   _array_spikegeneratorgroup__lastindex[0] = 0;
   
                   for(int i=0; i<_dynamic_array_spikegeneratorgroup__timebins.size(); i++)
                   {
                       _dynamic_array_spikegeneratorgroup__timebins[i] = _static_array__dynamic_array_spikegeneratorgroup__timebins[i];
                   }
                   
   _array_spikegeneratorgroup__period_bins[0] = 0.0;

  }

  // translate to GeNN synaptic arrays
   initialize_sparse_synapses(brian::_dynamic_array_synapses__synaptic_pre, brian::_dynamic_array_synapses__synaptic_post,
                             rowLengthsynapses, indsynapses, maxRowLengthsynapses,
                             100, 2500,
                             sparseSynapseIndicessynapses);
  convert_dynamic_arrays_2_sparse_synapses(brian::_dynamic_array_synapses_weight,
					   sparseSynapseIndicessynapses,
                                           weightsynapses,
                                           100, 2500);
      initialize_sparse_synapses(brian::_dynamic_array_synapses_2__synaptic_pre, brian::_dynamic_array_synapses_2__synaptic_post,
                             rowLengthsynapses_2, indsynapses_2, maxRowLengthsynapses_2,
                             100, 100,
                             sparseSynapseIndicessynapses_2);
      initialize_sparse_synapses(brian::_dynamic_array_synapses_1__synaptic_pre, brian::_dynamic_array_synapses_1__synaptic_post,
                             rowLengthsynapses_1, indsynapses_1, maxRowLengthsynapses_1,
                             2500, 100,
                             sparseSynapseIndicessynapses_1);
  convert_dynamic_arrays_2_sparse_synapses(brian::_dynamic_array_synapses_1_g_raw,
					   sparseSynapseIndicessynapses_1,
                                           g_rawsynapses_1,
                                           2500, 100);
  convert_dynamic_arrays_2_sparse_synapses(brian::_dynamic_array_synapses_1_Apost,
					   sparseSynapseIndicessynapses_1,
                                           Apostsynapses_1,
                                           2500, 100);
  convert_dynamic_arrays_2_sparse_synapses(brian::_dynamic_array_synapses_1_Apre,
					   sparseSynapseIndicessynapses_1,
                                           Apresynapses_1,
                                           2500, 100);
  convert_dynamic_arrays_2_sparse_synapses(brian::_dynamic_array_synapses_1_lastupdate,
					   sparseSynapseIndicessynapses_1,
                                           lastupdatesynapses_1,
                                           2500, 100);
    
  // copy variable arrays
 
  std::copy_n(brian::_array_neurongroup_i, 2500, ineurongroup);
  std::copy_n(brian::_array_neurongroup_V, 2500, Vneurongroup);
  std::copy_n(brian::_array_neurongroup_g_PN_iKC, 2500, g_PN_iKCneurongroup);
  std::copy_n(brian::_array_neurongroup_h, 2500, hneurongroup);
  std::copy_n(brian::_array_neurongroup_m, 2500, mneurongroup);
  std::copy_n(brian::_array_neurongroup_n, 2500, nneurongroup);
  std::copy_n(brian::_array_neurongroup_lastspike, 2500, lastspikeneurongroup);
  std::copy_n(brian::_array_neurongroup_not_refractory, 2500, not_refractoryneurongroup);
 
  std::copy_n(brian::_array_neurongroup_1_i, 100, ineurongroup_1);
  std::copy_n(brian::_array_neurongroup_1_V, 100, Vneurongroup_1);
  std::copy_n(brian::_array_neurongroup_1_g_eKC_eKC, 100, g_eKC_eKCneurongroup_1);
  std::copy_n(brian::_array_neurongroup_1_g_iKC_eKC, 100, g_iKC_eKCneurongroup_1);
  std::copy_n(brian::_array_neurongroup_1_h, 100, hneurongroup_1);
  std::copy_n(brian::_array_neurongroup_1_m, 100, mneurongroup_1);
  std::copy_n(brian::_array_neurongroup_1_n, 100, nneurongroup_1);
  std::copy_n(brian::_array_neurongroup_1_lastspike, 100, lastspikeneurongroup_1);
  std::copy_n(brian::_array_neurongroup_1_not_refractory, 100, not_refractoryneurongroup_1);

  // copy scalar variables
  
  // initialise random seeds (if any are used)
 ;
 ;

  // Perform final stage of initialization, uploading manually initialized variables to GPU etc
  initializeSparse();
  
  //------------------------------------------------------------------
  // output general parameters to output file and start the simulation
  fprintf(stderr, "# We are running with fixed time step %f \n", DT);

  t= 0.;
  void *devPtr;
    

  eng.run(totalTime); // run for the full duration
    

  cerr << t << " done ..." << endl;
 

  // get the final results from the GPU 
  eng.getStateFromGPU();
  eng.getSpikesFromGPU();

  // translate GeNN arrays back to synaptic arrays
  
      convert_sparse_synapses_2_dynamic_arrays(rowLengthsynapses, indsynapses, maxRowLengthsynapses,
                                               weightsynapses, 100, 2500, brian::_dynamic_array_synapses__synaptic_pre, brian::_dynamic_array_synapses__synaptic_post, brian::_dynamic_array_synapses_weight, b2g::FULL_MONTY);
     
     
      convert_sparse_synapses_2_dynamic_arrays(rowLengthsynapses_1, indsynapses_1, maxRowLengthsynapses_1,
                                               g_rawsynapses_1, 2500, 100, brian::_dynamic_array_synapses_1__synaptic_pre, brian::_dynamic_array_synapses_1__synaptic_post, brian::_dynamic_array_synapses_1_g_raw, b2g::FULL_MONTY);
      convert_sparse_synapses_2_dynamic_arrays(rowLengthsynapses_1, indsynapses_1, maxRowLengthsynapses_1,
                                               Apostsynapses_1, 2500, 100, brian::_dynamic_array_synapses_1__synaptic_pre, brian::_dynamic_array_synapses_1__synaptic_post, brian::_dynamic_array_synapses_1_Apost, b2g::FULL_MONTY);
      convert_sparse_synapses_2_dynamic_arrays(rowLengthsynapses_1, indsynapses_1, maxRowLengthsynapses_1,
                                               Apresynapses_1, 2500, 100, brian::_dynamic_array_synapses_1__synaptic_pre, brian::_dynamic_array_synapses_1__synaptic_post, brian::_dynamic_array_synapses_1_Apre, b2g::FULL_MONTY);
      convert_sparse_synapses_2_dynamic_arrays(rowLengthsynapses_1, indsynapses_1, maxRowLengthsynapses_1,
                                               lastupdatesynapses_1, 2500, 100, brian::_dynamic_array_synapses_1__synaptic_pre, brian::_dynamic_array_synapses_1__synaptic_post, brian::_dynamic_array_synapses_1_lastupdate, b2g::FULL_MONTY);
    
  // copy variable arrays
 
  std::copy_n(ineurongroup, 2500, brian::_array_neurongroup_i);
  std::copy_n(Vneurongroup, 2500, brian::_array_neurongroup_V);
  std::copy_n(g_PN_iKCneurongroup, 2500, brian::_array_neurongroup_g_PN_iKC);
  std::copy_n(hneurongroup, 2500, brian::_array_neurongroup_h);
  std::copy_n(mneurongroup, 2500, brian::_array_neurongroup_m);
  std::copy_n(nneurongroup, 2500, brian::_array_neurongroup_n);
  std::copy_n(lastspikeneurongroup, 2500, brian::_array_neurongroup_lastspike);
  std::copy_n(not_refractoryneurongroup, 2500, brian::_array_neurongroup_not_refractory);
 
  std::copy_n(ineurongroup_1, 100, brian::_array_neurongroup_1_i);
  std::copy_n(Vneurongroup_1, 100, brian::_array_neurongroup_1_V);
  std::copy_n(g_eKC_eKCneurongroup_1, 100, brian::_array_neurongroup_1_g_eKC_eKC);
  std::copy_n(g_iKC_eKCneurongroup_1, 100, brian::_array_neurongroup_1_g_iKC_eKC);
  std::copy_n(hneurongroup_1, 100, brian::_array_neurongroup_1_h);
  std::copy_n(mneurongroup_1, 100, brian::_array_neurongroup_1_m);
  std::copy_n(nneurongroup_1, 100, brian::_array_neurongroup_1_n);
  std::copy_n(lastspikeneurongroup_1, 100, brian::_array_neurongroup_1_lastspike);
  std::copy_n(not_refractoryneurongroup_1, 100, brian::_array_neurongroup_1_not_refractory);

  // copy scalar variables

    

  _write_arrays();
  _dealloc_arrays();
    

  cerr << "everything finished." << endl;
  return 0;
}


#ifndef _ENGINE_CC_
#define _ENGINE_CC_

//--------------------------------------------------------------------------
/*! \file engine.cc
\brief Implementation of the engine class.
*/
//--------------------------------------------------------------------------

#include "engine.h"
#include "network.h"

engine::engine()
{
  allocateMem();
  initialize();
  Network::_last_run_time= 0.0;
  Network::_last_run_completed_fraction= 0.0;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

engine::~engine()
{
}


//--------------------------------------------------------------------------
/*! \brief Method for simulating the model for a given period of time
 */
//--------------------------------------------------------------------------

void engine::run(double duration)  //!< Duration of time to run the model for
{
  std::clock_t start, current; 
  const double t_start = t;

  start = std::clock();
  int riT= (int) (duration/DT+1e-2);
  double elapsed_realtime;

  for (int i= 0; i < riT; i++) {
      // The StateMonitor and run_regularly operations are ordered by their "order" value
      stepTime();
      // The stepTimeGPU function already updated everything for the next time step
      iT--;
      t = iT*DT;
      _run_spikegeneratorgroup_codeobject();
      pushspikegeneratorgroupSpikesToDevice();
      pullneurongroup_1CurrentSpikesFromDevice();
      pullneurongroupCurrentSpikesFromDevice();
      // report state 
      // report spikes
      _run_spikemonitor_2_codeobject();
      _run_spikemonitor_1_codeobject();
      _run_spikemonitor_codeobject();
      // Bring the time step back to the value for the next loop iteration
      iT++;
      t = iT*DT;
  }  
  current= std::clock();
  elapsed_realtime= (double) (current - start)/CLOCKS_PER_SEC;
  Network::_last_run_time = elapsed_realtime;
  if (duration > 0.0)
  {
      Network::_last_run_completed_fraction = (t-t_start)/duration;
  } else {
      Network::_last_run_completed_fraction = 1.0;
  }
}

//--------------------------------------------------------------------------
/*! \brief Method for copying all variables of the last time step from the GPU
 
  This is a simple wrapper for the convenience function copyStateFromDevice() which is provided by GeNN.
*/
//--------------------------------------------------------------------------

void engine::getStateFromGPU()
{
  copyStateFromDevice();
}

//--------------------------------------------------------------------------
/*! \brief Method for copying all spikes of the last time step from the GPU
 
  This is a simple wrapper for the convenience function copySpikesFromDevice() which is provided by GeNN.
*/
//--------------------------------------------------------------------------

void engine::getSpikesFromGPU()
{
  copyCurrentSpikesFromDevice();
}



#endif	


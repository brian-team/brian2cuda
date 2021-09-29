
#ifndef ENGINE_H
#define ENGINE_H

//--------------------------------------------------------------------------
/*! \file engine.h

\brief Header file containing the class definition for the engine to conveniently run a model in GeNN
*/
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
/*! \brief This class contains the methods for running the model.
 */
//--------------------------------------------------------------------------

#include <ctime>
#include "magicnetwork_model_CODE/definitions.h"
#include "network.h"

double Network::_last_run_time = 0.0;
double Network::_last_run_completed_fraction = 0.0;

class engine
{
 public:
  // end of data fields 

  engine();
  ~engine();
  void free_device_mem(); 
  void run(double);
  void getStateFromGPU(); 
  void getSpikesFromGPU(); 
};

#endif

// define the time step

#include <stdint.h>
#include "modelSpec.h"
#include "brianlib/randomkit/randomkit.cc"

#include "objects.h"
#include "objects.cpp"
// We need these to compile objects.cpp, but they are only used in _write_arrays which we never call.
double Network::_last_run_time = 0.0;
double Network::_last_run_completed_fraction = 0.0;

#include "code_objects/synapses_group_variable_set_conditional_codeobject.cpp"
#include "code_objects/synapses_1_group_variable_set_conditional_codeobject.cpp"
#include "code_objects/synapses_1_group_variable_set_conditional_codeobject_1.cpp"

#include "code_objects/synapses_max_row_length.cpp"
#include "code_objects/synapses_1_max_row_length.cpp"
#include "code_objects/synapses_2_max_row_length.cpp"

//--------------------------------------------------------------------------
/*! \brief This function defines the Brian2GeNN_model
*/
//--------------------------------------------------------------------------

//
// define the neuron model classes

class neurongroupNEURON : public NeuronModels::Base
{
public:
    DECLARE_MODEL(neurongroupNEURON, 12, 8);

    SET_SIM_CODE("// Update \"constant over DT\" subexpressions (if any)\n\
\n\
\n\
\n\
// PoissonInputs targetting this group (if any)\n\
\n\
\n\
\n\
// Update state variables and the threshold condition\n\
\n\
$(not_refractory) = $(not_refractory) || (! ($(V) > (0 * $(mV))));\n\
double _BA_V = 1.0f*((((1.0f*(((1.0 * $(E_K)) * $(g_K)) * (_brian_pow($(n), 4.0)))/$(C)) + (1.0f*((((1.0 * $(E_Na)) * $(g_Na)) * $(h)) * (_brian_pow($(m), 3.0)))/$(C))) + (1.0f*((1.0 * $(E_e)) * $(g_PN_iKC))/$(C))) + (1.0f*((1.0 * $(E_leak)) * $(g_leak))/$(C)))/((((1.0f*(((- 1.0) * $(g_K)) * (_brian_pow($(n), 4.0)))/$(C)) - (1.0f*(((1.0 * $(g_Na)) * $(h)) * (_brian_pow($(m), 3.0)))/$(C))) - (1.0f*(1.0 * $(g_PN_iKC))/$(C))) - (1.0f*(1.0 * $(g_leak))/$(C)));\n\
double _V = (- _BA_V) + (($(V) + _BA_V) * exp(DT * ((((1.0f*(((- 1.0) * $(g_K)) * (_brian_pow($(n), 4.0)))/$(C)) - (1.0f*(((1.0 * $(g_Na)) * $(h)) * (_brian_pow($(m), 3.0)))/$(C))) - (1.0f*(1.0 * $(g_PN_iKC))/$(C))) - (1.0f*(1.0 * $(g_leak))/$(C)))));\n\
double _g_PN_iKC = $(g_PN_iKC) * exp(1.0f*(- DT)/$(tau_PN_iKC));\n\
double _BA_h = 1.0f*((0.329137207652868 * exp(1.0f*((- 0.0555555555555556) * $(V))/$(mV))) * exp(1.0f*(0.0555555555555556 * $(VT))/$(mV)))/($(ms) * ((1.0f*(- 4.0)/($(ms) + (((2980.95798704173 * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV))))) - (1.0f*((0.329137207652868 * exp(1.0f*((- 0.0555555555555556) * $(V))/$(mV))) * exp(1.0f*(0.0555555555555556 * $(VT))/$(mV)))/$(ms))));\n\
double _h = (- _BA_h) + ((_BA_h + $(h)) * exp(DT * ((1.0f*(- 4.0)/($(ms) + (((2980.95798704173 * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV))))) - (1.0f*((0.329137207652868 * exp(1.0f*((- 0.0555555555555556) * $(V))/$(mV))) * exp(1.0f*(0.0555555555555556 * $(VT))/$(mV)))/$(ms)))));\n\
double _BA_m = 1.0f*(((1.0f*((- 0.32) * $(V))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV))))) + (1.0f*(0.32 * $(VT))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV)))))) + (1.0f*(4.16 * $(mV))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV))))))/((((((1.0f*((- 0.28) * $(V))/(((((0.000335462627902512 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*(0.2 * $(V))/$(mV))) * exp(1.0f*((- 0.2) * $(VT))/$(mV))) - ((1.0 * (_brian_pow($(mV), 1.0))) * $(ms)))) + (1.0f*(0.32 * $(V))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV)))))) + (1.0f*(0.28 * $(VT))/(((((0.000335462627902512 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*(0.2 * $(V))/$(mV))) * exp(1.0f*((- 0.2) * $(VT))/$(mV))) - ((1.0 * (_brian_pow($(mV), 1.0))) * $(ms))))) - (1.0f*(0.32 * $(VT))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV)))))) + (1.0f*(11.2 * $(mV))/(((((0.000335462627902512 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*(0.2 * $(V))/$(mV))) * exp(1.0f*((- 0.2) * $(VT))/$(mV))) - ((1.0 * (_brian_pow($(mV), 1.0))) * $(ms))))) - (1.0f*(4.16 * $(mV))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV))))));\n\
double _m = (- _BA_m) + ((_BA_m + $(m)) * exp(DT * ((((((1.0f*((- 0.28) * $(V))/(((((0.000335462627902512 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*(0.2 * $(V))/$(mV))) * exp(1.0f*((- 0.2) * $(VT))/$(mV))) - ((1.0 * (_brian_pow($(mV), 1.0))) * $(ms)))) + (1.0f*(0.32 * $(V))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV)))))) + (1.0f*(0.28 * $(VT))/(((((0.000335462627902512 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*(0.2 * $(V))/$(mV))) * exp(1.0f*((- 0.2) * $(VT))/$(mV))) - ((1.0 * (_brian_pow($(mV), 1.0))) * $(ms))))) - (1.0f*(0.32 * $(VT))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV)))))) + (1.0f*(11.2 * $(mV))/(((((0.000335462627902512 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*(0.2 * $(V))/$(mV))) * exp(1.0f*((- 0.2) * $(VT))/$(mV))) - ((1.0 * (_brian_pow($(mV), 1.0))) * $(ms))))) - (1.0f*(4.16 * $(mV))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV))))))));\n\
double _BA_n = 1.0f*(((1.0f*((- 0.032) * $(V))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV))))) + (1.0f*(0.032 * $(VT))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV)))))) + (1.0f*(0.48 * $(mV))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV))))))/((((1.0f*(0.032 * $(V))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV))))) - (1.0f*(0.032 * $(VT))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV)))))) - (1.0f*(0.48 * $(mV))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV)))))) - (1.0f*((0.642012708343871 * exp(1.0f*((- 0.025) * $(V))/$(mV))) * exp(1.0f*(0.025 * $(VT))/$(mV)))/$(ms)));\n\
double _n = (- _BA_n) + ((_BA_n + $(n)) * exp(DT * ((((1.0f*(0.032 * $(V))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV))))) - (1.0f*(0.032 * $(VT))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV)))))) - (1.0f*(0.48 * $(mV))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV)))))) - (1.0f*((0.642012708343871 * exp(1.0f*((- 0.025) * $(V))/$(mV))) * exp(1.0f*(0.025 * $(VT))/$(mV)))/$(ms)))));\n\
$(V) = _V;\n\
$(g_PN_iKC) = _g_PN_iKC;\n\
$(h) = _h;\n\
$(m) = _m;\n\
$(n) = _n;\n\
char _cond = ($(V) > (0 * $(mV))) && $(not_refractory);");
    SET_THRESHOLD_CONDITION_CODE("_cond");
    SET_RESET_CODE("$(lastspike) = t;\n\
$(not_refractory) = false;");

    SET_SUPPORT_CODE("\n\
template < typename T1, typename T2 > struct _higher_type;\n\
template < > struct _higher_type<int,int> { typedef int type; };\n\
template < > struct _higher_type<int,long> { typedef long type; };\n\
template < > struct _higher_type<int,float> { typedef float type; };\n\
template < > struct _higher_type<int,double> { typedef double type; };\n\
template < > struct _higher_type<long,int> { typedef long type; };\n\
template < > struct _higher_type<long,long> { typedef long type; };\n\
template < > struct _higher_type<long,float> { typedef float type; };\n\
template < > struct _higher_type<long,double> { typedef double type; };\n\
template < > struct _higher_type<float,int> { typedef float type; };\n\
template < > struct _higher_type<float,long> { typedef float type; };\n\
template < > struct _higher_type<float,float> { typedef float type; };\n\
template < > struct _higher_type<float,double> { typedef double type; };\n\
template < > struct _higher_type<double,int> { typedef double type; };\n\
template < > struct _higher_type<double,long> { typedef double type; };\n\
template < > struct _higher_type<double,float> { typedef double type; };\n\
template < > struct _higher_type<double,double> { typedef double type; };\n\
template < typename T1, typename T2 >\n\
SUPPORT_CODE_FUNC typename _higher_type<T1,T2>::type \n\
_brian_mod(T1 x, T2 y)\n\
{{\n\
    return x-y*floor(1.0*x/y);\n\
}}\n\
template < typename T1, typename T2 >\n\
SUPPORT_CODE_FUNC typename _higher_type<T1,T2>::type\n\
_brian_floordiv(T1 x, T2 y)\n\
{{\n\
    return floor(1.0*x/y);\n\
}}\n\
#ifdef _MSC_VER\n\
#define _brian_pow(x, y) (pow((double)(x), (y)))\n\
#else\n\
#define _brian_pow(x, y) (pow((x), (y)))\n\
#endif\n\
\n\
\n\
\n\
");

    SET_PARAM_NAMES({
        "VT",        "E_K",        "E_e",        "ms",        "g_Na",        "g_leak",        "E_leak",        "tau_PN_iKC",        "C",        "E_Na",        "g_K",        "mV"    });
    SET_VARS({
        {"i", "int32_t"},        {"V", "double"},        {"g_PN_iKC", "double"},        {"h", "double"},        {"m", "double"},        {"n", "double"},        {"lastspike", "double"},        {"not_refractory", "char"}    });
    SET_EXTRA_GLOBAL_PARAMS({
    });
    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(neurongroupNEURON);
class neurongroup_1NEURON : public NeuronModels::Base
{
public:
    DECLARE_MODEL(neurongroup_1NEURON, 14, 9);

    SET_SIM_CODE("// Update \"constant over DT\" subexpressions (if any)\n\
\n\
\n\
\n\
// PoissonInputs targetting this group (if any)\n\
\n\
\n\
\n\
// Update state variables and the threshold condition\n\
\n\
$(not_refractory) = $(not_refractory) || (! ($(V) > (0.0 * $(mV))));\n\
double _BA_V = 1.0f*(((((1.0f*(((1.0 * $(E_K)) * $(g_K)) * (_brian_pow($(n), 4.0)))/$(C)) + (1.0f*((((1.0 * $(E_Na)) * $(g_Na)) * $(h)) * (_brian_pow($(m), 3.0)))/$(C))) + (1.0f*((1.0 * $(E_e)) * $(g_iKC_eKC))/$(C))) + (1.0f*((1.0 * $(E_i)) * $(g_eKC_eKC))/$(C))) + (1.0f*((1.0 * $(E_leak)) * $(g_leak))/$(C)))/(((((1.0f*(((- 1.0) * $(g_K)) * (_brian_pow($(n), 4.0)))/$(C)) - (1.0f*(((1.0 * $(g_Na)) * $(h)) * (_brian_pow($(m), 3.0)))/$(C))) - (1.0f*(1.0 * $(g_eKC_eKC))/$(C))) - (1.0f*(1.0 * $(g_iKC_eKC))/$(C))) - (1.0f*(1.0 * $(g_leak))/$(C)));\n\
double _V = (- _BA_V) + (($(V) + _BA_V) * exp(DT * (((((1.0f*(((- 1.0) * $(g_K)) * (_brian_pow($(n), 4.0)))/$(C)) - (1.0f*(((1.0 * $(g_Na)) * $(h)) * (_brian_pow($(m), 3.0)))/$(C))) - (1.0f*(1.0 * $(g_eKC_eKC))/$(C))) - (1.0f*(1.0 * $(g_iKC_eKC))/$(C))) - (1.0f*(1.0 * $(g_leak))/$(C)))));\n\
double _g_eKC_eKC = $(g_eKC_eKC) * exp(1.0f*(- DT)/$(tau_eKC_eKC));\n\
double _g_iKC_eKC = $(g_iKC_eKC) * exp(1.0f*(- DT)/$(tau_iKC_eKC));\n\
double _BA_h = 1.0f*((0.329137207652868 * exp(1.0f*((- 0.0555555555555556) * $(V))/$(mV))) * exp(1.0f*(0.0555555555555556 * $(VT))/$(mV)))/($(ms) * ((1.0f*(- 4.0)/($(ms) + (((2980.95798704173 * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV))))) - (1.0f*((0.329137207652868 * exp(1.0f*((- 0.0555555555555556) * $(V))/$(mV))) * exp(1.0f*(0.0555555555555556 * $(VT))/$(mV)))/$(ms))));\n\
double _h = (- _BA_h) + ((_BA_h + $(h)) * exp(DT * ((1.0f*(- 4.0)/($(ms) + (((2980.95798704173 * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV))))) - (1.0f*((0.329137207652868 * exp(1.0f*((- 0.0555555555555556) * $(V))/$(mV))) * exp(1.0f*(0.0555555555555556 * $(VT))/$(mV)))/$(ms)))));\n\
double _BA_m = 1.0f*(((1.0f*((- 0.32) * $(V))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV))))) + (1.0f*(0.32 * $(VT))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV)))))) + (1.0f*(4.16 * $(mV))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV))))))/((((((1.0f*((- 0.28) * $(V))/(((((0.000335462627902512 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*(0.2 * $(V))/$(mV))) * exp(1.0f*((- 0.2) * $(VT))/$(mV))) - ((1.0 * (_brian_pow($(mV), 1.0))) * $(ms)))) + (1.0f*(0.32 * $(V))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV)))))) + (1.0f*(0.28 * $(VT))/(((((0.000335462627902512 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*(0.2 * $(V))/$(mV))) * exp(1.0f*((- 0.2) * $(VT))/$(mV))) - ((1.0 * (_brian_pow($(mV), 1.0))) * $(ms))))) - (1.0f*(0.32 * $(VT))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV)))))) + (1.0f*(11.2 * $(mV))/(((((0.000335462627902512 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*(0.2 * $(V))/$(mV))) * exp(1.0f*((- 0.2) * $(VT))/$(mV))) - ((1.0 * (_brian_pow($(mV), 1.0))) * $(ms))))) - (1.0f*(4.16 * $(mV))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV))))));\n\
double _m = (- _BA_m) + ((_BA_m + $(m)) * exp(DT * ((((((1.0f*((- 0.28) * $(V))/(((((0.000335462627902512 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*(0.2 * $(V))/$(mV))) * exp(1.0f*((- 0.2) * $(VT))/$(mV))) - ((1.0 * (_brian_pow($(mV), 1.0))) * $(ms)))) + (1.0f*(0.32 * $(V))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV)))))) + (1.0f*(0.28 * $(VT))/(((((0.000335462627902512 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*(0.2 * $(V))/$(mV))) * exp(1.0f*((- 0.2) * $(VT))/$(mV))) - ((1.0 * (_brian_pow($(mV), 1.0))) * $(ms))))) - (1.0f*(0.32 * $(VT))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV)))))) + (1.0f*(11.2 * $(mV))/(((((0.000335462627902512 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*(0.2 * $(V))/$(mV))) * exp(1.0f*((- 0.2) * $(VT))/$(mV))) - ((1.0 * (_brian_pow($(mV), 1.0))) * $(ms))))) - (1.0f*(4.16 * $(mV))/((((- 1.0) * (_brian_pow($(mV), 1.0))) * $(ms)) + ((((25.7903399171931 * (_brian_pow($(mV), 1.0))) * $(ms)) * exp(1.0f*((- 0.25) * $(V))/$(mV))) * exp(1.0f*(0.25 * $(VT))/$(mV))))))));\n\
double _BA_n = 1.0f*(((1.0f*((- 0.032) * $(V))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV))))) + (1.0f*(0.032 * $(VT))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV)))))) + (1.0f*(0.48 * $(mV))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV))))))/((((1.0f*(0.032 * $(V))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV))))) - (1.0f*(0.032 * $(VT))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV)))))) - (1.0f*(0.48 * $(mV))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV)))))) - (1.0f*((0.642012708343871 * exp(1.0f*((- 0.025) * $(V))/$(mV))) * exp(1.0f*(0.025 * $(VT))/$(mV)))/$(ms)));\n\
double _n = (- _BA_n) + ((_BA_n + $(n)) * exp(DT * ((((1.0f*(0.032 * $(V))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV))))) - (1.0f*(0.032 * $(VT))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV)))))) - (1.0f*(0.48 * $(mV))/((((- 1.0) * $(mV)) * $(ms)) + ((((20.0855369231877 * $(mV)) * $(ms)) * exp(1.0f*((- 0.2) * $(V))/$(mV))) * exp(1.0f*(0.2 * $(VT))/$(mV)))))) - (1.0f*((0.642012708343871 * exp(1.0f*((- 0.025) * $(V))/$(mV))) * exp(1.0f*(0.025 * $(VT))/$(mV)))/$(ms)))));\n\
$(V) = _V;\n\
$(g_eKC_eKC) = _g_eKC_eKC;\n\
$(g_iKC_eKC) = _g_iKC_eKC;\n\
$(h) = _h;\n\
$(m) = _m;\n\
$(n) = _n;\n\
char _cond = ($(V) > (0.0 * $(mV))) && $(not_refractory);");
    SET_THRESHOLD_CONDITION_CODE("_cond");
    SET_RESET_CODE("$(lastspike) = t;\n\
$(not_refractory) = false;");

    SET_SUPPORT_CODE("\n\
template < typename T1, typename T2 > struct _higher_type;\n\
template < > struct _higher_type<int,int> { typedef int type; };\n\
template < > struct _higher_type<int,long> { typedef long type; };\n\
template < > struct _higher_type<int,float> { typedef float type; };\n\
template < > struct _higher_type<int,double> { typedef double type; };\n\
template < > struct _higher_type<long,int> { typedef long type; };\n\
template < > struct _higher_type<long,long> { typedef long type; };\n\
template < > struct _higher_type<long,float> { typedef float type; };\n\
template < > struct _higher_type<long,double> { typedef double type; };\n\
template < > struct _higher_type<float,int> { typedef float type; };\n\
template < > struct _higher_type<float,long> { typedef float type; };\n\
template < > struct _higher_type<float,float> { typedef float type; };\n\
template < > struct _higher_type<float,double> { typedef double type; };\n\
template < > struct _higher_type<double,int> { typedef double type; };\n\
template < > struct _higher_type<double,long> { typedef double type; };\n\
template < > struct _higher_type<double,float> { typedef double type; };\n\
template < > struct _higher_type<double,double> { typedef double type; };\n\
template < typename T1, typename T2 >\n\
SUPPORT_CODE_FUNC typename _higher_type<T1,T2>::type \n\
_brian_mod(T1 x, T2 y)\n\
{{\n\
    return x-y*floor(1.0*x/y);\n\
}}\n\
template < typename T1, typename T2 >\n\
SUPPORT_CODE_FUNC typename _higher_type<T1,T2>::type\n\
_brian_floordiv(T1 x, T2 y)\n\
{{\n\
    return floor(1.0*x/y);\n\
}}\n\
#ifdef _MSC_VER\n\
#define _brian_pow(x, y) (pow((double)(x), (y)))\n\
#else\n\
#define _brian_pow(x, y) (pow((x), (y)))\n\
#endif\n\
\n\
\n\
\n\
");

    SET_PARAM_NAMES({
        "E_i",        "VT",        "E_K",        "E_e",        "ms",        "g_Na",        "tau_eKC_eKC",        "g_leak",        "E_leak",        "C",        "E_Na",        "tau_iKC_eKC",        "g_K",        "mV"    });
    SET_VARS({
        {"i", "int32_t"},        {"V", "double"},        {"g_eKC_eKC", "double"},        {"g_iKC_eKC", "double"},        {"h", "double"},        {"m", "double"},        {"n", "double"},        {"lastspike", "double"},        {"not_refractory", "char"}    });
    SET_EXTRA_GLOBAL_PARAMS({
    });
    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(neurongroup_1NEURON);

//
// define the synapse model classes
class synapsesWEIGHTUPDATE : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(synapsesWEIGHTUPDATE, 1, 1);

    SET_SIM_CODE("$(addToInSyn,($(scale) * $(weight)));");
    SET_LEARN_POST_CODE("");
    SET_SYNAPSE_DYNAMICS_CODE("");

    SET_SIM_SUPPORT_CODE("\n\
template < typename T1, typename T2 > struct _higher_type;\n\
template < > struct _higher_type<int,int> { typedef int type; };\n\
template < > struct _higher_type<int,long> { typedef long type; };\n\
template < > struct _higher_type<int,float> { typedef float type; };\n\
template < > struct _higher_type<int,double> { typedef double type; };\n\
template < > struct _higher_type<long,int> { typedef long type; };\n\
template < > struct _higher_type<long,long> { typedef long type; };\n\
template < > struct _higher_type<long,float> { typedef float type; };\n\
template < > struct _higher_type<long,double> { typedef double type; };\n\
template < > struct _higher_type<float,int> { typedef float type; };\n\
template < > struct _higher_type<float,long> { typedef float type; };\n\
template < > struct _higher_type<float,float> { typedef float type; };\n\
template < > struct _higher_type<float,double> { typedef double type; };\n\
template < > struct _higher_type<double,int> { typedef double type; };\n\
template < > struct _higher_type<double,long> { typedef double type; };\n\
template < > struct _higher_type<double,float> { typedef double type; };\n\
template < > struct _higher_type<double,double> { typedef double type; };\n\
template < typename T1, typename T2 >\n\
SUPPORT_CODE_FUNC typename _higher_type<T1,T2>::type \n\
_brian_mod(T1 x, T2 y)\n\
{{\n\
    return x-y*floor(1.0*x/y);\n\
}}\n\
template < typename T1, typename T2 >\n\
SUPPORT_CODE_FUNC typename _higher_type<T1,T2>::type\n\
_brian_floordiv(T1 x, T2 y)\n\
{{\n\
    return floor(1.0*x/y);\n\
}}\n\
#ifdef _MSC_VER\n\
#define _brian_pow(x, y) (pow((double)(x), (y)))\n\
#else\n\
#define _brian_pow(x, y) (pow((x), (y)))\n\
#endif\n\
\n\
\n\
\n\
");
    SET_LEARN_POST_SUPPORT_CODE("");
    SET_SYNAPSE_DYNAMICS_SUPPORT_CODE("");

    SET_PARAM_NAMES({
        "scale"    });

    SET_VARS({
        {"weight", "double"}    });

    SET_EXTRA_GLOBAL_PARAMS({
    });

    //SET_NEEDS_PRE_SPIKE_TIME(true);
    //SET_NEEDS_POST_SPIKE_TIME(true);

};

IMPLEMENT_MODEL(synapsesWEIGHTUPDATE);

class synapsesPOSTSYN : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(synapsesPOSTSYN, 0, 0);

    SET_APPLY_INPUT_CODE("$(Isyn) += 0; $(g_PN_iKC) += $(inSyn); $(inSyn)= 0;");
};
IMPLEMENT_MODEL(synapsesPOSTSYN);
class synapses_2WEIGHTUPDATE : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(synapses_2WEIGHTUPDATE, 2, 0);

    SET_SIM_CODE("$(addToInSyn,($(scale) * $(w_eKC_eKC)));");
    SET_LEARN_POST_CODE("");
    SET_SYNAPSE_DYNAMICS_CODE("");

    SET_SIM_SUPPORT_CODE("\n\
template < typename T1, typename T2 > struct _higher_type;\n\
template < > struct _higher_type<int,int> { typedef int type; };\n\
template < > struct _higher_type<int,long> { typedef long type; };\n\
template < > struct _higher_type<int,float> { typedef float type; };\n\
template < > struct _higher_type<int,double> { typedef double type; };\n\
template < > struct _higher_type<long,int> { typedef long type; };\n\
template < > struct _higher_type<long,long> { typedef long type; };\n\
template < > struct _higher_type<long,float> { typedef float type; };\n\
template < > struct _higher_type<long,double> { typedef double type; };\n\
template < > struct _higher_type<float,int> { typedef float type; };\n\
template < > struct _higher_type<float,long> { typedef float type; };\n\
template < > struct _higher_type<float,float> { typedef float type; };\n\
template < > struct _higher_type<float,double> { typedef double type; };\n\
template < > struct _higher_type<double,int> { typedef double type; };\n\
template < > struct _higher_type<double,long> { typedef double type; };\n\
template < > struct _higher_type<double,float> { typedef double type; };\n\
template < > struct _higher_type<double,double> { typedef double type; };\n\
template < typename T1, typename T2 >\n\
SUPPORT_CODE_FUNC typename _higher_type<T1,T2>::type \n\
_brian_mod(T1 x, T2 y)\n\
{{\n\
    return x-y*floor(1.0*x/y);\n\
}}\n\
template < typename T1, typename T2 >\n\
SUPPORT_CODE_FUNC typename _higher_type<T1,T2>::type\n\
_brian_floordiv(T1 x, T2 y)\n\
{{\n\
    return floor(1.0*x/y);\n\
}}\n\
#ifdef _MSC_VER\n\
#define _brian_pow(x, y) (pow((double)(x), (y)))\n\
#else\n\
#define _brian_pow(x, y) (pow((x), (y)))\n\
#endif\n\
\n\
\n\
\n\
");
    SET_LEARN_POST_SUPPORT_CODE("");
    SET_SYNAPSE_DYNAMICS_SUPPORT_CODE("");

    SET_PARAM_NAMES({
        "w_eKC_eKC",        "scale"    });

    SET_VARS({
    });

    SET_EXTRA_GLOBAL_PARAMS({
    });

    //SET_NEEDS_PRE_SPIKE_TIME(true);
    //SET_NEEDS_POST_SPIKE_TIME(true);

};

IMPLEMENT_MODEL(synapses_2WEIGHTUPDATE);

class synapses_2POSTSYN : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(synapses_2POSTSYN, 0, 0);

    SET_APPLY_INPUT_CODE("$(Isyn) += 0; $(g_eKC_eKC) += $(inSyn); $(inSyn)= 0;");
};
IMPLEMENT_MODEL(synapses_2POSTSYN);
class synapses_1WEIGHTUPDATE : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(synapses_1WEIGHTUPDATE, 6, 4);

    SET_SIM_CODE("double _Apost = $(Apost) * exp(1.0f*(- (t - $(lastupdate)))/$(tau_post));\n\
double _Apre = $(Apre) * exp(1.0f*(- (t - $(lastupdate)))/$(tau_pre));\n\
$(Apost) = _Apost;\n\
$(Apre) = _Apre;\n\
$(addToInSyn,$(g_raw));\n\
$(Apre) += $(dApre);\n\
$(g_raw) = _clip($(g_raw) + $(Apost), 0 * $(siemens), $(g_max));\n\
$(lastupdate) = t;");
    SET_LEARN_POST_CODE("double _Apost = $(Apost) * exp(1.0f*(- (t - $(lastupdate)))/$(tau_post));\n\
double _Apre = $(Apre) * exp(1.0f*(- (t - $(lastupdate)))/$(tau_pre));\n\
$(Apost) = _Apost;\n\
$(Apre) = _Apre;\n\
$(Apost) += $(dApost);\n\
$(g_raw) = _clip($(g_raw) + $(Apre), 0 * $(siemens), $(g_max));\n\
$(lastupdate) = t;");
    SET_SYNAPSE_DYNAMICS_CODE("");

    SET_SIM_SUPPORT_CODE("\n\
SUPPORT_CODE_FUNC double _clip(const float value, const float a_min, const float a_max)\n\
{\n\
    if (value < a_min)\n\
        return a_min;\n\
    if (value > a_max)\n\
        return a_max;\n\
    return value;\n\
}\n\
template < typename T1, typename T2 > struct _higher_type;\n\
template < > struct _higher_type<int,int> { typedef int type; };\n\
template < > struct _higher_type<int,long> { typedef long type; };\n\
template < > struct _higher_type<int,float> { typedef float type; };\n\
template < > struct _higher_type<int,double> { typedef double type; };\n\
template < > struct _higher_type<long,int> { typedef long type; };\n\
template < > struct _higher_type<long,long> { typedef long type; };\n\
template < > struct _higher_type<long,float> { typedef float type; };\n\
template < > struct _higher_type<long,double> { typedef double type; };\n\
template < > struct _higher_type<float,int> { typedef float type; };\n\
template < > struct _higher_type<float,long> { typedef float type; };\n\
template < > struct _higher_type<float,float> { typedef float type; };\n\
template < > struct _higher_type<float,double> { typedef double type; };\n\
template < > struct _higher_type<double,int> { typedef double type; };\n\
template < > struct _higher_type<double,long> { typedef double type; };\n\
template < > struct _higher_type<double,float> { typedef double type; };\n\
template < > struct _higher_type<double,double> { typedef double type; };\n\
template < typename T1, typename T2 >\n\
SUPPORT_CODE_FUNC typename _higher_type<T1,T2>::type \n\
_brian_mod(T1 x, T2 y)\n\
{{\n\
    return x-y*floor(1.0*x/y);\n\
}}\n\
template < typename T1, typename T2 >\n\
SUPPORT_CODE_FUNC typename _higher_type<T1,T2>::type\n\
_brian_floordiv(T1 x, T2 y)\n\
{{\n\
    return floor(1.0*x/y);\n\
}}\n\
#ifdef _MSC_VER\n\
#define _brian_pow(x, y) (pow((double)(x), (y)))\n\
#else\n\
#define _brian_pow(x, y) (pow((x), (y)))\n\
#endif\n\
\n\
\n\
\n\
");
    SET_LEARN_POST_SUPPORT_CODE("\n\
SUPPORT_CODE_FUNC double _clip(const float value, const float a_min, const float a_max)\n\
{\n\
    if (value < a_min)\n\
        return a_min;\n\
    if (value > a_max)\n\
        return a_max;\n\
    return value;\n\
}\n\
template < typename T1, typename T2 > struct _higher_type;\n\
template < > struct _higher_type<int,int> { typedef int type; };\n\
template < > struct _higher_type<int,long> { typedef long type; };\n\
template < > struct _higher_type<int,float> { typedef float type; };\n\
template < > struct _higher_type<int,double> { typedef double type; };\n\
template < > struct _higher_type<long,int> { typedef long type; };\n\
template < > struct _higher_type<long,long> { typedef long type; };\n\
template < > struct _higher_type<long,float> { typedef float type; };\n\
template < > struct _higher_type<long,double> { typedef double type; };\n\
template < > struct _higher_type<float,int> { typedef float type; };\n\
template < > struct _higher_type<float,long> { typedef float type; };\n\
template < > struct _higher_type<float,float> { typedef float type; };\n\
template < > struct _higher_type<float,double> { typedef double type; };\n\
template < > struct _higher_type<double,int> { typedef double type; };\n\
template < > struct _higher_type<double,long> { typedef double type; };\n\
template < > struct _higher_type<double,float> { typedef double type; };\n\
template < > struct _higher_type<double,double> { typedef double type; };\n\
template < typename T1, typename T2 >\n\
SUPPORT_CODE_FUNC typename _higher_type<T1,T2>::type \n\
_brian_mod(T1 x, T2 y)\n\
{{\n\
    return x-y*floor(1.0*x/y);\n\
}}\n\
template < typename T1, typename T2 >\n\
SUPPORT_CODE_FUNC typename _higher_type<T1,T2>::type\n\
_brian_floordiv(T1 x, T2 y)\n\
{{\n\
    return floor(1.0*x/y);\n\
}}\n\
#ifdef _MSC_VER\n\
#define _brian_pow(x, y) (pow((double)(x), (y)))\n\
#else\n\
#define _brian_pow(x, y) (pow((x), (y)))\n\
#endif\n\
\n\
\n\
\n\
");
    SET_SYNAPSE_DYNAMICS_SUPPORT_CODE("");

    SET_PARAM_NAMES({
        "tau_pre",        "dApre",        "g_max",        "siemens",        "tau_post",        "dApost"    });

    SET_VARS({
        {"lastupdate", "double"},        {"Apost", "double"},        {"g_raw", "double"},        {"Apre", "double"}    });

    SET_EXTRA_GLOBAL_PARAMS({
    });

    //SET_NEEDS_PRE_SPIKE_TIME(true);
    //SET_NEEDS_POST_SPIKE_TIME(true);

};

IMPLEMENT_MODEL(synapses_1WEIGHTUPDATE);

class synapses_1POSTSYN : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(synapses_1POSTSYN, 0, 0);

    SET_APPLY_INPUT_CODE("$(Isyn) += 0; $(g_iKC_eKC) += $(inSyn); $(inSyn)= 0;");
};
IMPLEMENT_MODEL(synapses_1POSTSYN);

// parameter values
// neurons
neurongroupNEURON::ParamValues neurongroup_p
(
    - 0.063,    - 0.095,    0.0,    0.001,    7.15e-06,    2.67e-08,    - 0.06356,    0.002,    3e-10,    0.05,    1.4299999999999999e-06,    0.001);
neurongroup_1NEURON::ParamValues neurongroup_1_p
(
    - 0.092,    - 0.063,    - 0.095,    0.0,    0.001,    7.15e-06,    0.005,    2.67e-08,    - 0.06356,    3e-10,    0.05,    0.01,    1.4299999999999999e-06,    0.001);

// synapses
synapsesWEIGHTUPDATE::ParamValues synapses_p
(
    0.675);
synapses_2WEIGHTUPDATE::ParamValues synapses_2_p
(
    7.500000000000001e-08,    0.675);
synapses_1WEIGHTUPDATE::ParamValues synapses_1_p
(
    0.01,    1.0000000000000002e-10,    3.7500000000000005e-09,    1.0,    0.01,    - 1.0000000000000002e-10);

// initial variables (neurons)
neurongroupNEURON::VarValues neurongroup_ini
(
    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar());
neurongroup_1NEURON::VarValues neurongroup_1_ini
(
    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar());
 
// initial variables (synapses)
// one additional initial variable for hidden_weightmatrix
synapsesWEIGHTUPDATE::VarValues synapses_ini
(
    uninitialisedVar());
synapses_2WEIGHTUPDATE::VarValues synapses_2_ini
;
synapses_1WEIGHTUPDATE::VarValues synapses_1_ini
(
    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar(),    uninitialisedVar());


void modelDefinition(NNmodel &model)
{
  _init_arrays();
  _load_arrays();
    

  rk_randomseed(brian::_mersenne_twister_states[0]);
    

  {
	  using namespace brian;
   	  
   _array_defaultclock_dt[0] = 0.0001;
   _array_defaultclock_dt[0] = 0.0001;
   _array_defaultclock_dt[0] = 0.0001;
   _dynamic_array_spikegeneratorgroup_spike_number.resize(19676);
   
                   for(int i=0; i<_dynamic_array_spikegeneratorgroup_spike_number.size(); i++)
                   {
                       _dynamic_array_spikegeneratorgroup_spike_number[i] = _static_array__dynamic_array_spikegeneratorgroup_spike_number[i];
                   }
                   
   _dynamic_array_spikegeneratorgroup_neuron_index.resize(19676);
   
                   for(int i=0; i<_dynamic_array_spikegeneratorgroup_neuron_index.size(); i++)
                   {
                       _dynamic_array_spikegeneratorgroup_neuron_index[i] = _static_array__dynamic_array_spikegeneratorgroup_neuron_index[i];
                   }
                   
   _dynamic_array_spikegeneratorgroup_spike_time.resize(19676);
   
                   for(int i=0; i<_dynamic_array_spikegeneratorgroup_spike_time.size(); i++)
                   {
                       _dynamic_array_spikegeneratorgroup_spike_time[i] = _static_array__dynamic_array_spikegeneratorgroup_spike_time[i];
                   }
                   
   _dynamic_array_spikegeneratorgroup__timebins.resize(19676);
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
                   
   
                   for(int i=0; i<_num__array_neurongroup_1_lastspike; i++)
                   {
                       _array_neurongroup_1_lastspike[i] = - 10000.0;
                   }
                   
   
                   for(int i=0; i<_num__array_neurongroup_1_not_refractory; i++)
                   {
                       _array_neurongroup_1_not_refractory[i] = true;
                   }
                   
   _dynamic_array_synapses_1_delay.resize(1);
   _dynamic_array_synapses_1_delay.resize(1);
   _dynamic_array_synapses_1_delay[0] = 0.0;
   _dynamic_array_synapses_2_delay.resize(1);
   _dynamic_array_synapses_2_delay.resize(1);
   _dynamic_array_synapses_2_delay[0] = 0.0;
   _run_synapses_group_variable_set_conditional_codeobject();
   _run_synapses_1_group_variable_set_conditional_codeobject();
   _run_synapses_1_group_variable_set_conditional_codeobject_1();
   
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
                   
   _array_defaultclock_timestep[0] = 0;
   _array_defaultclock_t[0] = 0.0;
   _array_spikegeneratorgroup__lastindex[0] = 0;
   
                   for(int i=0; i<_dynamic_array_spikegeneratorgroup__timebins.size(); i++)
                   {
                       _dynamic_array_spikegeneratorgroup__timebins[i] = _static_array__dynamic_array_spikegeneratorgroup__timebins[i];
                   }
                   
   _array_spikegeneratorgroup__period_bins[0] = 0.0;

  }

  _run_synapses_max_row_length();
  _run_synapses_1_max_row_length();
  _run_synapses_2_max_row_length();

  const long maxRowsynapses= std::max(*std::max_element(brian::_dynamic_array_synapses_N_outgoing.begin(),brian::_dynamic_array_synapses_N_outgoing.end()),1);
  const long maxColsynapses= std::max(*std::max_element(brian::_dynamic_array_synapses_N_incoming.begin(),brian::_dynamic_array_synapses_N_incoming.end()),1);
  const long maxRowsynapses_1= std::max(*std::max_element(brian::_dynamic_array_synapses_1_N_outgoing.begin(),brian::_dynamic_array_synapses_1_N_outgoing.end()),1);
  const long maxColsynapses_1= std::max(*std::max_element(brian::_dynamic_array_synapses_1_N_incoming.begin(),brian::_dynamic_array_synapses_1_N_incoming.end()),1);
  const long maxRowsynapses_2= std::max(*std::max_element(brian::_dynamic_array_synapses_2_N_outgoing.begin(),brian::_dynamic_array_synapses_2_N_outgoing.end()),1);
  const long maxColsynapses_2= std::max(*std::max_element(brian::_dynamic_array_synapses_2_N_incoming.begin(),brian::_dynamic_array_synapses_2_N_incoming.end()),1);
  
    // GENN_PREFERENCES set in brian2genn
    GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::OPTIMAL;
    GENN_PREFERENCES.blockSizeSelectMethod = BlockSizeSelect::OCCUPANCY;

    GENN_PREFERENCES.userNvccFlags = "";

    model.setDT(0.0001);

    model.setName("magicnetwork_model");
    model.setPrecision(GENN_DOUBLE);
    model.addNeuronPopulation<neurongroupNEURON>("neurongroup", 2500, neurongroup_p, neurongroup_ini);
    model.addNeuronPopulation<neurongroup_1NEURON>("neurongroup_1", 100, neurongroup_1_p, neurongroup_1_ini);
    model.addNeuronPopulation<NeuronModels::SpikeSource>("spikegeneratorgroup", 100, {}, {});
    {
    // TODO: Consider flexible use of DENSE and SPARSE (but beware of difficulty of judging which to use at compile time)
    const unsigned int delaySteps = NO_DELAY;
    auto *syn = model.addSynapsePopulation<synapsesWEIGHTUPDATE, synapsesPOSTSYN>(
        "synapses", SynapseMatrixType::SPARSE_INDIVIDUALG, delaySteps,
        "spikegeneratorgroup", "neurongroup",
        synapses_p, synapses_ini,
        {}, {});
    syn->setSpanType(SynapseGroup::SpanType::POSTSYNAPTIC);
    syn->setMaxConnections(maxRowsynapses);
    syn->setMaxSourceConnections(maxColsynapses);
    }
    {
    // TODO: Consider flexible use of DENSE and SPARSE (but beware of difficulty of judging which to use at compile time)
    const unsigned int delaySteps = NO_DELAY;
    auto *syn = model.addSynapsePopulation<synapses_2WEIGHTUPDATE, synapses_2POSTSYN>(
        "synapses_2", SynapseMatrixType::SPARSE_INDIVIDUALG, delaySteps,
        "neurongroup_1", "neurongroup_1",
        synapses_2_p, synapses_2_ini,
        {}, {});
    syn->setSpanType(SynapseGroup::SpanType::POSTSYNAPTIC);
    syn->setMaxConnections(maxRowsynapses_2);
    syn->setMaxSourceConnections(maxColsynapses_2);
    }
    {
    // TODO: Consider flexible use of DENSE and SPARSE (but beware of difficulty of judging which to use at compile time)
    const unsigned int delaySteps = NO_DELAY;
    auto *syn = model.addSynapsePopulation<synapses_1WEIGHTUPDATE, synapses_1POSTSYN>(
        "synapses_1", SynapseMatrixType::SPARSE_INDIVIDUALG, delaySteps,
        "neurongroup", "neurongroup_1",
        synapses_1_p, synapses_1_ini,
        {}, {});
    syn->setSpanType(SynapseGroup::SpanType::POSTSYNAPTIC);
    syn->setMaxConnections(maxRowsynapses_1);
    syn->setMaxSourceConnections(maxColsynapses_1);
    }
}

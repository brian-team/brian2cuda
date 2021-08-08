{% macro cu_file() %}
#include<stdlib.h>
#include "brianlib/cuda_utils.h"
#include "objects.h"
#include<ctime>

{% for codeobj in code_objects | sort(attribute='name') %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

{% for name in user_headers %}
#include {{name}}
{% endfor %}

void brian_start()
{
    _init_arrays();
    _load_arrays();
    srand(time(NULL));

    // Initialize clocks (link timestep and dt to the respective arrays)
    {% for clock in clocks | sort(attribute='name') %}
    brian::{{clock.name}}.timestep = brian::{{array_specs[clock.variables['timestep']]}};
    brian::{{clock.name}}.dt = brian::{{array_specs[clock.variables['dt']]}};
    brian::{{clock.name}}.t = brian::{{array_specs[clock.variables['t']]}};
    {% endfor %}
}

void brian_end()
{
    _write_arrays();
    _dealloc_arrays();
}

{% for name, lines in run_funcs.items() | sort(attribute='name') %}
void {{name}}()
{
    using namespace brian;

    {{lines|autoindent}}
}

{% endfor %}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

void brian_start();
void brian_end();

{% for name, lines in run_funcs.items() | sort(attribute='name') %}
void {{name}}();
{% endfor %}

{% endmacro %}

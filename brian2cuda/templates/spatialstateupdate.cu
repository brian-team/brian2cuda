////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{# USES_VARIABLES { Cm, dt, v, N, Ic,
                  _ab_star0, _ab_star1, _ab_star2, _b_plus, _b_minus,
                  _v_star, _u_plus, _u_minus,
                  _v_previous,
                  _gtot_all, _I0_all,
                  _c,
                  _P_diag, _P_parent, _P_children,
                  _B, _morph_parent_i, _starts, _ends,
                  _morph_children, _morph_children_num, _morph_idxchild,
                  _invr0, _invrn} #}

{% extends 'common_group.cu' %}

{% block kernel %}

// three kernel definitions here

{% endblock %}

{% block kernel_call %}
{% endblock %}

{% block prepare_kernel %}
{% endblock %}

{% block occupancy %}
{% endblock %}

{% block kernel_info %}
{% endblock %}

{% block define_N %}
{% endblock %}

{% block extra_maincode %}

// three kernel calls here (via CUDA_SAFE_CALL)

// remark:
// of each ['_invr', 'Ri', 'Cm', 'dt', 'area', 'r_length_1', 'r_length_2', '_ab_star0', '_ab_star1', '_ab_star2', '_starts', '_ends', '_invr0', '_invrn', '_b_plus', '_b_minus']
// we have devVARNAME array on GPU now

{% endblock %}
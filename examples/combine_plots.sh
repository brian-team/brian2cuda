#!/bin/bash

convert \( \( CUBA_cpp/CUBA_cpp_rasterplot.png CUBA_CUDA/CUBA_CUDA_rasterplot.png -append \) \
	   \( CUBA_DISTDELAYS_cpp/CUBA_DISTDELAYS_cpp_rasterplot.png CUBA_DISTDELAYS_CUDA/CUBA_DISTDELAYS_CUDA_rasterplot.png -append \) +append \) \
	\( STDP_standalone_cpp/STDP_standalone_cpp_plots.png STDP_standalone_cuda/STDP_standalone_cuda_plots.png -append \) +append result_figures.png


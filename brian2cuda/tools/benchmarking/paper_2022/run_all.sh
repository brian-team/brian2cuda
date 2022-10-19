# Run all simulations for all figure in Alevi et al. 2022
# Any command line arguments are passed to `bash run_benchmark_suite.sh`

SCRIPTS_NO_PROFILING="\
    run_benchmarks_fig5_figS1.py \
    run_benchmarks_fig6AB.py \
    run_benchmarks_fig6C.py \
    run_benchmarks_fig8.py"

# XXX: fig9 simulations were executed on a different GPU (RTX2080)
#run_benchmarks_fig9.py"

SCRIPTS_WITH_PROFILING="\
    run_benchmarks_fig7A.py \
    run_benchmarks_fig7B.py \
    run_benchmarks_fig7CD.py"

cp ../run_benchmark_suite.py run_benchmark_suite.bak.py

for SCRIPT in $SCRIPTS_NO_PROFILING
do
    echo "Running benchmarks defined in $SCRIPT ..."
    cp $SCRIPT ../run_benchmark_suite.py
    (
        cd ..
        bash run_benchmark_suite.sh --name "${SCRIPT%.*}" $@
    )
done

for SCRIPT in $SCRIPTS_WITH_PROFILING
do
    echo "Running benchmarks defined in $SCRIPT ..."
    cp $SCRIPT ../run_benchmark_suite.py
    (
        cd ..
        bash run_benchmark_suite.sh --name "${SCRIPT%.*}" --profile $@
    )
done

echo "Restoring previous run_benchmark_suite.py ..."
mv run_benchmark_suite.bak.py ../run_benchmark_suite.py

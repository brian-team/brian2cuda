# example script to run multiple examples, the code and results will be in:
base='multirun-results'
codefolder=$base/code
resultsfolder=$base/results
lofgile=$base/log.txt

# Exit at first failing example run
#set -e -o pipefail

mkdir -p $base
for n in 5000 50000 500000; do
    for a_flag in "" "no-"; do
        for b_flag in "" "no-"; do
            for sp_flag in "" "no-"; do
                for bl in 1 15; do

                    args="--devicename cuda_standalone \
                        --resultsfolder $resultsfolder \
                        --codefolder $codefolder \
                        --profiling \
                        --monitors \
                        --N $n \
                        --"$a_flag"atomics \
                        --"$b_flag"bundle-mode \
                        --"$sp_flag"single-precision \
                        --num-blocks $bl"

                    cmd="python brunelhakim.py --no-heterog-delays $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile

                    cmd="python brunelhakim.py --heterog-delays --narrow-delaydistr $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile

                    cmd="python brunelhakim.py --heterog-delays --no-narrow-delaydistr $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile

                    cmd="python cobahh.py --scenario brian2-example $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile

                    cmd="python cobahh.py --scenario pseudocoupled-1000 $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile

                    cmd="python cobahh.py --scenario pseudocoupled-80 $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile

                    cmd="python cobahh.py --scenario uncoupled $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile

                    cmd="python mushroombody.py $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile

                    cmd="python cuba.py $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile

                    cmd="python stdp.py --delays none $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile

                    cmd="python stdp.py --delays homogeneous $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile

                    cmd="python stdp.py --delays heterogeneous $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile
                done
            done
        done
    done
done

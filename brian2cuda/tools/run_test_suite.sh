logdir=test_suit_logs
logfile="$logdir/$(date +%y-%m-%d_%T)"
mkdir -p "$logdir"
python run_test_suite.py 2>&1 | tee $logfile

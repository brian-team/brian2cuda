#!/usr/bin/env bash
# Start an ssh server on a compute node that listens on a specified port
# (default: 2323, set via -p <port>)

USAGE=$(cat << END
USAGE: $0 [-p <port>] -- <qsub-args>

    <qsub-args> are any arguments that the 'qsub' command accepts (e.g. -l cuda=1).

END
)

echo_usage() {
    echo "$USAGE"
}

# -p <port> sets the local ssh tunnel port
# All other command line arguments are passed to `qsub`

# Default local port that will be forwarded
LOCALPORT=2323

# Parse arguments from command line, this will set LOCALPORT if -p option was
# found and pass all remaining arguments to `qsub`
OPTS=$(getopt --options p: --name "$0" -- "$@")
if [ "$?" -ne 0 ]; then
    echo_usage
    exit 1
fi
eval set -- "$OPTS"
while true; do
    case "$1" in
		-p )
			LOCALPORT="$2"
			shift 2
            ;;
        -- )
            shift
            break
            ;;
        * )
            echo "ERROR: Unknown parameter $1"
            echo_usage
            exit 1
    esac
done

source /opt/ge/default/common/settings.sh

echo "Running: qsub submit-ssh.sh"
OUT=$(qsub $@ /cognition/home/local/sshd/.conf/submit-ssh.sh $LOCALPORT)

if [ "$?" -ne 0 ]; then
    echo "ERROR: qsub command failed with error code $?"
    exit 1
fi

SSHSERVER_JOB_ID=$(echo $OUT | awk '{print $3}')
echo "Running: get-sshserver-tunnel-cmd.sh"
/cognition/home/local/sshd/.conf/get-sshserver-tunnel-cmd.sh $SSHSERVER_JOB_ID

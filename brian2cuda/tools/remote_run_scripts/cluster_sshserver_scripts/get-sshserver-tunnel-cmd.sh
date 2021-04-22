#!/usr/bin/env bash
# Test if submit-ssh.sh is running, wait if it is queuing, give information if
# it failed
#
# $1 is the job id of the sshserver.sh script

echo_output_error() {
    CAT_OUTPUT="cat ~/.cluster-sshd/submit-ssh.o"
    echo $CAT_OUTPUT && $CAT_OUTPUT

    CAT_ERROR="cat ~/.cluster-sshd/submit-ssh.e"
    echo $CAT_ERROR && $CAT_ERROR
}

SSHSERVER_JOB_ID="$1"
while true; do
    STATE=$(qstat | grep $SSHSERVER_JOB_ID | awk '{print $5}')
    case "$STATE" in
        r )
            echo "Job ($SSHSERVER_JOB_ID) is running!"
            break
            ;;
        qw )
            echo "Job ($SSHSERVER_JOB_ID) is waiting in queue (qw) [time: $(date)]"
            sleep 1
            ;;
        "" )
            echo "ERROR: Job ($SSHSERVER_JOB_ID) is not in job queue. Running qacct:"
            qacct -j $SSHSERVER_JOB_ID
            echo_output_error
            exit 1
            ;;
        * )
            echo "Job ($SSHSERVER_JOB_ID) has unexpected STATE `$STATE`. Running qstat:"
            qstat -j $SSHSERVER_JOB_ID
            echo_output_error
            exit 1
            ;;
    esac
done

# Print out the ssh command that was written to the ssh-server output file
cat ~/.cluster-sshd/sshserver-tunnel-cmd


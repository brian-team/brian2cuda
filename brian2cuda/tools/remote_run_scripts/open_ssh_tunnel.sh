#!/usr/bin/env bash
# You can execute this script on your local machine. It will start a ssh server
# on a compute node and will open a ssh tunnel to that server for you (which
# stays open in your terminal). This script will also take care of stopping the
# ssh server and freeing the resources once you close the ssh tunnel.
#
# All command line arguments will be passed to `qsub` (e.g. -l cuda=1)

# You can set the local port for your ssh tunnel here
LOCALPORT=1234

ERROR_MSG=$(cat << END

ERROR: ssh command to terminate ssh server on cluster failed! Please terminate
the sshserver manually with 'qdel ssh-server' on the cluster head node!

END
)

# Close sshserver on compute node when exiting this script
function cleanup {
    echo "Script exiting... closing sshserver on compute node..."
    ssh cluster "source /opt/ge/default/common/settings.sh && qdel ssh-server"
    if [ "$?" -eq 0 ]; then
        echo "... done!"
    else
        echo "$ERROR_MSG"
    fi
}

# Register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

# Load configuration file
script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$script_path/_load_remote_config.sh" ~/.brian2cuda-remote-dev.conf

# Start sshserver on compute node
ssh -t cluster "/cognition/home/local/sshd/start-sshserver.sh -p $LOCALPORT -- $@"
if [ "$?" -ne 0 ]; then
    echo "ERROR start-sshserver.sh did not succeed. Terminating."
    exit 1
fi

# Get information about running server
TUNNEL_CMD=$(ssh cluster "cat ~/.cluster-sshd/sshserver-tunnel-cmd")
# Open ssh tunnel to ssh server on compute node
$TUNNEL_CMD

#!/usr/bin/env bash
#$ -N ssh-server                    # Set consistent job name (allows for easy deletion alias)
#$ -q cognition-all.q               # Use the cognition queue
#$ -o ~/.cluster-sshd/submit-ssh.o  # Set consistent output file to get ssh tunnel command
#$ -o ~/.cluster-sshd/submit-ssh.e  # Set consistent error file

# If you want to request multiple CPUs, add the following line to the above block or to your qsub command
# #$ -binding linear:2  # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)

# If you also want to request a GPU, add the following line to the above block:
# #$ -l cuda=1          # request one GPU

# Pass local port for ssh tunnel as $1 argument (default: 2323)
LOCALPORT=${1:-2323}

/cognition/home/local/sshd/.conf/sshserver.sh $LOCALPORT

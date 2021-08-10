#!/usr/bin/env bash

# $1 local port that will be forwarded (default: 2323)
LOCALPORT=${1:-2323}

if [ ! -d ~/.cluster-sshd ]; then
	mkdir ~/.cluster-sshd
fi
if [ ! -f ~/.cluster-sshd/ssh_host_dsa_key ]; then
	ssh-keygen -q -N "" -t dsa -f ~/.cluster-sshd/ssh_host_dsa_key
fi
if [ ! -f ~/.cluster-sshd/ssh_host_rsa_key ]; then
	ssh-keygen -q -N "" -t rsa -b 4096 -f ~/.cluster-sshd/ssh_host_rsa_key
fi
if [ ! -f ~/.cluster-sshd/ssh_host_ecdsa_key ]; then
	ssh-keygen -q -N "" -t ecdsa -f ~/.cluster-sshd/ssh_host_ecdsa_key
fi
if [ ! -f ~/.cluster-sshd/ssh_host_ed25519_key ]; then
	ssh-keygen -q -N "" -t ed25519 -f ~/.cluster-sshd/ssh_host_ed25519_key
fi

read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done

USER=`whoami`
IP=`/sbin/ifconfig |grep 192.168.1|awk '{print $2}'`
TUNNEL_CMD="ssh -L $LOCALPORT:$IP:$PORT $USER@cluster.ml.tu-berlin.de"
echo "$TUNNEL_CMD"
echo "$TUNNEL_CMD" > ~/.cluster-sshd/sshserver-tunnel-cmd
/usr/sbin/sshd -D  -f /cognition/home/local/sshd/.conf/sshd_config -p $PORT

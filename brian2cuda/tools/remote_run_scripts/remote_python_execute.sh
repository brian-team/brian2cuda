# $1 is the python script name

python_script="$1"

# 1. copy $python_script to remote (localhost -p 1234 or sprekeler_cluster)

path_conda_sh_remote="~/anaconda3/etc/profile.d/conda.sh"
# conda environment for brian2cuda on the remote
conda_env_remote="b2c"

ssh -p 1234 -i ~/.ssh/id_internal localhost "/bin/bash" << EOF
    source $path_conda_sh_remote
    conda activate $conda_env_remote
    python $python_script
EOF

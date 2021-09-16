#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=msa_test
#SBATCH --time=0-01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --qos=devel
#SBATCH --output=log/slurm-%j.out

source $HOME/.bashrc
cd /cluster/home/yingyue/code/rankine

nt=$SLURM_NTASKS
ppn=$SLURM_NTASKS_PER_NODE

##single_wind_obs exps
t=0
for real in `seq 1 1000`; do
    offset_node=`echo $t / $ppn |bc`
    echo $real
    #srun -N 1 -n 1 -r $offset_node python run_single_wind_obs.py $real &
    srun -N 1 -n 1 -r $offset_node python run_position_obs.py $real &
    t=$((t+1))
    if [ $t == $nt ]; then
        t=0
        wait
    fi
done
wait
kill -HUP $PPID

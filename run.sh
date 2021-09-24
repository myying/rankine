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

##single_wind_obs position_obs exps
#t=0
#for real in `seq 1 1000`; do
#    offset_node=`echo $t / $ppn |bc`
#    echo $real
#    srun -N 1 -n 1 -r $offset_node python run_single_wind_obs.py $real &
#    #srun -N 1 -n 1 -r $offset_node python run_position_obs.py $real &
#    t=$((t+1))
#    if [ $t == $nt ]; then
#        t=0
#        wait
#    fi
#done

###localization tuning runs
#t=0
#for real in `seq 51 100`; do
#    for loc_sprd in 3; do
#        for ns in 2 3 4 5 6 7; do
#            offset_node=`echo $t / $ppn |bc`
#            echo $real $loc_sprd $ns
#            srun -N 1 -n 1 -r $offset_node python run_localization_tuning.py $real $loc_sprd $ns &
#            t=$((t+1))
#            if [ $t == $nt ]; then
#                t=0
#                wait
#            fi
#        done
#    done
#done

###full network assimilation exps
t=0
for real in `seq 1 200`; do
    for loc_sprd in 1 3 5; do
        for phase_amp in "1.0 0.0" "0.5 0.5" "0.0 1.0"; do
            offset_node=`echo $t / $ppn |bc`
            echo $real $loc_sprd $phase_amp 
            srun -N 1 -n 1 -r $offset_node python run_full_network.py $real $loc_sprd $phase_amp &
            t=$((t+1))
            if [ $t == $nt ]; then
                t=0
                wait
            fi
        done
    done
done

wait
kill -HUP $PPID

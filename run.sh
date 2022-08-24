#!/bin/bash
###List of python jobs
###Submit the python jobs in HPC, here srun is the particular command that works on my machine

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
for loc_sprd in 1 3 5; do
    for vmax_sprd in 0; do
        for rmw_sprd in 0.0; do
            for phase_amp in 1.0 0.5 0.0; do
                for real in `seq 1 100`; do
                    offset_node=`echo $t / $ppn |bc`
                    echo $real $loc_sprd $phase_amp
                    srun -N 1 -n 1 -r $offset_node python run_full_network.py $real $loc_sprd $vmax_sprd $rmw_sprd $phase_amp &
                    t=$((t+1))
                    if [ $t == $nt ]; then
                        t=0
                        wait
                    fi
                done
            done
        done
    done
done

###cycling DA exps
t=0
for filter_kind in "NoDA 1 1" "EnSRF 1 1" "EnSRF 3 1" "EnSRF 3 3" "EnSRF 2 1" "EnSRF 2 2" "EnSRF 4 1" "EnSRF 4 4"; do
    for real in `seq 1 100`; do
        offset_node=`echo $t / $ppn |bc`
        echo $real $filter_kind
        srun -N 1 -n 1 -r $offset_node python run_cycling.py $real $filter_kind &
        t=$((t+1))
        if [ $t == $nt ]; then
            t=0
            wait
        fi
    done
done

wait
kill -HUP $PPID

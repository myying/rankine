#!/bin/bash
###List of python jobs
###Submit the python jobs in HPC, here mpirun is the particular command that works on my machine

source $HOME/.bashrc
cd /cluster/home/yingyue/code/rankine

##single_wind_obs position_obs exps
for real in `seq 1 1000`; do
    mpirun -np 1 python run_single_wind_obs.py $real &
    mpirun -np 1 python run_position_obs.py $real &
done

###localization tuning runs
for real in `seq 1 100`; do
    for loc_sprd in 3; do
        for ns in 2 3 4 5 6 7; do
            mpirun -np 1 python run_localization_tuning.py $real $loc_sprd $ns &
        done
    done
done

###full network assimilation exps
for loc_sprd in 1 3 5; do
    for real in `seq 1 100`; do
        mpirun -np 1 python run_full_network.py $real $loc_sprd 0 0.0 1.0 1 &
        mpirun -np 1 python run_full_network.py $real $loc_sprd 0 0.0 0.0 1 &
        mpirun -np 1 python run_full_network.py $real $loc_sprd 0 0.0 1.0 2 &
    done
done
for real in `seq 1 100`; do
    mpirun -np 1 python run_full_network.py $real 3 2 0.0 1.0 1 &
    mpirun -np 1 python run_full_network.py $real 3 9 0.0 1.0 1 &
    mpirun -np 1 python run_full_network.py $real 3 0 0.5 1.0 1 &
    mpirun -np 1 python run_full_network.py $real 3 0 1.0 1.0 1 &
done

###cycling DA exps
for filter_kind in "NoDA 1 1" "EnSRF 1 1" "EnSRF 3 1" "EnSRF 3 3" "EnSRF 2 1" "EnSRF 2 2" "EnSRF 4 1" "EnSRF 4 4"; do
    for real in `seq 1 100`; do
        mpirun -np 1 python run_cycling.py $real $filter_kind perfect_model 1.0 0 &
        mpirun -np 1 python run_cycling.py $real $filter_kind perfect_model 0.0 0 &
        mpirun -np 1 python run_cycling.py $real $filter_kind perfect_model 1.0 1 &
        mpirun -np 1 python run_cycling.py $real $filter_kind imperfect_model 1.0 0 &
    done
done


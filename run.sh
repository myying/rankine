#!/bin/bash -x
#SBATCH --account=nn2993k
#SBATCH --job-name=vortex
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

source ~/.bashrc
cd /cluster/home/yingyue/code/rankine

for L in 4 6; do
    for ROI in 8 12 16 20 24 32 40 48 64; do
        python run_full.py 2 $L $ROI &
    done
done
wait

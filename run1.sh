#!/bin/bash
#PBS -N run
#PBS -A NMMM0021
#PBS -l select=1:ncpus=20:mpiprocs=20
#PBS -l walltime=06:00:00
#PBS -q premium
#PBS -j oe
#PBS -o log
source ~/.bashrc
casename=imperfect_model

n=20
s=0
for i in {001..500}; do
  python run_truth.py $casename $i &
  s=$((s+1))
  if [[ $s == $n ]]; then
    s=0
    wait
  fi
done
wait

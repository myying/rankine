#!/bin/bash
casename=perfect_model
filter_type=EnSRF
ns=$1

for n in `seq 1 20 500`; do
  cat > tmp << EOF
#PBS -N run
#PBS -A NMMM0021
#PBS -l select=1:ncpus=20:mpiprocs=20
#PBS -l walltime=05:00:00
#PBS -q premium
#PBS -j oe
#PBS -o log
source ~/.bashrc

for i in \`seq $n $((n+20))\`; do
  python run_cycle.py $casename \$i $filter_type $ns &
done
wait
EOF

  qsub tmp
  rm tmp
done

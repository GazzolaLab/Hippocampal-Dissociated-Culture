#!/bin/bash
#SBATCH -J run_full_scale
#SBATCH -o ./results/run_full_scale.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=144
#SBATCH -t 2:00:00
#SBATCH -p gg      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#



export NEURONROOT=$SCRATCH/bin/nrnpython_gcc
export PYTHONPATH=$HOME/model/Hippocampal-Dissociated-Culture:$NEURONROOT/lib/python:$SCRATCH/python3.11/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/bin:$SCRATCH/python3.11/site-packages/bin:$PATH

export DATA_PREFIX=$SCRATCH/MiV
export CONFIG_PREFIX="config"
export MAIN_CONFIG="Full_Scale.yaml"

results_path=$SCRATCH/MiV/results/Full_Scale_$SLURM_JOB_ID
export results_path

mkdir -p ${results_path}



mpirun -np 288 `which run-network` \
       --use-coreneuron \
       --config-file=${MAIN_CONFIG}  \
       --config-prefix=${CONFIG_PREFIX} \
       --template-paths="templates" \
       --dataset-prefix="${DATA_PREFIX}" \
       --results-path=${results_path} \
       --io-size=80 \
       --tstop=2000 \
       --v-init=-75 \
       --results-write-time=60 \
       --stimulus-onset=0.0 \
       --max-walltime-hours=1 \
       --dt 0.025 \
       --coordinates-namespace="Generated Coordinates" \
       --verbose


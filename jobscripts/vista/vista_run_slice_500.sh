#!/bin/bash
#SBATCH -J run_slice
#SBATCH -o ./results/run_slice.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=144
#SBATCH -t 2:00:00
#SBATCH -p gh      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#



export NEURONROOT=$SCRATCH/bin/nrnpython_gcc
export PYTHONPATH=$HOME/model/Hippocampal-Dissociated-Culture:$NEURONROOT/lib/python:$SCRATCH/python3.11/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/bin:$SCRATCH/python3.11/site-packages/bin:$PATH

export DATA_PREFIX=$SCRATCH/MiV
export CONFIG_PREFIX="config"
export MAIN_CONFIG="Slice_500.yaml"

results_path=$SCRATCH/MiV/results/Microcircuit_$SLURM_JOB_ID
export results_path

mkdir -p ${results_path}

mpirun -n 2304 run-network \
       --use-coreneuron \
      --config-file=${MAIN_CONFIG}  \
      --config-prefix=${CONFIG_PREFIX} \
      --template-paths="templates" \
      --dataset-prefix="${DATA_PREFIX}" \
      --results-path=${results_path} \
      --spike-input-path="${DATA_PREFIX}/Slice/CA1_Slice_500.h5" \
      --spike-input-namespace="Input Spikes A Diag" \
      --spike-input-attr="Spike Train" \
      --io-size=32 \
      --tstop=10000 \
      --v-init=-75 \
      --results-write-time=60 \
      --stimulus-onset=0.0 \
      --max-walltime-hours=1 \
      --dt 0.025 \
      --coordinates-namespace="Generated Coordinates" \
      --verbose


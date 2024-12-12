#!/bin/bash
#SBATCH -J cut_slice
#SBATCH -o ./results/cut_slice.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=144
#SBATCH -t 2:00:00
#SBATCH -p gg      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

set -x


export NEURONROOT=$SCRATCH/bin/nrnpython
export PYTHONPATH=$HOME/model/Hippocampal-Dissociated-Culture:$NEURONROOT/lib/python:$SCRATCH/python3.9/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/bin:$SCRATCH/python3.9/site-packages/bin:$PATH


export DATA_PREFIX=$SCRATCH/MiV
export CONFIG_PREFIX="config"
export MAIN_CONFIG="Full_Scale.yaml"

results_path=$SCRATCH/MiV/results/cut_slice_$SLURM_JOB_ID
export results_path

cd $SLURM_SUBMIT_DIR

mkdir -p $results_path

mpirun -n 144 `which cut-slice` \
    --arena-id=A --trajectory-id=Diag \
    --config=$MAIN_CONFIG \
    --config-prefix=./config \
    --dataset-prefix="$DATA_PREFIX" \
    --output-path=$results_path \
    --io-size=16 \
    --spike-input-path="${DATA_PREFIX}/Full_Scale/Full_Scale_CA1_all_spike_trains.h5" \
    --spike-input-namespace='Input Spikes A Diag' \
    --spike-input-attr="Spike Train" \
    --coordinates-namespace="Generated Coordinates" \
    --distance-limits -250 250 \
    --write-selection \
    --verbose

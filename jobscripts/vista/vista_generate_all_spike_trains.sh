#!/bin/bash
#SBATCH -J generate_all_spike_trains
#SBATCH -o ./results/generate_all_spike_trains.%j.o
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=576
#SBATCH -t 2:00:00
#SBATCH -p gg      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

export NEURONROOT=$SCRATCH/bin/nrnpython
export PYTHONPATH=$HOME/model/Hippocampal-Dissociated-Culture:$NEURONROOT/lib/python:$SCRATCH/python3.9/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/bin:$SCRATCH/python3.9/site-packages/bin:$PATH

export DATA_PREFIX=$SCRATCH/MiV

dataset_prefix=$SCRATCH/MiV/Full_Scale

mpirun -n 144 `which generate-input-spike-trains` \
       --config=Full_Scale.yaml \
       --config-prefix=./config \
       --selectivity-path=${dataset_prefix}/Full_Scale_CA1_input_features.h5 \
       --selectivity-namespace="Constant Selectivity" \
       --coords-path=${dataset_prefix}/CA1_Full_Scale.h5 \
       --output-path=${dataset_prefix}/Full_Scale_CA1_all_spike_trains.h5 \
       --io-size 8 \
       --chunk-size=10000 \
       --value-chunk-size=100000 \
       -v


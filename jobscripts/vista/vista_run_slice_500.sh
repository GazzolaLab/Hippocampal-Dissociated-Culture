#!/bin/bash
#SBATCH -J run_slice
#SBATCH -o ./results/run_slice_500_gpu.%j.o
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=72
#SBATCH -t 2:00:00
#SBATCH -p gh      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

#module load gcc/14.2.0
#module load python3_mpi

module load nvidia/24.9
module load cuda/12.6
module load openmpi/5.0.5_nvc249
module load phdf5/1.14.4

# export NEURONROOT=$SCRATCH/bin/nrnpython_gcc
# export PYTHONPATH=$HOME/model/Hippocampal-Dissociated-Culture:$NEURONROOT/lib/python:$SCRATCH/python3.11/site-packages:$PYTHONPATH
# export PATH=$NEURONROOT/bin:$SCRATCH/python3.11/site-packages/bin:$PATH

export NEURONROOT=$SCRATCH/bin/nrnpython
export PYTHONPATH=$HOME/model/Hippocampal-Dissociated-Culture:$NEURONROOT/lib/python:$SCRATCH/python3.9/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/bin:$SCRATCH/python3.9/site-packages/bin:$PATH

export DATA_PREFIX=$SCRATCH/MiV
export CONFIG_PREFIX="config"
export MAIN_CONFIG="Slice_500.yaml"

results_path=$SCRATCH/MiV/results/Slice_500_$SLURM_JOB_ID
export results_path

mkdir -p ${results_path}

echo `which run-network`

ibrun run-network \
      --use-coreneuron \
      --coreneuron-gpu \
       --config-file=${MAIN_CONFIG}  \
       --config-prefix=${CONFIG_PREFIX} \
       --template-paths="templates" \
       --dataset-prefix="${DATA_PREFIX}" \
       --results-path=${results_path} \
       --spike-input-path="${DATA_PREFIX}/Slice/CA1_Slice_500.h5" \
       --spike-input-namespace="Input Spikes A Diag" \
       --spike-input-attr="Spike Train" \
       --microcircuit-inputs \
       --io-size=32 \
       --tstop=5000 \
       --v-init=-75 \
       --results-write-time=60 \
       --stimulus-onset=0.0 \
       --max-walltime-hours=1 \
       --dt 0.025 \
       --coordinates-namespace="Generated Coordinates" \
       --verbose


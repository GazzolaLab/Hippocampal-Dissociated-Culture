#!/bin/bash
#SBATCH -J MiV_run_full_scale
#SBATCH -o ./results/MiV_run_full_scale.%j.o
#SBATCH --nodes=160
#SBATCH --ntasks-per-node=56
#SBATCH -t 4:00:00
#SBATCH -p normal      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load phdf5

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/python3.10:$PYTHONPATH
export PATH=$NEURONROOT/bin:$SCRATCH/site-packages/python3.10/bin:$PATH

export I_MPI_ADJUST_SCATTER=2
export I_MPI_ADJUST_SCATTERV=2
export I_MPI_ADJUST_ALLGATHER=2
export I_MPI_ADJUST_ALLGATHERV=2
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

export DATA_PREFIX=$SCRATCH/striped2/MiV
export CONFIG_PREFIX="config"
export MAIN_CONFIG="Full_Scale.yaml"

results_path=$DATA_PREFIX/results/Full_Scale_$SLURM_JOB_ID
export results_path

mkdir -p ${results_path}

ibrun run-network \
    --config-file=${MAIN_CONFIG}  \
    --config-prefix=${CONFIG_PREFIX} \
    --template-paths="templates" \
    --dataset-prefix="${DATA_PREFIX}" \
    --results-path=${results_path} \
    --spike-input-path="${DATA_PREFIX}/Full_Scale/Full_Scale_CA1_all_spike_trains_20250311.h5" \
    --spike-input-namespace="Input Spikes A Diag" \
    --spike-input-attr="Spike Train" \
    --coordinates-namespace="Generated Coordinates" \
    --io-size=80 \
    --tstop=5000 \
    --v-init=-75 \
    --results-write-time=60 \
    --stimulus-onset=0.0 \
    --max-walltime-hours=8 \
    --mechanisms-path mechanisms/build \
    --use-coreneuron \
    --dt 0.025 \
    --arena-id A \
    --stimulus-id Diag \
    --verbose


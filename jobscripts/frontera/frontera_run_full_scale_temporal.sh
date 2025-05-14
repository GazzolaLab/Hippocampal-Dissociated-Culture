#!/bin/bash
#SBATCH -J MiV_run_full_scale_temporal
#SBATCH -o ./results/MiV_run_full_scale_temporal.%j.o
#SBATCH --nodes=160
#SBATCH --ntasks-per-node=56
#SBATCH -t 8:00:00
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
export MAIN_CONFIG="Full_Scale_Temporal_Features.yaml"

results_path=$DATA_PREFIX/results/Full_Scale_Temporal_Features_$SLURM_JOB_ID
export results_path

mkdir -p ${results_path}

ibrun run-network \
    --config-file=${MAIN_CONFIG}  \
    --config-prefix=${CONFIG_PREFIX} \
    --template-paths="templates" \
    --dataset-prefix="${DATA_PREFIX}" \
    --results-path=${results_path} \
    --spike-input-path="${DATA_PREFIX}/Full_Scale/Full_Scale_CA1_temporal_input_spike_trains_20250513.h5" \
    --spike-input-namespace="Temporal Feature Spikes test_temporal_features_20240510" \
    --spike-input-attr="Spike Train" \
    --coordinates-namespace="Generated Coordinates" \
    --io-size=80 \
    --tstop=10000 \
    --v-init=-75 \
    --results-write-time=60 \
    --stimulus-onset=0.0 \
    --max-walltime-hours=8 \
    --mechanisms-path mechanisms/build \
    --use-coreneuron \
    --dt 0.025 \
    --verbose


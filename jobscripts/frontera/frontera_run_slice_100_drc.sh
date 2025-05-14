#!/bin/bash
#SBATCH -J MiV_run_slice_100
#SBATCH -o ./results/MiV_run_slice_100_dynamical_response_features.%j.o
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=56
#SBATCH -t 2:00:00
#SBATCH -p development      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load phdf5

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$NEURONROOT/lib/python:$SCRATCH/site-packages/python3.10:$PYTHONPATH
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
export MAIN_CONFIG="Slice_100_Dynamic_Response_Features.yaml"

results_path=$DATA_PREFIX/results/CA1_Slice_100_$SLURM_JOB_ID
export results_path

mkdir -p ${results_path}

ibrun run-network \
    --config-file=${MAIN_CONFIG}  \
    --config-prefix=${CONFIG_PREFIX} \
    --template-paths="templates" \
    --dataset-prefix="${DATA_PREFIX}" \
    --results-path=${results_path} \
    --spike-input-path="${DATA_PREFIX}/Slice/CA1_Slice_100_dynamical_response_features.h5" \
    --spike-input-namespace="Spatiotemporal Feature Spikes drc_features_20240514" \
    --spike-input-attr="Spike Train" \
    --coordinates-namespace="Generated Coordinates" \
    --microcircuit-inputs \
    --mechanisms-path mechanisms/build \
    --io-size=10 \
    --tstop=10000 \
    --v-init=-75 \
    --results-write-time=60 \
    --stimulus-onset=0.0 \
    --max-walltime-hours=1 \
    --dt 0.025 \
    --use-coreneuron \
    --verbose


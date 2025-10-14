#!/bin/bash
#SBATCH -J MiV_optimize_network
#SBATCH -o ./results/MiV_optimize_network.%j.o
#SBATCH --nodes=385
#SBATCH --ntasks-per-node=56
#SBATCH -t 24:00:00
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

export CONFIG_PREFIX="config"

export DATA_PREFIX="/tmp/MiV_optimize_network"
export CDTools=/home1/apps/CDTools/2.0

export PATH=${CDTools}/bin:$PATH

#results_path=$SCRATCH/MiV/results/optimize_network_$SLURM_JOB_ID
results_path=$SCRATCH/MiV/results/optimize_network_7368181
results_file=dmosopt.optimize_network_20251007_1249.h5 

export results_path
export results_file

mkdir -p ${results_path}

distribute.bash ${SCRATCH}/striped2/MiV/MiV_optimize_network

ibrun -n 21505 \
    optimize-network \
    --config-path=./config/optimize_network.yaml \
    --optimize-file-dir=$results_path \
    --optimize-file-name=$results_file \
    --nprocs-per-worker=336 \
    --n-epochs=4 \
    --population-size=400 \
    --num-generations=400 \
    --n-initial=10 \
    --initial-method=slh \
    --surrogate-method=megp \
    --mechanisms_path=mechanisms/build \
    --no_cleanup \
    --dataset_prefix="$DATA_PREFIX" \
    --config_prefix="$CONFIG_PREFIX" \
    --results_path=$results_path \
    --arena_id=A \
    --stimulus_id=Diag \
    --coordinates_namespace="Generated Coordinates" \
    --spike_input_path="${DATA_PREFIX}/Slice/CA1_Slice_100_dynamical_response_features_20250912.h5" \
    --spike_input_namespaces='Spatiotemporal Feature Spikes drc_features_20250912' \
    --spike_input_attr='Spike Train' \
    --max_walltime_hours=24 \
    --io_size=1

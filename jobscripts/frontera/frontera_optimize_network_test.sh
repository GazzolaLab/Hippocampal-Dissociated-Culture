#!/bin/bash
#SBATCH -J MiV_optimize_network_test
#SBATCH -o ./results/MiV_optimize_network_test.%j.o
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=56
#SBATCH -t 2:00:00
#SBATCH -p development      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load phdf5

export KEOPS_CACHE_FOLDER="/tmp/keops"

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:/tmp/site-packages/python3.10:$PYTHONPATH
export PATH=$NEURONROOT/bin:/tmp/site-packages/python3.10/bin:$PATH

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

results_path=$SCRATCH/MiV/results/optimize_network_$SLURM_JOB_ID
results_path=$SCRATCH/MiV/results/optimize_network_7402624
results_file=dmosopt.optimize_network_20251026_1214.h5

export results_path
export results_file

mkdir -p ${results_path}

distribute.bash ${SCRATCH}/striped2/MiV/MiV_optimize_network
distribute.bash ${WORK}/site-packages

#   --optimize-file-name=$results_file 
 
ibrun -n 559 \
    optimize-network \
    --config-path=./config/optimize_network_test.yaml \
    --optimize-file-dir=$results_path \
    --optimize-file-name=$results_file \
    --nprocs-per-worker=279 \
    --n-epochs=2 \
    --population-size=200 \
    --num-generations=200 \
    --n-initial=5 \
    --initial-method=slh \
    --surrogate-method=megp \
    --optimizer-method=age \
    --mechanisms_path=mechanisms/build \
    --no_cleanup \
    --dataset_prefix="$DATA_PREFIX" \
    --config_prefix="$CONFIG_PREFIX" \
    --results_path=$results_path \
    --arena_id=A \
    --stimulus_id=Diag \
    --coordinates_namespace="Generated Coordinates" \
    --spike_input_path="${DATA_PREFIX}/Slice/CA1_Slice_100_dynamical_response_features_20250922.h5" \
    --spike_input_namespaces='Spatiotemporal Feature Spikes drc_features_20250922' \
    --spike_input_attr='Spike Train' \
    --max_walltime_hours=2 \
    --io_size=1

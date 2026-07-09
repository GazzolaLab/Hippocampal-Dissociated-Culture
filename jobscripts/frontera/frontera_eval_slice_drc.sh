#!/bin/bash
#SBATCH -J MiV_eval_slice_drc
#SBATCH -o ./results/MiV_eval_slice_drc.%j.o
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=56
#SBATCH -t 2:00:00
#SBATCH -p normal      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load phdf5/1.14.6

source $SCRATCH/venv/bin/activate

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH

export I_MPI_ADJUST_SCATTER=2
export I_MPI_ADJUST_SCATTERV=2
export I_MPI_ADJUST_ALLGATHER=2
export I_MPI_ADJUST_ALLGATHERV=2
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

export DATA_PREFIX=$SCRATCH/data/MiV
export CONFIG_PREFIX="config"
export MAIN_CONFIG="./config/eval_network_slice.yaml"

results_path=$DATA_PREFIX/results/Slice_100_Dynamic_Response_Features_$SLURM_JOB_ID
export results_path

mkdir -p ${results_path}

#    --params-path=./config/ca1_slice-best.json \
#    --params-label="joint-c+o" \

ibrun eval-network \
    --config-path=${MAIN_CONFIG}  \
    --config-prefix=${CONFIG_PREFIX}  \
    --output-path=${results_path}/eval_results.json \
    --template-paths="templates" \
    --dataset-prefix="${DATA_PREFIX}" \
    --results-path=${results_path} \
    --spike-input-path="${DATA_PREFIX}/Slice/CA1_Slice_100_dynamical_response_features_20250922.h5" \
    --spike-input-attr="Spike Train" \
    --coordinates-namespace="Generated Coordinates" \
    --io-size=80 \
    --tstop=10000 \
    --v-init=-75 \
    --results-write-time=60 \
    --stimulus-onset=0.0 \
    --max-walltime-hours=4 \
    --mechanisms-path=mechanisms/build \
    --use-coreneuron \
    --dt 0.025 \
    --verbose


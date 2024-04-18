#!/bin/bash
#SBATCH -J run_microcircuit
#SBATCH -o ./results/run_microcircuit.%j.o
#SBATCH --nodes=30
#SBATCH --ntasks-per-node=56
#SBATCH -t 2:00:00
#SBATCH -p development      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python3/3.9.2
module load phdf5/1.10.4

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$SCRATCH/site-packages/intel19/bin:$PATH

export I_MPI_HYDRA_TOPOLIB=ipl
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

export DATA_PREFIX=$SCRATCH/striped2/MiV
export CONFIG_PREFIX="config"
export MAIN_CONFIG="Microcircuit_reducedPYR.yaml"

results_path=$SCRATCH/MiV/results/Microcircuit_$SLURM_JOB_ID
export results_path

mkdir -p ${results_path}

ibrun run-network \
    --config-file=${MAIN_CONFIG}  \
    --config-prefix=${CONFIG_PREFIX} \
    --template-paths="templates" \
    --dataset-prefix="${DATA_PREFIX}" \
    --results-path=${results_path} \
    --io-size=20 \
    --tstop=3000 \
    --v-init=-75 \
    --results-write-time=60 \
    --stimulus-onset=0.0 \
    --max-walltime-hours=1 \
    --dt 0.025 \
    --verbose


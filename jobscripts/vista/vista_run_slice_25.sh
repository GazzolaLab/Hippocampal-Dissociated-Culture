#!/bin/bash
#SBATCH -J run_slice_25
#SBATCH -o ./results/run_slice_25.%j.o
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=72
#SBATCH -t 2:00:00
#SBATCH -p gh      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

export NEURONROOT=$SCRATCH/bin/nrnpython_nvhpc
export PYTHONPATH=$HOME/model/Hippocampal-Dissociated-Culture:$NEURONROOT/lib/python:$SCRATCH/python3.9/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/bin:$SCRATCH/python3.9/site-packages/bin:$PATH

export DATA_PREFIX=$SCRATCH/MiV
export CONFIG_PREFIX="config"
export MAIN_CONFIG="Slice_25.yaml"

results_path=$SCRATCH/MiV/results/Slice_25_$SLURM_JOB_ID
export results_path
mkdir -p ${results_path}

nvidia-smi

#export UCX_TLS=rc,mm,cuda_copy,gdr_copy,cuda_ipc
#export UCX_TLS=rc,sm,cuda_copy,gdr_copy,cuda_ipc 

ibrun ./aarch64/special -mpi -python ./run_slice_25.py \


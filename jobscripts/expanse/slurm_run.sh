#!/bin/bash -
#SBATCH -o sim_run.stdout
#SBATCH -e sim_run.stderr
#SBATCH --ntasks-per-node 128
#SBATCH --mem=128GB
#SBATCH --job-name=mivsim
#SBATCH --nodes=1
#SBATCH --account=uic409
#SBATCH --partition compute
#SBATCH --time 48:00:00
#SBATCH --constraint="lustre"

module purge

# SDSC Expanse - specific load
module load shared
module load cpu/0.15.4
module load slurm/expanse/21.08.8
module load sdsc/1.0

# Dependencies
module load gcc/10.2.0
module load openmpi cmake anaconda3 

# Load MiV-Simulator
SIMULATOR_PATH=/expanse/lustre/projects/uic409/skim449/MiV2
module use ${SIMULATOR_PATH}/modules
module load miv-simulator
eval "$(conda shell.bash hook)"  # Reset anaconda3 hook
conda activate ${SIMULATOR_PATH}/conda_env/miv
which python
which pip

export MIV_RUN_DIR=${SLURM_SUBMIT_DIR}
cd ${MIV_RUN_DIR}

# RUN
set -e

CONFIG_PREFIX="config"
DATASET_PREFIX="datasets"
MAIN_CONFIG="Microcircuit.yaml"
RESULT_DIR=results

rm -rf $RESULT_DIR
mkdir $RESULT_DIR

mpirun -n 64 run-network \
    --config-file=${MAIN_CONFIG}  \
    --config-prefix=${CONFIG_PREFIX} \
    --arena-id=A \
    --stimulus-id=Diag \
    --template-paths="templates" \
    --dataset-prefix="./datasets" \
    --results-path=$RESULT_DIR \
    --io-size=4 \
    --tstop=300000 \
    --v-init=-75 \
    --results-write-time=60 \
    --stimulus-onset=0.0 \
    --max-walltime-hours=48 \
    --dt 0.025 \
    --verbose

echo "Simulation Terminated"

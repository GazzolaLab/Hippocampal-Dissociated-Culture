#!/bin/bash -
#SBATCH -o Input_feat.stdout
#SBATCH -e Input_feat.stderr
#SBATCH --ntasks-per-node 32
#SBATCH --mem=64GB
#SBATCH --job-name=input
#SBATCH --nodes=1
#SBATCH --account=uic409
#SBATCH --partition shared
#SBATCH --time 24:00:00
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
SIMULATOR_PATH=/expanse/lustre/projects/uiuc409/skim449/MiV2
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

# Creating input features and spike trains
echo "generate input features"
mpirun -np 1 generate-input-features \
        -p STIM \
        --config=$MAIN_CONFIG \
        --config-prefix=$CONFIG_PREFIX \
        --coords-path=${DATASET_PREFIX}/Microcircuit_coords.h5 \
        --output-path=${DATASET_PREFIX}/Microcircuit_input_features.h5 \
        -v

echo "generate input spike trains"
mpirun -np 32 generate-input-spike-trains \
             --config=$MAIN_CONFIG \
             --config-prefix=$CONFIG_PREFIX \
             --selectivity-path=${DATASET_PREFIX}/Microcircuit_input_features.h5 \
             --output-path=${DATASET_PREFIX}/Microcircuit_input_spikes.h5 \
             --n-trials=3 -p STIM -v

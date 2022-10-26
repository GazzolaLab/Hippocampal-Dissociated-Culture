#!/bin/bash -
#SBATCH -o sim_build.stdout
#SBATCH -e sim_build.stderr
#SBATCH --ntasks-per-node 8
#SBATCH --mem=32GB
#SBATCH --job-name=Coords
#SBATCH --nodes=1
#SBATCH --account=uic409
#SBATCH --partition shared
#SBATCH --time 10:00:00
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

rm -rf ${DATASET_PREFIX}
mkdir -p ${DATASET_PREFIX}
rm -rf x86_64
nrnivmodl mechanisms/* .

# Creating H5Types definitions
make-h5types --config-prefix $CONFIG_PREFIX -c $MAIN_CONFIG --output-path ${DATASET_PREFIX}/MiV_h5types.h5

# Generating soma coordinates and measuring distances
echo "generating soma coordinates"
generate-soma-coordinates -v \
    --config=$MAIN_CONFIG \
    --config-prefix=$CONFIG_PREFIX \
    --types-path=${DATASET_PREFIX}/MiV_h5types.h5 \
    --output-path=${DATASET_PREFIX}/Microcircuit_coords.h5 \
    --output-namespace='Generated Coordinates'

echo "measure distance"
mpirun -np 1 measure-distances -v \
             -i PYR -i PVBC -i OLM -i STIM \
             --config=$MAIN_CONFIG \
             --config-prefix=$CONFIG_PREFIX \
             --coords-path=${DATASET_PREFIX}/Microcircuit_coords.h5

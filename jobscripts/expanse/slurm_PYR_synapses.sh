#!/bin/bash -
#SBATCH -o PYR_syn.stdout
#SBATCH -e PYR_syn.stderr
#SBATCH --ntasks-per-node 64
#SBATCH --mem=128GB
#SBATCH --job-name=PYRsyn
#SBATCH --nodes=1
#SBATCH --account=uic409
#SBATCH --partition compute
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

rm -f ${DATASET_PREFIX}/PYR_*.h5

# Creating dendritic trees in NeuroH5 format
echo "Neurotree import"
neurotrees_import PYR ${DATASET_PREFIX}/PYR_tree.h5 morphology/PYR.swc

echo "Copy tree structure into config file"
h5copy -p -s '/H5Types' -d '/H5Types' -i ${DATASET_PREFIX}/MiV_h5types.h5 -o ${DATASET_PREFIX}/PYR_tree.h5

# Distributing synapses along dendritic trees
echo "copy tree structure into forest"
mpirun -np 1 neurotrees_copy --write-size 3000 --fill --output ${DATASET_PREFIX}/PYR_forest.h5 ${DATASET_PREFIX}/PYR_tree.h5 PYR 1000

echo "distribute synpase locations"
mpirun -np 64 distribute-synapse-locs \
              --template-path templates \
              --config=$MAIN_CONFIG \
              --config-prefix=$CONFIG_PREFIX \
              --mechanisms-path="mechanisms" \
              --populations PYR \
              --forest-path=${DATASET_PREFIX}/PYR_forest.h5 \
              --output-path=${DATASET_PREFIX}/PYR_forest.h5 \
              --distribution=poisson \
              --io-size=5 --write-size=8 -v \
              --chunk-size=10000 --value-chunk-size=10000

# Generating connections
echo "generate distance connections"
mpirun -np 64 generate-distance-connections \
    --config=$MAIN_CONFIG \
    --config-prefix=$CONFIG_PREFIX \
    --forest-path=${DATASET_PREFIX}/PYR_forest.h5 \
    --connectivity-path=${DATASET_PREFIX}/Microcircuit_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=${DATASET_PREFIX}/Microcircuit_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=20 --cache-size=20 --write-size=100 -v

#!/bin/bash
#SBATCH -J generate_distance_connections_MiV_PYR
#SBATCH -o ./results/generate_distance_connections_MiV_PYR.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH -t 2:00:00
#SBATCH -p development      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load phdf5

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/python3.10:$PYTHONPATH
export PATH=$NEURONROOT/bin:$SCRATCH/site-packages/python3.10/bin:$PATH

export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

export DATA_PREFIX=$SCRATCH/striped2/MiV


ibrun generate-distance-connections -v \
    --config-prefix=./config \
    --config=Full_Scale.yaml \
    --forest-path=$DATA_PREFIX/Full_Scale/PYR_forest_syns.h5 \
    --connectivity-path=$DATA_PREFIX/Full_Scale/CA1_PYR_connections_20240405.h5 \
    --connectivity-namespace=Connections \
    --coords-path=$DATA_PREFIX/Full_Scale/Full_Scale_CA1_coords.h5 \
    --coords-namespace="Generated Coordinates" --dry-run --debug \
    --io-size=4 --cache-size=10 --write-size=250 --value-chunk-size=640000 --chunk-size=10000

#!/bin/bash
#SBATCH -J distribute_synapses_MiV_PYR
#SBATCH -o ./results/distribute_synapses_MiV_PYR.%j.o
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=56
#SBATCH -t 1:00:00
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

ibrun reposition-trees --verbose \
    --config Full_Scale.yaml \
    --config-prefix config \
    --population PYR \
    --forest-path $DATA_PREFIX/Full_Scale/PYR_forest_full.h5 \
    --coords-path $DATA_PREFIX/Full_Scale/Full_Scale_CA1_coords.h5 \
    --coords-namespace 'Generated Coordinates' \
    --output-path $DATA_PREFIX/Full_Scale/PYR_forest_repositioned.h5 \
    --io-size=20 --cache-size=50 --write-size=50 \
    --chunk-size=10000 --value-chunk-size=200000


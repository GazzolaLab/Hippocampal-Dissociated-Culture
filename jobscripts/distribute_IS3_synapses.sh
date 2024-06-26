#!/bin/bash 
mpirun.mpich -np 8 distribute-synapse-locs \
             --template-path templates \
              --config=Full_Scale.yaml \
              --populations IS3 \
              --forest-path=./datasets/Full_Scale/IS3_trees.h5 \
              --output-path=./datasets/Full_Scale/IS3_forest.h5 \
              --distribution=poisson \
              --io-size=1 -v


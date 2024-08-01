#!/bin/bash

dataset_prefix=./datasets/Full_Scale

mpirun.mpich -n 8 generate-input-features \
        -p EC \
        --config=Full_Scale.yaml \
        --config-prefix=./config \
        --coords-path=${dataset_prefix}/Full_Scale_CA1_coords.h5 \
        --output-path=${dataset_prefix}/Full_Scale_CA1_input_features.h5 \
        -v


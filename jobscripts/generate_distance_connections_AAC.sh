#!/bin/bash

mpirun.mpich -n 8 generate-distance-connections \
    --config-prefix=./config \
    --config=Full_Scale.yaml \
    --forest-path=datasets/Full_Scale/AAC_forest.h5 \
    --connectivity-path=datasets/Full_Scale/AAC_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=datasets/Full_Scale/Full_Scale_CA1_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=1 --cache-size=20 --write-size=1 -v


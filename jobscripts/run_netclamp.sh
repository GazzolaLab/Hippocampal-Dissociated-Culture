#!/bin/bash

#config=Network_Clamp_PYR_Gfluct_gid_48041.yaml
config=Network_Clamp_PYR_gid_48041.yaml

mpirun -np 1 network-clamp go \
       -c $config \
       --config-prefix=config \
       -p PYR -g 48041 -t 1000 --dt 0.025 \
       --template-paths=templates \
       --dataset-prefix="./datasets" \
       --spike-events-path datasets/Microcircuit/MiV_input_spikes.h5 \
       --spike-events-namespace 'Input Spikes' \
       --spike-events-t 'Spike Train' \
       --results-path=results/netclamp


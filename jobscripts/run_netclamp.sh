#!/bin/bash

mpirun -np 1 network-clamp go \
       -c Network_Clamp_PYR_gid_48041.yaml \
       --config-prefix=config \
       -p PYR -g 48041 -t 9450 --dt 0.025 \
       --template-paths=templates \
       --dataset-prefix="./datasets" \
       --spike-events-path datasets/Microcircuit/Microcircuit_input_spikes.h5 \
       --spike-events-namespace 'Input Spikes' \
       --results-path=results/netclamp


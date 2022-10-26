# Hippocampal Dissociated Culture

MiV-Simulator simulation case for experiment + simulator comparison work.

## Description

## Features

- Electrical stimulation
- Gfluct3: spontaneous firing mechanism

## Run Files

- aggregate.py
- jobscripts: slurm job scripts 
    - Expanse (SDSC)
        1. slurm_build_coords.sh
        2. _build connection_
            - slurm_PYR_synapses.sh
            - slurm_PVBC_synapses.sh
            - slurm_OLM_synapses.sh
        3. slurm_aggregate.sh
        4. slurm_run.sh

## Output

- datasets: h5 data files containing simulation construction
- results: h5 data files containing result

## Simulation Files

- config: Yaml configuration file collection
- morphology: morphology files (.swc)
- mechanisms: NEURON mechanism mod files
- templates: neuron model file

## References

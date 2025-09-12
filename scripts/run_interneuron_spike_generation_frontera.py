import os
import sys
from interneuron_spike_generation import run_interneuron_spike_generation

run_interneuron_spike_generation(#signal_id = "drc_features_20250905",
                                 stimulus_duration = 10,
                                 input_signal_file = "datasets/dynamical_response_spike_trains_n150_10s.h5",
                                 neuron_type = "PV",
                                 population_name = "PVBC",
                                 register_population = False,
                                 config = "Full_Scale_Dynamic_Response_Features.yaml",
                                 output_path = "PVBC_dynamical_response_spike_trains_10s.h5",
                                 dataset_prefix = "/scratch1/03320/iraikov/striped2/MiV",
                                 output_prefix = "/scratch1/03320/iraikov/striped2/MiV/results/livn",
                                 plot=False,
                                 io_kwargs={'io_size': 4,
                                            'write_size': 50000,
                                            'chunk_size': 10000,
                                            'value_chunk_size': 100000,
                                        }
                                 )

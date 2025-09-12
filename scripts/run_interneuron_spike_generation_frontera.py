import os
import sys
from interneuron_spike_generation import run_interneuron_spike_generation

for (neuron_type, population_name) in [#("VIP", "IS1"),
                                       #("VIP", "IS2"),
                                       #("VIP", "IS3")
                                       ("SST", "OLM")
                                      ]:

    run_interneuron_spike_generation(signal_id = "drc_features_20250912",
                                     stimulus_duration = 10,
                                     input_signal_file = "/scratch1/03320/iraikov/striped2/MiV/results/livn/EC_dynamical_response_spike_trains_20250912.h5",
                                     neuron_type = neuron_type,
                                     population_name = population_name,
                                     register_population = False,
                                     config = "Full_Scale_Dynamic_Response_Features.yaml",
                                     dataset_prefix = "/scratch1/03320/iraikov/striped2/MiV",
                                     output_prefix = "/scratch1/03320/iraikov/striped2/MiV/results/livn",
                                     plot=False,
                                     dry_run=False,
                                     io_kwargs={'io_size': 4,
                                                'write_size': 50000,
                                                'chunk_size': 10000,
                                                'value_chunk_size': 100000,
                                            })

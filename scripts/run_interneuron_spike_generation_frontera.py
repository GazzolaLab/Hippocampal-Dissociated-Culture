import os
import sys
from interneuron_spike_generation import run_interneuron_spike_generation

for (neuron_type, population_name) in [#("PV", "AAC"),
                                       #("PV", "PVBC"),
                                       #("PV", "BS"),
                                       #("CCK", "CCKBC"),
                                       #("VIP", "IS1"),
                                       #("VIP", "IS2"),
                                       #("VIP", "IS3"),
                                       #("SST", "OLM"),
                                       #("SST", "SCA"),
                                       ("NPY", "IVY"),
                                       ("NPY", "NGFC")
                                      ]:

    run_interneuron_spike_generation(signal_id = "drc_features_20250922",
                                     stimulus_duration = 10,
                                     input_signal_file = "/scratch1/03320/iraikov/striped2/MiV/Full_Scale/Full_Scale_CA1_dynamical_response_spike_trains_20250922.h5",
                                     neuron_type = neuron_type,
                                     population_name = population_name,
                                     register_population = False,
                                     config = "Full_Scale_Dynamic_Response_Features.yaml",
                                     dataset_prefix = "/scratch1/03320/iraikov/striped2/MiV",
                                     output_prefix = "/scratch1/03320/iraikov/striped2/MiV/results",
                                     plot=False,
                                     dry_run=False,
                                     io_kwargs={'io_size': 4,
                                                'write_size': 50000,
                                                'chunk_size': 10000,
                                                'value_chunk_size': 100000,
                                            })

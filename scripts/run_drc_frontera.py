import os
import sys
import drc

drc.run_dynamical_response_characterization(signal_id = "drc_features_20240912",
                                            stimulus_duration = 10,
                                            population_name = "PYR",
                                            register_population = False,
                                            config = "Full_Scale_Dynamic_Response_Features.yaml",
                                            output_path = "PYR_dynamical_response_spike_trains_10s.h5",
                                            dataset_prefix = "/scratch1/03320/iraikov/striped2/MiV",
                                            output_prefix = "/scratch1/03320/iraikov/striped2/MiV/results/livn",
                                            plot=False,
                                            io_kwargs={'io_size': 4,
                                                       'write_size': 50000,
                                                       'chunk_size': 10000,
                                                       'value_chunk_size': 100000,
                                                       }
                                            )

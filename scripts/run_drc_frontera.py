import os
import sys
import drc

drc.run_dynamical_response_characterization(signal_id = "drc_features_20250922",
                                            sampling_strategy="random",
                                            stimulus_duration = 10,
                                            modulation_hz = 5,
                                            population_name = "EC",
                                            register_population = False,
                                            config = "Full_Scale_Dynamic_Response_Features.yaml",
                                            output_path = "EC_dynamical_response_spikes_20250922.h5",
                                            dataset_prefix = "/scratch1/03320/iraikov/striped2/MiV",
                                            output_prefix = "/scratch1/03320/iraikov/striped2/MiV/results/livn",
                                            plot=False,
                                            dry_run=False,
                                            io_kwargs={'io_size': 4,
                                                       'write_size': 50000,
                                                       'chunk_size': 10000,
                                                       'value_chunk_size': 100000,
                                                       }
                                            )

# drc.run_dynamical_response_characterization(signal_id = "drc_features_20250922",
#                                             sampling_strategy="random",
#                                             input_signal_file = "/scratch1/03320/iraikov/striped2/MiV/results/livn/EC_dynamical_response_spikes_20250922.h5",
#                                             stimulus_duration = 10,
#                                             population_name = "CA3",
#                                             register_population = False,
#                                             config = "Full_Scale_Dynamic_Response_Features.yaml",
#                                             output_path = "CA3_dynamical_response_spikes_20250922.h5",
#                                             dataset_prefix = "/scratch1/03320/iraikov/striped2/MiV",
#                                             output_prefix = "/scratch1/03320/iraikov/striped2/MiV/results/livn",
#                                             plot=False,
#                                             dry_run=False,
#                                             io_kwargs={'io_size': 4,
#                                                        'write_size': 50000,
#                                                        'chunk_size': 10000,
#                                                        'value_chunk_size': 100000,
#                                                        }
#                                             )

# drc.run_dynamical_response_characterization(signal_id = "drc_features_20250922",
#                                            sampling_strategy="random",
#                                             input_signal_file = "/scratch1/03320/iraikov/striped2/MiV/results/livn/EC_dynamical_response_spikes_20250922.h5",
#                                             stimulus_duration = 10,
#                                             population_name = "CA2",
#                                             register_population = False,
#                                             config = "Full_Scale_Dynamic_Response_Features.yaml",
#                                             output_path = "CA2_dynamical_response_spikes_20250922.h5",
#                                             dataset_prefix = "/scratch1/03320/iraikov/striped2/MiV",
#                                             output_prefix = "/scratch1/03320/iraikov/striped2/MiV/results/livn",
#                                             plot=False,
#                                             dry_run=False,
#                                             io_kwargs={'io_size': 4,
#                                                        'write_size': 50000,
#                                                        'chunk_size': 10000,
#                                                        'value_chunk_size': 100000,
#                                                        }
#                                             )

# drc.run_dynamical_response_characterization(signal_id = "drc_features_20250922",
#                                             sampling_strategy="random",
#                                             input_signal_file = "/scratch1/03320/iraikov/striped2/MiV/results/livn/EC_dynamical_response_spikes_20250922.h5",
#                                             stimulus_duration = 10,
#                                             population_name = "PYR",
#                                             register_population = False,
#                                             config = "Full_Scale_Dynamic_Response_Features.yaml",
#                                             output_path = "PYR_dynamical_response_spikes_20250922.h5",
#                                             dataset_prefix = "/scratch1/03320/iraikov/striped2/MiV",
#                                             output_prefix = "/scratch1/03320/iraikov/striped2/MiV/results/livn",
#                                             plot=False,
#                                             dry_run=False,
#                                             io_kwargs={'io_size': 4,
#                                                        'write_size': 50000,
#                                                        'chunk_size': 10000,
#                                                        'value_chunk_size': 100000,
#                                                        }
#                                             )

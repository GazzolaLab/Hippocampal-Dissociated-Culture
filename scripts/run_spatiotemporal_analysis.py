import numpy as np
from tqdm import tqdm
from typing import Optional, List
from mpi4py import MPI

from analyze_spatiotemporal_responses import (process_model_spatiotemporal_responses,
                                              aggregate_processed_responses)
from mfdfa_analysis import add_mfdfa_to_population_response
from plot_mfdfa_analysis import create_mfdfa_report
from plot_spatiotemporal_analysis import (save_all_plots,
                                          plot_feature_activities,
                                          plot_dimensional_receptive_fields,
                                          plot_receptive_field_heatmaps)                             

def analyze_spatiotemporal_responses(model_output_path,
                                     model_output_namespace_id,
                                     input_features_path,
                                     input_signal_id,
                                     populations: Optional[List] = None,
                                     time_range=None,
                                     time_variable="t",
                                     include_artificial=True,
                                     time_bin_ms=50.0,
                                     correlation_method='pearson',
                                     use_signal_dimensions=False,  # Fast signal correlation mode
                                     use_binned_features=False,    # Binned feature correlation mode
                                     n_bins_per_dim=10,             # Number of bins per dimension
                                     include_derivatives=True,
                                     include_frequency_bands=True,
                                     frequency_bands=[(1, 4), (4, 8), (8, 15), (15, 30), (30, 100)],
                                     output_dir=None,
                                     fig_format='svg',
                                     analyses=['feature_activities',
                                               'receptive_fields',
                                               'mfdfa_analysis'],
                                     max_gids=None,
                                     max_features=None,
                                     sample_seed=None,
                                     comm=None,
                                     root=0):
    """
    Analysis of model responses to spatio-temporal feature stimuli.
    
    Parameters:
    -----------
    model_output_path : path
        Path to HDF5 file with model responses.
    model_output_namespace_id : str
        Namespace with model output spikes.
    input_features_path : path
        Path to HDF5 file with input signal and features.
    input_signal_id : str
        Namespaces with input signal data.
    populations : List, optional
        List of populations to analyze (default: all)
    time_range : Tuple, optional
        Time range to analyze [tmin, tmax]
    time_variable : str
        Name of the time variable in the spike data
    include_artificial : bool
        Whether to include artificial cells
    output_dir : str, optional
        Directory to save output figures.
    analyses : List
        List of analyses to perform
    
    Returns:
    --------
    Dict
        Dictionary containing processed data and figures
    """

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.rank
    size = comm.size

    population_processed_data = process_model_spatiotemporal_responses(
        model_output_path = model_output_path,
        model_output_namespace_id = model_output_namespace_id,
        input_features_path = input_features_path,
        input_signal_id = input_signal_id,
        populations = populations,
        time_range=time_range,
        time_variable=time_variable,
        include_artificial=include_artificial,
        time_bin_ms=time_bin_ms,
        correlation_method=correlation_method,
        use_signal_dimensions=use_signal_dimensions,  # Fast signal correlation mode
        use_binned_features=use_binned_features,    # Binned feature correlation mode
        n_bins_per_dim=n_bins_per_dim,             # Number of bins per dimension
        include_derivatives=include_derivatives,
        include_frequency_bands=include_frequency_bands,
        frequency_bands=frequency_bands,
        max_gids=max_gids,
        max_features=max_features,
        sample_seed=sample_seed,
        aggregate_results=False, # don't aggregate results yet
        comm=comm,
        root=root
    )

    processed_responses = population_processed_data
    if 'mfdfa_analysis' in analyses:
        processed_responses = add_mfdfa_to_population_response(
            processed_responses,
            time_bin_ms=time_bin_ms,
            analysis_types=['rate', 'isi'],
            comm=comm,
            root=root
        )

    # Aggregate results (with MFDFA extension)
    final_results = aggregate_processed_responses(processed_responses, comm, root)
        
    if rank == root:
        
        processed_data = next(iter(final_results.values()))

        if 'feature_activity' in analyses:
            fig1 = plot_feature_activities(
                processed_data['feature_activities'],
                processed_data['input_metadata']['feature_data'],
                processed_data['input_metadata']['dimensions']
            )

        if 'receptive_fields' in analyses:
            fig2 = plot_dimensional_receptive_fields(processed_data, max_neurons=20)
            
            fig3 = plot_receptive_field_heatmaps(
                processed_data, 
                'temporal_frequency', 
                'spatial_position'
            )
        
        saved_files = save_all_plots(
            processed_data,
            output_dir="./figures/spatiotemporal_analysis",
            file_format=fig_format,
            dpi=300
        )

        if 'mfdfa_analysis' in analyses:
            create_mfdfa_report(
                final_results,
                output_dir='./figures/spatiotemporal_analysis',
                analysis_types=['rate', 'isi'],
                max_individual_plots=9
            )
        
    comm.barrier()


if __name__ == "__main__":
    analyze_spatiotemporal_responses(
        model_output_path = "./results/Full_Scale_Dynamic_Response_Features_7356752/Full_Scale_results.h5",
        model_output_namespace_id = "Spike Events",
        input_features_path = "./datasets/EC_dynamical_response_spikes_20250922.h5",
        input_signal_id = "drc_features_20250922",
        populations = ["PYR"],
        include_artificial = False,
        output_dir="figures/spatiotemporal_analysis",
        fig_format="png",
        correlation_method='mutual_info',
        time_bin_ms=50.0,
        #use_signal_dimensions=True,  # Compute correlations between input signal and output firing rates
        use_binned_features=True,
#        analyses=['tuning_curves'],
#        analyses=['tuning_curves', 'sensitivity_analysis', 'response_examples', 'dynamic_responses'],
        max_gids=10000,
        max_features=10000,
        sample_seed=67
    )
    

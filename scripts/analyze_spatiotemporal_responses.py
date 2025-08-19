import os
import sys
from typing import List, Optional, Tuple, Dict, Any
from collections import defaultdict
from mpi4py import MPI
import numpy as np
from scipy import signal
from scipy.stats import linregress, variation
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import h5py
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from neuroh5.io import (
    read_cell_attributes,
    read_population_names,
    read_population_ranges,
    NeuroH5ProjectionGen,
    bcast_cell_attributes,
)
from miv_simulator import spikedata
from tqdm import tqdm

plt.style.use('ggplot')
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=MEDIUM_SIZE)

mpl.rcParams['font.family'] = 'sans-serif'


def process_model_spatiotemporal_responses(
    model_output_path,
    model_output_namespace_id,
    input_features_path,
    input_signal_id,
    populations: Optional[List] = None,
    time_range=None,
    time_variable="t",
    include_artificial=True,
    **kwargs,
):
    """
    Process model responses to spatio-temporal stimuli.
    
    Parameters:
    -----------
    model_output_path : str
        Path to model output HDF5 file
    model_output_namespace_id : str
        Namespace ID for spike events in the model output
    input_features_path : str
        Path to input features HDF5 file
    input_signal_id : str
        ID of the input signal to analyze
    populations : List, optional
        List of populations to analyze (default: all)
    time_range : Tuple, optional
        Time range to analyze [tmin, tmax]
    time_variable : str
        Name of the time variable in the spike data
    include_artificial : bool
        Whether to include artificial cells
        
    Returns:
    --------
    Dict
        Processed responses with metrics
    """
    # Load spike data from model output
    (population_ranges, N) = read_population_ranges(model_output_path)
    population_names = read_population_names(model_output_path)

    total_num_cells = 0
    pop_num_cells = {}
    pop_start_inds = {}
    for k in population_names:
        pop_start_inds[k] = population_ranges[k][0]
        pop_num_cells[k] = population_ranges[k][1]
        total_num_cells += population_ranges[k][1]

    # Replace None with list of populations
    if populations is None:
        include = []
        for pop in population_names:
            include.append(pop)
    else:
        if isinstance(populations, str):
            include = [populations]
        else:
            include = populations

    # sort according to start index
    include.sort(key=lambda x: pop_start_inds[x])

    spkdata = spikedata.read_spike_events(
        model_output_path,
        include,
        model_output_namespace_id,
        include_artificial=include_artificial,
        spike_train_attr_name=time_variable,
        time_range=time_range,
    )

    spkpoplst = spkdata["spkpoplst"]
    spkindlst = spkdata["spkindlst"]
    spktlst = spkdata["spktlst"]
    num_cell_spks = spkdata["num_cell_spks"]
    pop_active_cells = spkdata["pop_active_cells"]
    tmin = spkdata["tmin"]
    tmax = spkdata["tmax"]
    simulation_duration = tmax - tmin
    fraction_active = {
        pop_name: float(len(pop_active_cells[pop_name]))
        / float(pop_num_cells[pop_name])
        for pop_name in include
    }

    pop_spk_dict = {
        pop_name: (pop_spkinds, pop_spkts)
        for (pop_name, pop_spkinds, pop_spkts) in zip(spkpoplst, spkindlst, spktlst)
    }

    # Load the input signal and feature metadata
    with h5py.File(input_features_path, 'r') as f:
        signal_group = f[f'Signals/{input_signal_id}']
        
        # Get basic metadata
        input_signal = signal_group['data'][:]
        
        # Get dimensions metadata
        dimensions = signal_group['dimensions'][:]
        
        # Extract dimension information
        dim_info = {}
        for dim in dimensions:
            name = dim['name'].decode('utf-8') if isinstance(dim['name'], bytes) else dim['name']
            dim_info[name] = {
                'range': (dim['range_min'], dim['range_max']),
                'scale': dim['scale'].decode('utf-8') if isinstance(dim['scale'], bytes) else dim['scale'],
                'priority': dim['priority']
            }
        
        # Get feature data if available
        feature_data = {}
        if 'feature_data' in signal_group:
            feature_group = signal_group['feature_data']
            for key in feature_group.keys():
                feature_data[key] = feature_group[key][:]
                
        # Get basic parameters
        duration = signal_group.attrs.get('duration', 10.0)  # seconds
        sample_rate = signal_group.attrs.get('sample_rate', 1000)  # Hz
        sample_dt_ms = signal_group.attrs.get('sample_dt_ms', 1.0)  # ms
        n_channels = signal_group.attrs.get('n_channels', input_signal.shape[1] if len(input_signal.shape) > 1 else 1)
    
    processed_responses = {}
    
    for pop_name in include:
        if pop_name not in pop_spk_dict:
            continue
        
        pop_spkinds, pop_spkts = pop_spk_dict[pop_name]
        gid_spike_dict = spikedata.make_spike_dict(pop_spkinds, pop_spkts)
        
        all_spikes = pop_spkts
        cell_metrics = {}

        # basic cell metrics
        for gid, spike_times in gid_spike_dict.items():
            if len(spike_times) == 0:
                cell_metrics[gid] = {
                    'firing_rate': 0,
                    'n_spikes': 0,
                    'spike_times': [],
                    'isi': [],
                    'cv_isi': 0,
                    'burst_index': 0,
                    'spatiotemporal_responses': {}
                }
                continue

            spike_times = np.array(spike_times)

            n_spikes = len(spike_times)
            firing_rate = n_spikes / (simulation_duration / 1000)  # Hz

            # ISI and CV
            isi = np.diff(spike_times)  # in ms
            cv_isi = np.std(isi) / np.mean(isi) if len(isi) > 0 else 0

            # burst index (fraction of ISIs < 10ms)
            burst_threshold = 10  # ms
            burst_index = np.sum(isi < burst_threshold) / len(isi) if len(isi) > 0 else 0

            # Initialize spatio-temporal responses structure
            spatiotemporal_responses = {}
            
            cell_metrics[gid] = {
                'firing_rate': firing_rate,
                'n_spikes': n_spikes,
                'spike_times': spike_times,
                'isi': isi,
                'cv_isi': cv_isi,
                'burst_index': burst_index,
                'spatiotemporal_responses': spatiotemporal_responses
            }

        # Compute population spike rate histogram (10ms bins)
        all_spikes = np.array(sorted(all_spikes))
        bin_size = 10  # ms
        n_bins = int(simulation_duration / bin_size)
        pop_rate, bin_edges = np.histogram(
            all_spikes, bins=n_bins, range=(0, simulation_duration)
        )

        # Normalize to spikes/s per cell
        n_cells = len(gid_spike_dict)
        if n_cells > 0:
            pop_rate = pop_rate / (bin_size/1000) / n_cells  # Hz per cell
        
        # Connect the spike responses to features based on metadata
        if feature_data:
            # If we have feature GIDs and dimensions, correlate them with spiking activity
            
            # Extract feature dimension data
            gids = feature_data.get('gids', [])
            positions = feature_data.get('positions', [])
            
            # Find key dimensions
            temporal_dim = next((name for name in dim_info if 'temporal' in name.lower() and 'frequency' not in name.lower()), None)
            freq_dim = next((name for name in dim_info if 'frequency' in name.lower()), None)
            spatial_dim = next((name for name in dim_info if 'spatial' in name.lower() and 'width' not in name.lower()), None)
            width_dim = next((name for name in dim_info if 'width' in name.lower()), None)
            
            # For each cell, compute responses across feature dimensions
            for gid in cell_metrics:
                spike_times = cell_metrics[gid]['spike_times']
                
                # Skip cells without spikes
                if len(spike_times) == 0:
                    continue
                
                # Analyze response along each dimension
                for dim_name, dim_data in dim_info.items():
                    # Get values for this dimension
                    dim_values = []
                    for i, feature_gid in enumerate(gids):
                        if i < len(positions) and len(positions[i]) > 0:
                            # Find the index of this dimension in the positions array
                            dim_idx = list(dim_info.keys()).index(dim_name)
                            if dim_idx < positions[i].shape[0]:
                                dim_values.append((feature_gid, positions[i][dim_idx]))
                    
                    if not dim_values:
                        continue
                        
                    # Sort by dimension value
                    dim_values.sort(key=lambda x: x[1])
                    
                    # Create bins along this dimension
                    n_bins = min(10, len(dim_values))
                    bin_edges = np.linspace(dim_data['range'][0], dim_data['range'][1], n_bins + 1)
                    
                    # Bin features by dimension value
                    binned_features = [[] for _ in range(n_bins)]
                    for feature_gid, value in dim_values:
                        bin_idx = min(n_bins - 1, max(0, int((value - dim_data['range'][0]) / 
                                                           (dim_data['range'][1] - dim_data['range'][0]) * n_bins)))
                        binned_features[bin_idx].append(feature_gid)
                    
                    # Compute response for each bin
                    bin_responses = []
                    for bin_idx, bin_features in enumerate(binned_features):
                        if not bin_features:
                            bin_responses.append(0)
                            continue
                            
                        # For this bin, compute average response to features in this bin
                        bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                        
                        # TODO: Customize this based on feature type
                        # For now, we'll just use the firing rate as the response
                        bin_responses.append(cell_metrics[gid]['firing_rate'])
                    
                    # Store tuning curve for this dimension
                    cell_metrics[gid]['spatiotemporal_responses'][dim_name] = {
                        'bin_centers': [(bin_edges[i] + bin_edges[i+1])/2 for i in range(n_bins)],
                        'responses': bin_responses
                    }
        
        # Compute spatial receptive fields if input signal has spatial structure
        if n_channels > 1:
            # Divide simulation duration into time windows
            window_size = 500  # ms
            n_windows = int(simulation_duration / window_size)
            
            for gid in cell_metrics:
                spike_times = cell_metrics[gid]['spike_times']
                
                # Skip cells without spikes
                if len(spike_times) == 0:
                    continue
                
                # Initialize spatial receptive field data
                spatial_rf = np.zeros(n_channels)
                temporal_rf = np.zeros(n_windows)
                
                # Count spikes in each time window
                for i in range(n_windows):
                    window_start = i * window_size
                    window_end = (i + 1) * window_size
                    window_spikes = spike_times[(spike_times >= window_start) & (spike_times < window_end)]
                    temporal_rf[i] = len(window_spikes) / (window_size / 1000)  # Convert to Hz
                
                cell_metrics[gid]['spatiotemporal_responses']['spatial_rf'] = spatial_rf
                cell_metrics[gid]['spatiotemporal_responses']['temporal_rf'] = temporal_rf
                cell_metrics[gid]['spatiotemporal_responses']['temporal_windows'] = np.arange(n_windows) * window_size
        
        # Store all processed data
        processed_responses[pop_name] = {
            'input_metadata': {
                'signal': input_signal,
                'dimensions': dim_info,
                'feature_data': feature_data,
                'duration': duration,
                'sample_rate': sample_rate,
                'sample_dt_ms': sample_dt_ms,
                'n_channels': n_channels
            },
            'population_metrics': {
                'all_spikes': all_spikes,
                'population_rate': pop_rate,
                'bin_edges': bin_edges,
                'mean_rate': np.mean(pop_rate) if len(pop_rate) > 0 else 0,
                'n_active_cells': sum(1 for gid, metrics in cell_metrics.items() if metrics['n_spikes'] > 0),
                'total_cells': n_cells,
            },
            'cell_metrics': cell_metrics
        }
    
    # Return processed data for a single population or all populations
    if len(processed_responses) == 1:
        return next(iter(processed_responses.values()))
    else:
        return processed_responses


def plot_spatiotemporal_tuning(processed_data, dim_x, dim_y, metric='firing_rate'):
    """
    Plot spatio-temporal tuning across two dimensions.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed responses from process_model_spatiotemporal_responses
    dim_x : str
        First dimension name for x-axis
    dim_y : str
        Second dimension name for y-axis
    metric : str
        Metric to visualize (default: 'firing_rate')
        
    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure
    """
    cell_metrics = processed_data['cell_metrics']
    input_metadata = processed_data['input_metadata']
    
    # Filter active cells
    active_cells = {gid: metrics for gid, metrics in cell_metrics.items() if metrics['n_spikes'] > 0}
    
    if not active_cells:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No active cells found", ha='center', va='center', fontsize=14)
        return fig
    
    dimensions = input_metadata['dimensions']
    feature_data = input_metadata.get('feature_data', {})
    
    # Check if requested dimensions exist
    if dim_x not in dimensions or dim_y not in dimensions:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Dimensions {dim_x} or {dim_y} not found", ha='center', va='center', fontsize=14)
        return fig
    
    # Get example cells - choose cells with good tuning if possible
    n_example_cells = min(9, len(active_cells))
    
    # Sort cells by firing rate to get the most active cells
    sorted_cells = sorted(active_cells.items(), key=lambda x: x[1]['firing_rate'], reverse=True)
    example_cells = [gid for gid, _ in sorted_cells[:n_example_cells]]
    
    fig = plt.figure(figsize=(15, 12))
    
    # overall heatmap of population responses
    all_x_vals = []
    all_y_vals = []
    all_responses = []
    
    for gid, metrics in active_cells.items():
        # Check if cell has tuning data for both dimensions
        if (dim_x in metrics['spatiotemporal_responses'] and 
            dim_y in metrics['spatiotemporal_responses']):
            
            x_tuning = metrics['spatiotemporal_responses'][dim_x]
            y_tuning = metrics['spatiotemporal_responses'][dim_y]
            
            for x_bin, x_resp in zip(x_tuning['bin_centers'], x_tuning['responses']):
                for y_bin, y_resp in zip(y_tuning['bin_centers'], y_tuning['responses']):
                    all_x_vals.append(x_bin)
                    all_y_vals.append(y_bin)
                    
                    # Use firing rate or other metric as response
                    if metric == 'firing_rate':
                        all_responses.append(metrics['firing_rate'])
                    else:
                        all_responses.append(metrics.get(metric, 0))
    
    # population heatmap
    if all_x_vals and all_y_vals and all_responses:
        ax_heatmap = fig.add_subplot(3, 3, 1)
        
        # interpolate scattered data onto a regular grid
        x_range = np.linspace(min(all_x_vals), max(all_x_vals), 100)
        y_range = np.linspace(min(all_y_vals), max(all_y_vals), 100)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = griddata((all_x_vals, all_y_vals), all_responses, (X, Y), method='linear')
        
        im = ax_heatmap.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
        plt.colorbar(im, ax=ax_heatmap, label=metric)
        
        ax_heatmap.set_xlabel(dim_x)
        ax_heatmap.set_ylabel(dim_y)
        ax_heatmap.set_title(f'Population {metric} Heatmap')
        
        if dimensions[dim_x]['scale'] == 'log':
            ax_heatmap.set_xscale('log')
        if dimensions[dim_y]['scale'] == 'log':
            ax_heatmap.set_yscale('log')
    else:
        ax_heatmap = fig.add_subplot(3, 3, 1)
        ax_heatmap.text(0.5, 0.5, "Insufficient data for heatmap", 
                       ha='center', va='center', fontsize=12)
    
    # individual cell tuning curves for sample cells
    for i, gid in enumerate(example_cells):
        metrics = cell_metrics[gid]
        
        # Create subplot
        ax = fig.add_subplot(3, 3, i+2)
        
        # Check if cell has both tuning curves
        if (dim_x in metrics['spatiotemporal_responses'] and 
            dim_y in metrics['spatiotemporal_responses']):
            
            x_tuning = metrics['spatiotemporal_responses'][dim_x]
            y_tuning = metrics['spatiotemporal_responses'][dim_y]
            
            # Create x tuning curve
            ax.plot(x_tuning['bin_centers'], x_tuning['responses'], 
                   'b-', linewidth=2, label=dim_x)
            
            # Create secondary y-axis for y tuning curve
            ax2 = ax.twinx()
            ax2.plot(y_tuning['bin_centers'], y_tuning['responses'], 
                    'r-', linewidth=2, label=dim_y)
            
            ax.set_xlabel(dim_x)
            ax.set_ylabel(f'{dim_x} Response', color='b')
            ax2.set_ylabel(f'{dim_y} Response', color='r')
            
            if dimensions[dim_x]['scale'] == 'log':
                ax.set_xscale('log')
            
            ax.set_title(f'Cell {gid} (Rate: {metrics["firing_rate"]:.2f} Hz)')
        else:
            ax.text(0.5, 0.5, f"No tuning data for cell {gid}", 
                   ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_spatial_receptive_fields(processed_data, n_cells=9):
    """
    Plot spatial receptive fields for example cells.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed responses from process_model_spatiotemporal_responses
    n_cells : int
        Number of example cells to plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure
    """
    cell_metrics = processed_data['cell_metrics']
    input_metadata = processed_data['input_metadata']
    
    # Filter active cells
    active_cells = {gid: metrics for gid, metrics in cell_metrics.items() if metrics['n_spikes'] > 0}
    
    if not active_cells:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No active cells found", ha='center', va='center', fontsize=14)
        return fig
    
    # Sort cells by firing rate to get the most active cells
    sorted_cells = sorted(active_cells.items(), key=lambda x: x[1]['firing_rate'], reverse=True)
    example_cells = [gid for gid, _ in sorted_cells[:n_cells]]
    
    fig = plt.figure(figsize=(15, 12))
    
    n_channels = input_metadata['n_channels']
    
    # Plot receptive fields for example cells
    for i, gid in enumerate(example_cells):
        metrics = cell_metrics[gid]
        
        ax = fig.add_subplot(3, 3, i+1)
        
        # Check if cell has spatial RF data
        if ('spatiotemporal_responses' in metrics and 
            'spatial_rf' in metrics['spatiotemporal_responses']):
            
            spatial_rf = metrics['spatiotemporal_responses']['spatial_rf']
            
            # Normalize RF for visualization
            if np.max(spatial_rf) > 0:
                normalized_rf = spatial_rf / np.max(spatial_rf)
            else:
                normalized_rf = spatial_rf
            
            # Plot RF as bar chart
            ax.bar(range(n_channels), normalized_rf)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Normalized Response')
            ax.set_title(f'Cell {gid} (Rate: {metrics["firing_rate"]:.2f} Hz)')
            
            # If also has temporal RF, show inset
            if 'temporal_rf' in metrics['spatiotemporal_responses']:
                temporal_rf = metrics['spatiotemporal_responses']['temporal_rf']
                temporal_windows = metrics['spatiotemporal_responses']['temporal_windows']
                
                axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
                axins.plot(temporal_windows, temporal_rf)
                axins.set_title('Temporal RF', fontsize=8)
                axins.tick_params(labelsize=6)
        else:
            ax.text(0.5, 0.5, f"No spatial RF for cell {gid}", 
                   ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_dimensional_tuning_curves(processed_data):
    """
    Plot tuning curves for each dimension.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed responses from process_model_spatiotemporal_responses
        
    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure
    """
    cell_metrics = processed_data['cell_metrics']
    input_metadata = processed_data['input_metadata']
    
    # Filter active cells
    active_cells = {gid: metrics for gid, metrics in cell_metrics.items() if metrics['n_spikes'] > 0}
    
    if not active_cells:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No active cells found", ha='center', va='center', fontsize=14)
        return fig
    
    dimensions = input_metadata['dimensions']
    
    # Get dimensions that have tuning data
    tuned_dimensions = set()
    for gid, metrics in active_cells.items():
        for dim_name in metrics.get('spatiotemporal_responses', {}):
            if dim_name in dimensions:
                tuned_dimensions.add(dim_name)
    
    if not tuned_dimensions:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No dimensional tuning data found", ha='center', va='center', fontsize=14)
        return fig
    
    # Sort dimensions by priority
    sorted_dimensions = sorted(tuned_dimensions, 
                              key=lambda d: dimensions[d]['priority'], 
                              reverse=True)
    
    n_dims = len(sorted_dimensions)
    n_cols = min(3, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 4 * n_rows))
    
    # Sample cells - get the 10 most active cells
    sorted_cells = sorted(active_cells.items(), key=lambda x: x[1]['firing_rate'], reverse=True)
    example_cells = [gid for gid, _ in sorted_cells[:10]]
    
    # tuning curves for each dimension
    for i, dim_name in enumerate(sorted_dimensions):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Get tuning curves for example cells
        for gid in example_cells:
            if (dim_name in active_cells[gid].get('spatiotemporal_responses', {}) and
                'bin_centers' in active_cells[gid]['spatiotemporal_responses'][dim_name]):
                
                tuning = active_cells[gid]['spatiotemporal_responses'][dim_name]
                
                ax.plot(tuning['bin_centers'], tuning['responses'], 
                       '-o', linewidth=1.5, label=f'Cell {gid}')
        
        # Compute population average tuning
        bin_centers = None
        all_responses = []
        
        for gid, metrics in active_cells.items():
            if (dim_name in metrics.get('spatiotemporal_responses', {}) and
                'bin_centers' in metrics['spatiotemporal_responses'][dim_name]):
                
                tuning = metrics['spatiotemporal_responses'][dim_name]
                
                if bin_centers is None:
                    bin_centers = tuning['bin_centers']
                    all_responses = [[] for _ in bin_centers]
                
                # Collect responses for each bin
                for j, resp in enumerate(tuning['responses']):
                    if j < len(all_responses):
                        all_responses[j].append(resp)
        
        if bin_centers is not None:
            mean_responses = [np.mean(resps) if resps else 0 for resps in all_responses]
            sem_responses = [np.std(resps) / np.sqrt(len(resps)) if resps and len(resps) > 1 else 0 
                            for resps in all_responses]
            
            ax.plot(bin_centers, mean_responses, 'k-', linewidth=2.5, label='Population Mean')
            ax.fill_between(bin_centers, 
                           [m - s for m, s in zip(mean_responses, sem_responses)],
                           [m + s for m, s in zip(mean_responses, sem_responses)],
                           color='k', alpha=0.2)
        
        ax.set_xlabel(dim_name)
        ax.set_ylabel('Response')
        ax.set_title(f'{dim_name} Tuning')
        
        if dimensions[dim_name]['scale'] == 'log':
            ax.set_xscale('log')
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    return fig


def plot_feature_sensitivity_analysis(processed_data):
    """
    Analyze and plot sensitivity to different feature dimensions.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed responses from process_model_spatiotemporal_responses
        
    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure
    """
    cell_metrics = processed_data['cell_metrics']
    input_metadata = processed_data['input_metadata']
    
    # Filter active cells
    active_cells = {gid: metrics for gid, metrics in cell_metrics.items() if metrics['n_spikes'] > 0}
    
    if not active_cells:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No active cells found", ha='center', va='center', fontsize=14)
        return fig
    
    dimensions = input_metadata['dimensions']
    
    # Get dimensions that have tuning data
    tuned_dimensions = set()
    for gid, metrics in active_cells.items():
        for dim_name in metrics.get('spatiotemporal_responses', {}):
            if dim_name in dimensions:
                tuned_dimensions.add(dim_name)
    
    if not tuned_dimensions:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No dimensional tuning data found", ha='center', va='center', fontsize=14)
        return fig
    
    # Sort dimensions by priority
    sorted_dimensions = sorted(tuned_dimensions, 
                              key=lambda d: dimensions[d]['priority'], 
                              reverse=True)
    
    fig = plt.figure(figsize=(15, 12))
    
    # sensitivity metrics for each cell and dimension
    sensitivity_metrics = {}
    
    for gid, metrics in active_cells.items():
        sensitivity_metrics[gid] = {}
        
        for dim_name in sorted_dimensions:
            if (dim_name in metrics.get('spatiotemporal_responses', {}) and
                'bin_centers' in metrics['spatiotemporal_responses'][dim_name]):
                
                tuning = metrics['spatiotemporal_responses'][dim_name]
                
                # Calculate metrics:
                # 1. Modulation depth (max - min) / mean
                responses = np.array(tuning['responses'])
                if np.mean(responses) > 0:
                    mod_depth = (np.max(responses) - np.min(responses)) / np.mean(responses)
                else:
                    mod_depth = 0
                
                # 2. Coefficient of variation
                if np.mean(responses) > 0:
                    cv = np.std(responses) / np.mean(responses)
                else:
                    cv = 0
                
                # 3. Try to calculate selectivity index
                if np.sum(responses) > 0:
                    max_idx = np.argmax(responses)
                    bin_centers = np.array(tuning['bin_centers'])
                    preferred_value = bin_centers[max_idx]
                    
                    # Calculate weighted average (center of mass)
                    com = np.sum(responses * bin_centers) / np.sum(responses)
                    
                    # Distance from preferred to center of mass, normalized by range
                    dim_range = dimensions[dim_name]['range'][1] - dimensions[dim_name]['range'][0]
                    selectivity = abs(preferred_value - com) / dim_range
                else:
                    selectivity = 0
                
                sensitivity_metrics[gid][dim_name] = {
                    'modulation_depth': mod_depth,
                    'cv': cv,
                    'selectivity': selectivity
                }
    
    # distribution of sensitivity metrics for each dimension
    metric_to_plot = 'modulation_depth'  # Change this to plot different metrics
    
    ax1 = fig.add_subplot(2, 2, 1)
    
    # sensitivity values for each dimension
    dim_sensitivities = {dim: [] for dim in sorted_dimensions}
    
    for gid, metrics in sensitivity_metrics.items():
        for dim_name, dim_metrics in metrics.items():
            dim_sensitivities[dim_name].append(dim_metrics[metric_to_plot])
    
    ax1.boxplot([dim_sensitivities[dim] for dim in sorted_dimensions], 
               labels=sorted_dimensions)
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel(f'{metric_to_plot.replace("_", " ").title()}')
    ax1.set_title(f'Distribution of {metric_to_plot.replace("_", " ").title()} by Dimension')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # sensitivity correlation between dimensions
    if len(sorted_dimensions) >= 2:
        ax2 = fig.add_subplot(2, 2, 2)
        
        dim1 = sorted_dimensions[0]
        dim2 = sorted_dimensions[1]
        
        x_vals = []
        y_vals = []
        
        for gid, metrics in sensitivity_metrics.items():
            if dim1 in metrics and dim2 in metrics:
                x_vals.append(metrics[dim1][metric_to_plot])
                y_vals.append(metrics[dim2][metric_to_plot])
        
        ax2.scatter(x_vals, y_vals, alpha=0.7)
        ax2.set_xlabel(f'{dim1} {metric_to_plot}')
        ax2.set_ylabel(f'{dim2} {metric_to_plot}')
        ax2.set_title(f'Correlation between {dim1} and {dim2} Sensitivity')
        
        # Add regression line
        if len(x_vals) > 2:
            slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
            x_range = np.linspace(min(x_vals), max(x_vals), 100)
            y_fit = slope * x_range + intercept
            ax2.plot(x_range, y_fit, 'r--', 
                    label=f'R²={r_value**2:.2f}, p={p_value:.4f}')
            ax2.legend()
    
    # cell ranking by sensitivity
    ax3 = fig.add_subplot(2, 2, 3)
    
    overall_sensitivity = {}
    
    for gid, metrics in sensitivity_metrics.items():
        # Average sensitivity across dimensions
        sensitivities = [dim_metrics[metric_to_plot] for dim_metrics in metrics.values()]
        overall_sensitivity[gid] = np.mean(sensitivities) if sensitivities else 0
    
    sorted_cells = sorted(overall_sensitivity.items(), key=lambda x: x[1], reverse=True)
    top_n_cells = 20
    
    if sorted_cells:
        cell_gids = [gid for gid, _ in sorted_cells[:top_n_cells]]
        cell_values = [val for _, val in sorted_cells[:top_n_cells]]
        
        ax3.bar(range(len(cell_gids)), cell_values)
        ax3.set_xticks(range(len(cell_gids)))
        ax3.set_xticklabels([str(gid) for gid in cell_gids], rotation=90)
        ax3.set_xlabel('Cell GID')
        ax3.set_ylabel(f'Average {metric_to_plot.replace("_", " ").title()}')
        ax3.set_title(f'Top {top_n_cells} Cells by {metric_to_plot.replace("_", " ").title()}')
    
    # information-theoretic metrics if available
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Calculate mutual information for each dimension
    mi_values = {}
    
    for dim_name in sorted_dimensions:
        # Collect data for MI calculation
        feature_values = []
        response_values = []
        
        for gid, metrics in active_cells.items():
            if (dim_name in metrics.get('spatiotemporal_responses', {}) and
                'bin_centers' in metrics['spatiotemporal_responses'][dim_name]):
                
                tuning = metrics['spatiotemporal_responses'][dim_name]
                
                # Add each bin center and response as a data point
                for bin_center, response in zip(tuning['bin_centers'], tuning['responses']):
                    feature_values.append(bin_center)
                    response_values.append(response)
        
        # Calculate MI if enough data points
        if len(feature_values) >= 10:
            try:
                # Convert to numpy arrays
                feature_array = np.array(feature_values).reshape(-1, 1)
                response_array = np.array(response_values)
                
                # Calculate MI
                mi = mutual_info_regression(feature_array, response_array)[0]
                mi_values[dim_name] = mi
            except Exception as e:
                print(f"Error calculating MI for {dim_name}: {e}")
                mi_values[dim_name] = 0
        else:
            mi_values[dim_name] = 0
    
    if mi_values:
        dims = list(mi_values.keys())
        mis = [mi_values[dim] for dim in dims]
        
        ax4.bar(dims, mis)
        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Mutual Information (bits)')
        ax4.set_title('Information Content by Dimension')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    else:
        ax4.text(0.5, 0.5, "Insufficient data for MI calculation", 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_spatiotemporal_response_examples(processed_data, n_cells=10):
    """
    Plot detailed spatiotemporal response examples for selected cells.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed responses from process_model_spatiotemporal_responses
    n_cells : int
        Number of example cells to plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure
    """
    cell_metrics = processed_data['cell_metrics']
    input_metadata = processed_data['input_metadata']
    
    # Filter active cells
    active_cells = {gid: metrics for gid, metrics in cell_metrics.items() if metrics['n_spikes'] > 0}
    
    if not active_cells:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No active cells found", ha='center', va='center', fontsize=14)
        return fig
    
    dimensions = input_metadata['dimensions']
    input_signal = input_metadata['signal']
    
    # Sort cells by firing rate to get the most active cells
    sorted_cells = sorted(active_cells.items(), key=lambda x: x[1]['firing_rate'], reverse=True)
    example_cells = [gid for gid, _ in sorted_cells[:n_cells]]
    
    fig = plt.figure(figsize=(15, 4 * n_cells))
    
    # For each example cell, create a row of plots
    for i, gid in enumerate(example_cells):
        metrics = cell_metrics[gid]
        
        # spike raster
        ax1 = fig.add_subplot(n_cells, 3, i*3 + 1)
        
        spike_times = metrics['spike_times']
        if len(spike_times) > 0:
            ax1.vlines(spike_times, 0, 1, color='k')
            ax1.set_ylim(0, 1.2)
            ax1.set_yticks([])
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_title(f'Cell {gid} Spike Train')
        
        # ISI distribution
        ax2 = fig.add_subplot(n_cells, 3, i*3 + 2)
        
        isi = metrics['isi']
        if len(isi) > 0:
            ax2.hist(isi, bins=30, alpha=0.7)
            ax2.axvline(x=10, color='r', linestyle='--', label='10 ms')
            
            # Add burst index annotation
            burst_index = metrics['burst_index']
            ax2.text(0.7, 0.9, f'Burst index: {burst_index:.2f}', 
                    transform=ax2.transAxes, fontsize=10)
        
        ax2.set_xlabel('ISI (ms)')
        ax2.set_ylabel('Count')
        ax2.set_title('ISI Distribution')
        ax2.legend()
        
        # 2D tuning surface for key dimensions
        ax3 = fig.add_subplot(n_cells, 3, i*3 + 3)
        
        # Find key dimensions (preferably temporal and spatial)
        temporal_dim = next((name for name in dimensions if 'temporal' in name.lower() and 'frequency' not in name.lower()), None)
        freq_dim = next((name for name in dimensions if 'frequency' in name.lower()), None)
        spatial_dim = next((name for name in dimensions if 'spatial' in name.lower() and 'width' not in name.lower()), None)
        
        dim_x = freq_dim if freq_dim else (temporal_dim if temporal_dim else None)
        dim_y = spatial_dim if spatial_dim else (temporal_dim if temporal_dim != dim_x else None)
        
        if (dim_x and dim_y and 
            dim_x in metrics.get('spatiotemporal_responses', {}) and 
            dim_y in metrics.get('spatiotemporal_responses', {})):
            
            x_tuning = metrics['spatiotemporal_responses'][dim_x]
            y_tuning = metrics['spatiotemporal_responses'][dim_y]
            
            # Create meshgrid for 2D visualization
            x_vals = np.array(x_tuning['bin_centers'])
            y_vals = np.array(y_tuning['bin_centers'])
            
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = np.outer(y_tuning['responses'], x_tuning['responses'])
            
            im = ax3.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax3, label='Response')
            
            ax3.set_xlabel(dim_x)
            ax3.set_ylabel(dim_y)
            ax3.set_title(f'2D Tuning Surface')
            
            if dimensions[dim_x]['scale'] == 'log':
                ax3.set_xscale('log')
            if dimensions[dim_y]['scale'] == 'log':
                ax3.set_yscale('log')
        else:
            ax3.text(0.5, 0.5, "Insufficient tuning data", 
                    ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_dynamic_spatiotemporal_responses(processed_data):
    """
    Plot how spatiotemporal responses change over time.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed responses from process_model_spatiotemporal_responses
        
    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure
    """
    cell_metrics = processed_data['cell_metrics']
    input_metadata = processed_data['input_metadata']
    population_metrics = processed_data['population_metrics']
    
    duration = input_metadata['duration']
    
    fig = plt.figure(figsize=(15, 12))
    
    # population rate over time
    ax1 = fig.add_subplot(3, 1, 1)
    
    pop_rate = population_metrics['population_rate']
    bin_edges = population_metrics['bin_edges']
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ax1.plot(bin_centers, pop_rate)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Firing Rate (Hz/cell)')
    ax1.set_title('Population Firing Rate')
    
    # input signal if available
    if 'signal' in input_metadata and input_metadata['signal'] is not None:
        ax2 = fig.add_subplot(3, 1, 2)
        
        signal = input_metadata['signal']
        
        # If signal is 2D, plot first few channels
        if len(signal.shape) > 1:
            max_channels = min(5, signal.shape[1])
            for ch in range(max_channels):
                ax2.plot(signal[:, ch], label=f'Channel {ch}', alpha=0.7)
            ax2.legend()
        else:
            ax2.plot(signal)
        
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Input Signal')
    
    # spectrogram or time-frequency analysis of population response
    ax3 = fig.add_subplot(3, 1, 3)
    
    if len(pop_rate) > 50:  # Need enough data for spectrogram
        fs = 1000 / (bin_edges[1] - bin_edges[0])  # Hz
        
        f, t, Sxx = signal.spectrogram(pop_rate, fs=fs, nperseg=min(256, len(pop_rate)//4))
        
        im = ax3.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        plt.colorbar(im, ax=ax3, label='Power/Frequency (dB/Hz)')
        
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Population Response Spectrogram')
    else:
        ax3.text(0.5, 0.5, "Insufficient data for spectrogram", 
                ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    return fig


def analyze_spatiotemporal_responses(model_output_path,
                                     model_output_namespace_id,
                                     input_features_path,
                                     input_signal_id,
                                     populations: Optional[List] = None,
                                     time_range=None,
                                     time_variable="t",
                                     include_artificial=True,
                                     output_dir=None,
                                     analyses=['tuning_curves',
                                               'sensitivity_analysis',
                                               'receptive_fields',
                                               'response_examples',
                                               'dynamic_responses']):
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

    processed_data = process_model_spatiotemporal_responses(
        model_output_path = model_output_path,
        model_output_namespace_id = model_output_namespace_id,
        input_features_path = input_features_path,
        input_signal_id = input_signal_id,
        populations = populations,
        time_range=time_range,
        time_variable=time_variable,
        include_artificial=include_artificial
    )
    
    figures = {}
    
    input_metadata = processed_data['input_metadata']
    dimensions = input_metadata['dimensions']
    
    temporal_dim = next((name for name in dimensions if 'temporal' in name.lower() and 'frequency' not in name.lower()), None)
    freq_dim = next((name for name in dimensions if 'frequency' in name.lower()), None)
    spatial_dim = next((name for name in dimensions if 'spatial' in name.lower() and 'width' not in name.lower()), None)
    
    dim_x = freq_dim if freq_dim else (temporal_dim if temporal_dim else None)
    dim_y = spatial_dim if spatial_dim else (temporal_dim if temporal_dim != dim_x else None)
    
    if 'tuning_curves' in analyses:
        print("Generating dimensional tuning curves plot...")
        figures['dimensional_tuning'] = plot_dimensional_tuning_curves(processed_data)
        
        if dim_x and dim_y:
            print(f"Generating spatiotemporal tuning plot for {dim_x} vs {dim_y}...")
            figures['spatiotemporal_tuning'] = plot_spatiotemporal_tuning(
                processed_data, dim_x, dim_y)
    
    if 'sensitivity_analysis' in analyses:
        print("Generating feature sensitivity analysis plot...")
        figures['sensitivity_analysis'] = plot_feature_sensitivity_analysis(processed_data)
    
    if 'receptive_fields' in analyses:
        print("Generating spatial receptive fields plot...")
        figures['receptive_fields'] = plot_spatial_receptive_fields(processed_data)
    
    if 'response_examples' in analyses:
        print("Generating spatiotemporal response examples plot...")
        figures['response_examples'] = plot_spatiotemporal_response_examples(processed_data)
    
    if 'dynamic_responses' in analyses:
        print("Generating dynamic spatiotemporal responses plot...")
        figures['dynamic_responses'] = plot_dynamic_spatiotemporal_responses(processed_data)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in figures.items():
            fig.savefig(os.path.join(output_dir, f"{name}.svg"), dpi=600, bbox_inches='tight')
            plt.close(fig)
    
    return {
        'processed_data': processed_data,
        'figures': figures
    }

if __name__ == "__main__":
    # Test usage
    analyze_spatiotemporal_responses(
        model_output_path = "./results/Full_Scale_Spatiotemporal_Features/Full_Scale_results.h5",
        model_output_namespace_id = "Spike Events",
        input_features_path = "./input/EC_spatiotemporal_input_spike_trains.h5",
        input_signal_id = "test_spatiotemporal_features_20250515",
        populations = ["PYR"],
        include_artificial = False,
        output_dir="figures/spatiotemporal_analysis",
        analyses=['tuning_curves', 'sensitivity_analysis', 'response_examples', 'dynamic_responses']
    )

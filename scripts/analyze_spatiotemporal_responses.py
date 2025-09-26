import os
import sys
import warnings
from typing import List, Optional, Tuple, Dict, Any, TypeVar, Iterator, Callable
from collections import defaultdict
from functools import reduce
import random
from mpi4py import MPI
import numpy as np
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from collections import defaultdict
import warnings
from tqdm import tqdm
from neuroh5.io import read_cell_attributes, read_population_names, read_population_ranges
from miv_simulator import spikedata
import h5py
from drc import SpatioTemporalModality

K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type
T = TypeVar('T')  # Generic type


def reservoir_sample_dict(
    data: Dict[K, V], 
    n: int, 
    extract_fn: Callable[[K, V], T] = lambda k, v: (k, v),
    seed: Optional[int] = None
) -> List[T]:
    """
    Uniform reservoir sampling of n items from a dictionary.
    
    Args:
        data: Input dictionary to sample from
        n: Number of items to sample
        extract_fn: Function to extract desired value from (key, value) pair
                   Default returns (key, value) tuples
        seed: Random seed for reproducibility
    
    Returns:
        List of n sampled items (or fewer if dict has < n items)
    """
    if seed is not None:
        random.seed(seed)
    
    if n <= 0:
        return []
    
    items = list(data.items())
    if len(items) <= n:
        return [extract_fn(k, v) for k, v in items]
    
    def update_reservoir(state: Tuple[List[T], int], item: Tuple[K, V]) -> Tuple[List[T], int]:
        reservoir, i = state
        key, value = item
        
        if i < n:
            # Fill initial reservoir
            return reservoir + [extract_fn(key, value)], i + 1
        else:
            # Apply reservoir sampling rule
            j = random.randint(0, i)
            if j < n:
                new_reservoir = reservoir.copy()
                new_reservoir[j] = extract_fn(key, value)
                return new_reservoir, i + 1
            return reservoir, i + 1
    
    final_reservoir, _ = reduce(update_reservoir, items, ([], 0))
    return dict(final_reservoir)


def compute_feature_activity_timeseries(
    input_signal, 
    feature_data, 
    dimensions_info, 
    sample_rate=1000, 
    time_bin_ms=50.0
):
    """
    Compute activity time series for each input feature based on its spatio-temporal filter.
    
    Parameters:
    -----------
    input_signal : np.ndarray
        Input signal (time, channels)
    feature_data : dict
        Feature metadata from HDF5 file
    dimensions_info : dict
        Information about feature dimensions
    sample_rate : float
        Sampling rate in Hz
    time_bin_ms : float
        Time bin size in milliseconds for discretization
        
    Returns:
    --------
    dict
        {feature_gid: activity_timeseries} for each feature
    """
    if len(input_signal.shape) == 1:
        input_signal = input_signal.reshape(-1, 1)
    
    n_timepoints, n_channels = input_signal.shape
    duration_ms = (n_timepoints / sample_rate) * 1000
    n_time_bins = int(duration_ms / time_bin_ms)
    
    # Extract feature information
    gids = feature_data.get('gids', [])
    positions = feature_data.get('positions', [])
    
    feature_activities = {}
    
    # Get dimension indices
    dim_names = list(dimensions_info.keys())
    temp_pos_idx = dim_names.index('temporal_position') if 'temporal_position' in dim_names else 0
    temp_freq_idx = dim_names.index('temporal_frequency') if 'temporal_frequency' in dim_names else 1
    spatial_pos_idx = dim_names.index('spatial_position') if 'spatial_position' in dim_names else 2
    spatial_width_idx = dim_names.index('spatial_width') if 'spatial_width' in dim_names else 3

    spatio_temporal_modality = SpatioTemporalModality(input_shape=input_signal.shape,
                                                      sample_rate=sample_rate)

    for i, gid in enumerate(tqdm(gids, desc="Computing feature activities")):
        if i >= len(positions) or len(positions[i]) < 4:
            continue
            
        # Extract feature parameters
        pos = positions[i]
        temporal_position = pos[temp_pos_idx]  # 0-1, relative position in time
        temporal_frequency = pos[temp_freq_idx]  # Hz
        spatial_position = pos[spatial_pos_idx]  # 0-1, relative position in space
        spatial_width = pos[spatial_width_idx]  # 0-1, spatial tuning width
        
        # Create spatio-temporal filter
        input_filter = spatio_temporal_modality.create_input_filter(pos)
        
        activity = input_filter(input_signal)
        
        # Bin activity into time bins
        binned_activity = bin_timeseries(activity, time_bin_ms, sample_rate)
        
        feature_activities[int(gid)] = binned_activity
    
    return feature_activities



def bin_timeseries(timeseries, bin_size_ms, sample_rate):
    """
    Bin a continuous time series into discrete time bins.
    """
    samples_per_bin = int((bin_size_ms / 1000.0) * sample_rate)
    n_bins = len(timeseries) // samples_per_bin
    
    if n_bins == 0:
        return np.array([np.mean(timeseries)])
    
    # Reshape and average within bins
    truncated_length = n_bins * samples_per_bin
    reshaped = timeseries[:truncated_length].reshape(n_bins, samples_per_bin)
    binned = np.mean(reshaped, axis=1)
    
    return binned


def compute_spike_rate_timeseries(spike_times, duration_ms, time_bin_ms=50.0):
    """
    Convert spike times to binned firing rate time series.
    """
    n_bins = int(duration_ms / time_bin_ms)
    bin_edges = np.linspace(0, duration_ms, n_bins + 1)
    
    spike_counts, _ = np.histogram(spike_times, bins=bin_edges)
    
    # Convert to firing rate (Hz)
    firing_rates = spike_counts / (time_bin_ms / 1000.0)
    
    return firing_rates


def compute_signal_dimension_correlations(
    input_signal,
    neuron_spike_dict,
    duration_ms,
    sample_rate=1000,
    time_bin_ms=50.0,
    correlation_method='pearson',
    include_derivatives=True,
    include_frequency_bands=True,
    frequency_bands=[(1, 4), (4, 8), (8, 15), (15, 30), (30, 100)]
):
    """
    Compute correlations between neurons and input signal dimensions directly.
    This is much faster than correlating with individual features.
    
    Parameters:
    -----------
    input_signal : np.ndarray
        Input signal (time, channels)
    neuron_spike_dict : dict
        {neuron_gid: spike_times_array}
    duration_ms : float
        Total simulation duration in milliseconds
    sample_rate : float
        Sampling rate in Hz
    time_bin_ms : float
        Time bin size for correlation analysis
    correlation_method : str
        'pearson', 'spearman', or 'mutual_info'
    include_derivatives : bool
        Whether to include temporal derivatives
    include_frequency_bands : bool
        Whether to include frequency band decompositions
    frequency_bands : List[Tuple[float, float]]
        Frequency bands to extract (low_hz, high_hz)
        
    Returns:
    --------
    dict
        {neuron_gid: {signal_dimension: correlation_value}}
    """
    if len(input_signal.shape) == 1:
        input_signal = input_signal.reshape(-1, 1)
    
    n_timepoints, n_channels = input_signal.shape
    
    # Create all signal dimensions to correlate with
    signal_dimensions = {}
    
    # 1. Raw signal channels
    for ch in range(n_channels):
        binned_signal = bin_timeseries(input_signal[:, ch], time_bin_ms, sample_rate)
        signal_dimensions[f'channel_{ch}'] = binned_signal
    
    # 2. Temporal derivatives
    if include_derivatives:
        for ch in range(n_channels):
            signal_ch = input_signal[:, ch]
            
            # First derivative (velocity)
            derivative = np.gradient(signal_ch)
            binned_deriv = bin_timeseries(derivative, time_bin_ms, sample_rate)
            signal_dimensions[f'channel_{ch}_derivative'] = binned_deriv
            
            # Second derivative (acceleration) 
            second_deriv = np.gradient(derivative)
            binned_second_deriv = bin_timeseries(second_deriv, time_bin_ms, sample_rate)
            signal_dimensions[f'channel_{ch}_second_derivative'] = binned_second_deriv
    
    # 3. Spatial derivatives (if multichannel)
    if n_channels > 1:
        for t in range(0, n_timepoints, max(1, int(sample_rate * time_bin_ms / 1000))):
            spatial_profile = input_signal[t, :]
            spatial_grad = np.gradient(spatial_profile)
            # Store as single value per time bin
        
        # Compute spatial gradient over time
        spatial_gradients = []
        step_size = max(1, int(sample_rate * time_bin_ms / 1000))
        for t in range(0, n_timepoints, step_size):
            if t < n_timepoints:
                spatial_profile = input_signal[t, :]
                spatial_grad = np.gradient(spatial_profile)
                spatial_gradients.append(np.mean(spatial_grad))  # Mean spatial gradient
        
        signal_dimensions['spatial_gradient'] = np.array(spatial_gradients)
    
    # 4. Frequency band decompositions
    if include_frequency_bands and frequency_bands:
        from scipy import signal as sp_signal
        
        for ch in range(n_channels):
            signal_ch = input_signal[:, ch]
            
            for i, (low_freq, high_freq) in enumerate(frequency_bands):
                try:
                    # Create bandpass filter
                    nyquist = sample_rate / 2
                    low_norm = max(0.01, low_freq / nyquist)
                    high_norm = min(0.99, high_freq / nyquist)
                    
                    if low_norm < high_norm:
                        b, a = sp_signal.butter(2, [low_norm, high_norm], btype='band')
                        filtered = sp_signal.filtfilt(b, a, signal_ch)
                        
                        # Extract envelope (magnitude)
                        envelope = np.abs(sp_signal.hilbert(filtered))
                        binned_envelope = bin_timeseries(envelope, time_bin_ms, sample_rate)
                        
                        signal_dimensions[f'channel_{ch}_band_{low_freq}_{high_freq}Hz'] = binned_envelope
                        
                except Exception as e:
                    warnings.warn(f"Failed to compute frequency band {low_freq}-{high_freq}Hz: {e}")
    
    # 5. Global signal properties
    # RMS energy across all channels
    rms_energy = np.sqrt(np.mean(input_signal**2, axis=1))
    binned_rms = bin_timeseries(rms_energy, time_bin_ms, sample_rate)
    signal_dimensions['global_rms'] = binned_rms
    
    # Peak amplitude across channels
    peak_amplitude = np.max(np.abs(input_signal), axis=1)
    binned_peak = bin_timeseries(peak_amplitude, time_bin_ms, sample_rate)
    signal_dimensions['global_peak'] = binned_peak
    
    # Compute correlations with neurons
    correlations = {}
    
    # Convert spike trains to rate time series
    neuron_rates = {}
    for neuron_gid, spike_times in tqdm(neuron_spike_dict.items(), desc="Converting spikes to rates"):
        if len(spike_times) > 0:
            neuron_rates[neuron_gid] = compute_spike_rate_timeseries(
                spike_times, duration_ms, time_bin_ms
            )
        else:
            n_bins = int(duration_ms / time_bin_ms)
            neuron_rates[neuron_gid] = np.zeros(n_bins)
    
    # Compute correlations
    for neuron_gid, neuron_rate in tqdm(neuron_rates.items(), desc="Computing signal correlations"):
        correlations[neuron_gid] = {}
        
        for signal_name, signal_data in signal_dimensions.items():
            # Ensure same length
            min_length = min(len(neuron_rate), len(signal_data))
            neuron_data = neuron_rate[:min_length]
            signal_data_trimmed = signal_data[:min_length]
            
            # Skip if no activity
            if np.sum(neuron_data) == 0 or np.sum(np.abs(signal_data_trimmed)) == 0:
                correlations[neuron_gid][signal_name] = 0.0
                continue
            
            try:
                if correlation_method == 'pearson':
                    corr, p_val = pearsonr(neuron_data, signal_data_trimmed)
                    correlations[neuron_gid][signal_name] = corr if not np.isnan(corr) else 0.0
                elif correlation_method == 'spearman':
                    corr, p_val = spearmanr(neuron_data, signal_data_trimmed)
                    correlations[neuron_gid][signal_name] = corr if not np.isnan(corr) else 0.0
                elif correlation_method == 'mutual_info':
                    mi = mutual_info_regression(
                        neuron_data.reshape(-1, 1), 
                        signal_data_trimmed, 
                        discrete_features=False,
                        random_state=42
                    )[0]
                    correlations[neuron_gid][signal_name] = mi
                else:
                    raise ValueError(f"Unknown correlation method: {correlation_method}")
                    
            except Exception as e:
                warnings.warn(f"Correlation computation failed for neuron {neuron_gid}, signal {signal_name}: {e}")
                correlations[neuron_gid][signal_name] = 0.0
    
    return correlations


def compute_feature_neuron_correlations(
    feature_activities, 
    neuron_spike_dict, 
    duration_ms, 
    time_bin_ms=50.0,
    correlation_method='pearson'
):
    """
    Compute correlations between feature activities and neuron responses.
    
    Parameters:
    -----------
    feature_activities : dict
        {feature_gid: activity_timeseries} from compute_feature_activity_timeseries
    neuron_spike_dict : dict
        {neuron_gid: spike_times_array}
    duration_ms : float
        Total simulation duration in milliseconds
    time_bin_ms : float
        Time bin size for correlation analysis
    correlation_method : str
        'pearson', 'spearman', or 'mutual_info'
        
    Returns:
    --------
    dict
        {neuron_gid: {feature_gid: correlation_value}}
    """
    correlations = {}
    
    # Convert spike trains to rate time series
    neuron_rates = {}
    for neuron_gid, spike_times in tqdm(neuron_spike_dict.items(), desc="Converting spikes to rates"):
        if len(spike_times) > 0:
            neuron_rates[neuron_gid] = compute_spike_rate_timeseries(
                spike_times, duration_ms, time_bin_ms
            )
        else:
            # Handle silent neurons
            n_bins = int(duration_ms / time_bin_ms)
            neuron_rates[neuron_gid] = np.zeros(n_bins)
    
    # Compute correlations
    for neuron_gid, neuron_rate in tqdm(neuron_rates.items(), desc="Computing correlations"):
        correlations[neuron_gid] = {}
        
        for feature_gid, feature_activity in feature_activities.items():
            # Ensure same length
            min_length = min(len(neuron_rate), len(feature_activity))
            neuron_data = neuron_rate[:min_length]
            feature_data = feature_activity[:min_length]
            
            # Skip if no activity
            if np.sum(neuron_data) == 0 or np.sum(feature_data) == 0:
                correlations[neuron_gid][feature_gid] = 0.0
                continue
            
            try:
                if correlation_method == 'pearson':
                    corr, p_val = pearsonr(neuron_data, feature_data)
                    correlations[neuron_gid][feature_gid] = corr if not np.isnan(corr) else 0.0
                elif correlation_method == 'spearman':
                    corr, p_val = spearmanr(neuron_data, feature_data)
                    correlations[neuron_gid][feature_gid] = corr if not np.isnan(corr) else 0.0
                elif correlation_method == 'mutual_info':
                    # Use continuous MI estimator to avoid information loss
                    mi = mutual_info_regression(
                        neuron_data.reshape(-1, 1), 
                        feature_data, 
                        discrete_features=False,
                        random_state=42
                    )[0]
                    correlations[neuron_gid][feature_gid] = mi
                else:
                    raise ValueError(f"Unknown correlation method: {correlation_method}")
                    
            except Exception as e:
                warnings.warn(f"Correlation computation failed for neuron {neuron_gid}, feature {feature_gid}: {e}")
                correlations[neuron_gid][feature_gid] = 0.0
    
    return correlations


def build_dimensional_receptive_fields(
    correlations, 
    feature_data, 
    dimensions_info, 
    min_features_per_bin=3
):
    """
    Build receptive fields along each dimension from correlation data.
    
    Parameters:
    -----------
    correlations : dict
        {neuron_gid: {feature_gid: correlation}} from compute_feature_neuron_correlations
    feature_data : dict
        Feature metadata
    dimensions_info : dict
        Information about dimensions
    min_features_per_bin : int
        Minimum number of features required per bin
        
    Returns:
    --------
    dict
        {neuron_gid: {dim_name: {'bin_centers': array, 'responses': array, 'n_features': array}}}
    """
    receptive_fields = {}
    
    # Extract feature information
    gids = feature_data.get('gids', [])
    positions = feature_data.get('positions', [])
    
    # Create feature position lookup
    feature_positions = {}
    for i, gid in enumerate(gids):
        if i < len(positions) and len(positions[i]) >= 4:
            feature_positions[int(gid)] = positions[i]
    
    dim_names = list(dimensions_info.keys())
    
    for neuron_gid, neuron_correlations in tqdm(correlations.items(), desc="Building receptive fields"):
        receptive_fields[neuron_gid] = {}
        
        # Get valid features for this neuron (those with correlations)
        valid_features = [(fgid, corr) for fgid, corr in neuron_correlations.items() 
                         if fgid in feature_positions]
        
        if len(valid_features) < min_features_per_bin:
            continue
            
        for dim_idx, dim_name in enumerate(dim_names):
            # Extract dimension values and correlations
            dim_values = []
            dim_correlations = []
            
            for feature_gid, correlation in valid_features:
                pos = feature_positions[feature_gid]
                if dim_idx < len(pos):
                    dim_values.append(pos[dim_idx])
                    dim_correlations.append(correlation)
            
            if len(dim_values) < min_features_per_bin:
                continue
            
            # Create bins for this dimension
            dim_range = dimensions_info[dim_name]['range']
            scale = dimensions_info[dim_name].get('scale', 'linear')
            
            # Adaptive number of bins based on data
            n_bins = min(10, max(3, len(dim_values) // min_features_per_bin))
            
            if scale == 'log' and dim_range[0] > 0:
                bin_edges = np.logspace(np.log10(dim_range[0]), np.log10(dim_range[1]), n_bins + 1)
            else:
                bin_edges = np.linspace(dim_range[0], dim_range[1], n_bins + 1)
            
            # Bin the correlations
            binned_responses = []
            binned_counts = []
            bin_centers = []
            
            for i in range(n_bins):
                # Find features in this bin
                mask = (np.array(dim_values) >= bin_edges[i]) & (np.array(dim_values) < bin_edges[i + 1])
                if i == n_bins - 1:  # Include right edge in last bin
                    mask = (np.array(dim_values) >= bin_edges[i]) & (np.array(dim_values) <= bin_edges[i + 1])
                
                bin_corrs = np.array(dim_correlations)[mask]
                
                if len(bin_corrs) >= 1:  # At least 1 feature
                    binned_responses.append(np.mean(bin_corrs))
                    binned_counts.append(len(bin_corrs))
                else:
                    binned_responses.append(0.0)
                    binned_counts.append(0)
                
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            
            receptive_fields[neuron_gid][dim_name] = {
                'bin_centers': np.array(bin_centers),
                'responses': np.array(binned_responses),
                'n_features': np.array(binned_counts),
                'bin_edges': bin_edges
            }
    
    return receptive_fields


def compute_binned_feature_correlations(
    feature_activities,
    feature_data,
    dimensions_info,
    neuron_spike_dict,
    duration_ms,
    time_bin_ms=1.0,
    correlation_method='pearson',
    n_bins_per_dim=5
):
    """
    Compute correlations with binned feature activities to reduce computational complexity.
    
    Features are binned along each of the four dimensions, then activities are averaged
    within each bin, and correlations are computed with these averaged activities.
    
    Parameters:
    -----------
    feature_activities : dict
        {feature_gid: activity_timeseries} from compute_feature_activity_timeseries
    feature_data : dict
        Feature metadata
    dimensions_info : dict
        Information about feature dimensions
    neuron_spike_dict : dict
        {neuron_gid: spike_times_array}
    duration_ms : float
        Total simulation duration in milliseconds
    time_bin_ms : float
        Time bin size for correlation analysis
    correlation_method : str
        'pearson', 'spearman', or 'mutual_info'
    n_bins_per_dim : int
        Number of bins per dimension
        
    Returns:
    --------
    tuple
        (correlations_dict, binned_activities_dict, bin_info_dict)
    """
    # Extract feature information
    gids = feature_data.get('gids', [])
    positions = feature_data.get('positions', [])
    
    # Create feature position lookup
    feature_positions = {}
    for i, gid in enumerate(gids):
        if i < len(positions) and len(positions[i]) >= 4:
            feature_positions[int(gid)] = positions[i]
    
    # Get valid features (those with both positions and activities)
    valid_features = []
    for gid in feature_positions:
        if gid in feature_activities:
            valid_features.append(gid)
    
    if len(valid_features) == 0:
        return {}, {}, {}
    
    print(f"Binning {len(valid_features)} features across {len(dimensions_info)} dimensions...")
    
    dim_names = list(dimensions_info.keys())
    
    # Create bins for each dimension
    bin_info = {}
    for dim_idx, dim_name in enumerate(dim_names):
        dim_range = dimensions_info[dim_name]['range']
        scale = dimensions_info[dim_name].get('scale', 'linear')
        
        if scale == 'log' and dim_range[0] > 0:
            bin_edges = np.logspace(np.log10(dim_range[0]), np.log10(dim_range[1]), n_bins_per_dim + 1)
        else:
            bin_edges = np.linspace(dim_range[0], dim_range[1], n_bins_per_dim + 1)
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        bin_info[dim_name] = {
            'bin_edges': bin_edges,
            'bin_centers': bin_centers,
            'dim_idx': dim_idx,
            'n_bins': n_bins_per_dim
        }
    
    # Assign features to bins for each dimension
    feature_bins = {}
    for dim_name in dim_names:
        feature_bins[dim_name] = [[] for _ in range(n_bins_per_dim)]
    
    for gid in valid_features:
        pos = feature_positions[gid]
        
        for dim_name in dim_names:
            dim_idx = bin_info[dim_name]['dim_idx']
            if dim_idx < len(pos):
                value = pos[dim_idx]
                bin_edges = bin_info[dim_name]['bin_edges']
                
                # Find which bin this feature belongs to
                bin_idx = np.digitize(value, bin_edges) - 1
                bin_idx = max(0, min(bin_idx, n_bins_per_dim - 1))  # Clamp to valid range
                
                feature_bins[dim_name][bin_idx].append(gid)
    
    # Compute averaged activities for each bin
    binned_activities = {}
    
    for dim_name in dim_names:
        binned_activities[dim_name] = {}
        
        for bin_idx in range(n_bins_per_dim):
            features_in_bin = feature_bins[dim_name][bin_idx]
            
            if len(features_in_bin) == 0:
                # Empty bin - create zero activity
                example_activity = next(iter(feature_activities.values()))
                binned_activities[dim_name][bin_idx] = np.zeros_like(example_activity)
            else:
                # Average activities of features in this bin
                activities_in_bin = [feature_activities[gid] for gid in features_in_bin]
                
                # Ensure all have same length
                min_length = min(len(act) for act in activities_in_bin)
                trimmed_activities = [act[:min_length] for act in activities_in_bin]
                
                # Compute average
                avg_activity = np.mean(trimmed_activities, axis=0)
                binned_activities[dim_name][bin_idx] = avg_activity
    
    # Convert spike trains to rate time series
    neuron_rates = {}
    for neuron_gid, spike_times in tqdm(neuron_spike_dict.items(), desc="Converting spikes to rates"):
        if len(spike_times) > 0:
            neuron_rates[neuron_gid] = compute_spike_rate_timeseries(
                spike_times, duration_ms, time_bin_ms
            )
        else:
            n_bins = int(duration_ms / time_bin_ms)
            neuron_rates[neuron_gid] = np.zeros(n_bins)
    
    # Compute correlations with binned activities
    correlations = {}
    
    for neuron_gid, neuron_rate in tqdm(neuron_rates.items(), desc="Computing binned correlations"):
        correlations[neuron_gid] = {}
        
        for dim_name in dim_names:
            correlations[neuron_gid][dim_name] = {}
            
            for bin_idx in range(n_bins_per_dim):
                bin_activity = binned_activities[dim_name][bin_idx]
                
                # Ensure same length
                min_length = min(len(neuron_rate), len(bin_activity))
                neuron_data = neuron_rate[:min_length]
                bin_data = bin_activity[:min_length]
                
                # Skip if no activity
                if np.sum(neuron_data) == 0 or np.sum(np.abs(bin_data)) == 0:
                    correlations[neuron_gid][dim_name][bin_idx] = 0.0
                    continue
                
                try:
                    if correlation_method == 'pearson':
                        corr, p_val = pearsonr(neuron_data, bin_data)
                        correlations[neuron_gid][dim_name][bin_idx] = corr if not np.isnan(corr) else 0.0
                    elif correlation_method == 'spearman':
                        corr, p_val = spearmanr(neuron_data, bin_data)
                        correlations[neuron_gid][dim_name][bin_idx] = corr if not np.isnan(corr) else 0.0
                    elif correlation_method == 'mutual_info':
                        mi = mutual_info_regression(
                            neuron_data.reshape(-1, 1), 
                            bin_data, 
                            discrete_features=False,
                            random_state=42
                        )[0]
                        correlations[neuron_gid][dim_name][bin_idx] = mi
                    else:
                        raise ValueError(f"Unknown correlation method: {correlation_method}")
                        
                except Exception as e:
                    warnings.warn(f"Correlation computation failed for neuron {neuron_gid}, dim {dim_name}, bin {bin_idx}: {e}")
                    correlations[neuron_gid][dim_name][bin_idx] = 0.0
    
    return correlations, binned_activities, bin_info


def build_binned_receptive_fields(correlations, bin_info):
    """
    Build receptive fields from binned correlation data.
    
    Parameters:
    -----------
    correlations : dict
        {neuron_gid: {dim_name: {bin_idx: correlation}}} from compute_binned_feature_correlations
    bin_info : dict
        Bin information from compute_binned_feature_correlations
        
    Returns:
    --------
    dict
        {neuron_gid: {dim_name: {'bin_centers': array, 'responses': array}}}
    """
    receptive_fields = {}
    
    for neuron_gid, neuron_correlations in tqdm(correlations.items(), desc="Building binned receptive fields"):
        receptive_fields[neuron_gid] = {}
        
        for dim_name, dim_correlations in neuron_correlations.items():
            if dim_name in bin_info:
                bin_centers = bin_info[dim_name]['bin_centers']
                n_bins = bin_info[dim_name]['n_bins']
                
                # Extract correlations in bin order
                responses = []
                for bin_idx in range(n_bins):
                    responses.append(dim_correlations.get(bin_idx, 0.0))
                
                receptive_fields[neuron_gid][dim_name] = {
                    'bin_centers': bin_centers,
                    'responses': np.array(responses),
                    'n_bins': n_bins,
                    'bin_edges': bin_info[dim_name]['bin_edges']
                }
    
    return receptive_fields


def process_model_spatiotemporal_responses(
    model_output_path,
    model_output_namespace_id,
    input_features_path,
    input_signal_id,
    populations=None,
    time_range=None,
    time_variable="t",
    include_artificial=True,
    time_bin_ms=50.0,
    correlation_method='pearson',
    max_gids=None,
    sample_seed=None,
    use_signal_dimensions=False,  # Fast signal correlation mode
    use_binned_features=False,    # Binned feature correlation mode
    n_bins_per_dim=10,             # Number of bins per dimension
    include_derivatives=True,
    include_frequency_bands=True,
    frequency_bands=[(1, 4), (4, 8), (8, 15), (15, 30), (30, 100)],
    **kwargs
):
    """
    Characterize spatio-temporal responses using correlation analysis.
    
    This function offers three computational approaches:
    1. Feature-based: Correlates with individual features (detailed but slow)
    2. Signal dimension: Correlates with signal processing properties (fast but less specific)
    3. Binned feature: Correlates with averaged activities in feature space bins (balanced)
    
    Parameters:
    -----------
    model_output_path : str
        Path to model output HDF5 file
    model_output_namespace_id : str
        Namespace ID for spike events
    input_features_path : str
        Path to input features HDF5 file
    input_signal_id : str
        ID of input signal
    time_bin_ms : float
        Time bin size for correlation analysis (milliseconds)
    correlation_method : str
        'pearson', 'spearman', or 'mutual_info'
    max_gids : int, optional
        Maximum number of neurons to analyze (for computational efficiency)
    sample_seed : int, optional
        Random seed for neuron sampling
    use_signal_dimensions : bool
        If True, correlate directly with signal dimensions instead of features (fastest)
    use_binned_features : bool
        If True, bin features and correlate with averaged activities (balanced approach)
    n_bins_per_dim : int
        Number of bins per dimension for binned feature approach
    include_derivatives : bool
        Whether to include temporal derivatives in signal dimension analysis
    include_frequency_bands : bool
        Whether to include frequency band decompositions
    frequency_bands : List[Tuple[float, float]]
        Frequency bands to extract for signal dimension analysis
        
    Returns:
    --------
    dict
        Processed responses with correlation-based receptive fields
        
    Note:
    ----
    Computational complexity comparison:
    - Feature-based: O(N_neurons × N_features × N_timebins) - most detailed
    - Binned features: O(N_neurons × N_bins_per_dim × N_dimensions × N_timebins) - balanced
    - Signal dimensions: O(N_neurons × N_signal_dims × N_timebins) - fastest
    """
    
    (population_ranges, N) = read_population_ranges(model_output_path)
    population_names = read_population_names(model_output_path)
    
    if populations is None:
        include = list(population_names)
    else:
        include = [populations] if isinstance(populations, str) else populations
    
    include.sort(key=lambda x: population_ranges[x][0])
    
    # Load spike data
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
    tmin = spkdata["tmin"]
    tmax = spkdata["tmax"]
    simulation_duration = tmax - tmin
    
    pop_spk_dict = {
        pop_name: (pop_spkinds, pop_spkts)
        for (pop_name, pop_spkinds, pop_spkts) in zip(spkpoplst, spkindlst, spktlst)
    }
    
    # Load input signal and features
    with h5py.File(input_features_path, 'r') as f:
        signal_group = f[f'Signals/{input_signal_id}']
        
        if 'data' in signal_group:
            input_signal = signal_group['data'][:]
        elif 'stimulus' in signal_group:
            input_signal = signal_group['stimulus'][:]
        else:
            raise RuntimeError(f"Unable to find signal data in group {signal_group}")
        
        dimensions = signal_group['dimensions'][:]
        
        dim_info = {}
        for dim in dimensions:
            name = dim['name'].decode('utf-8') if isinstance(dim['name'], bytes) else dim['name']
            dim_info[name] = {
                'range': (dim['range_min'], dim['range_max']),
                'scale': dim['scale'].decode('utf-8') if isinstance(dim['scale'], bytes) else dim['scale'],
                'priority': dim['priority']
            }
        
        feature_data = {}
        if 'feature_data' in signal_group:
            feature_group = signal_group['feature_data']
            for key in feature_group.keys():
                feature_data[key] = feature_group[key][:]
        
        duration = signal_group.attrs.get('duration', 10.0)
        sample_rate = signal_group.attrs.get('sample_rate', 1000)
        n_channels = signal_group.attrs.get('n_channels', 
                                           input_signal.shape[1] if len(input_signal.shape) > 1 else 1)
    
    # Choose correlation approach based on parameters
    if use_signal_dimensions and use_binned_features:
        raise ValueError("Cannot use both use_signal_dimensions and use_binned_features simultaneously")
    
    if use_signal_dimensions:
        print("Using fast signal dimension correlation approach...")
        feature_activities = None  # Not needed for signal dimension approach
        
        # Process each population
        processed_responses = {}
        
        for pop_name in include:
            if pop_name not in pop_spk_dict:
                continue
            
            print(f"Processing population: {pop_name}")
            
            pop_spkinds, pop_spkts = pop_spk_dict[pop_name]
            gid_spike_dict = spikedata.make_spike_dict(pop_spkinds, pop_spkts)
            
            # Sample neurons if requested
            if max_gids is not None and len(gid_spike_dict) > max_gids:
                if sample_seed is not None:
                    np.random.seed(sample_seed)
                sampled_gids = np.random.choice(
                    list(gid_spike_dict.keys()), 
                    size=max_gids, 
                    replace=False
                )
                gid_spike_dict = {gid: gid_spike_dict[gid] for gid in sampled_gids}
            
            print("Computing signal dimension correlations...")
            correlations = compute_signal_dimension_correlations(
                input_signal,
                gid_spike_dict,
                simulation_duration,
                sample_rate,
                time_bin_ms,
                correlation_method,
                include_derivatives,
                include_frequency_bands,
                frequency_bands
            )
            
            # Build simple "receptive fields" based on signal dimensions
            print("Building signal dimension receptive fields...")
            receptive_fields = {}
            for neuron_gid, neuron_correlations in correlations.items():
                receptive_fields[neuron_gid] = {}
                
                # Group correlations by signal type
                for signal_name, correlation in neuron_correlations.items():
                    if 'channel_' in signal_name and not any(x in signal_name for x in ['derivative', 'band']):
                        # Raw channel responses
                        if 'raw_channels' not in receptive_fields[neuron_gid]:
                            receptive_fields[neuron_gid]['raw_channels'] = {
                                'signal_names': [],
                                'responses': []
                            }
                        receptive_fields[neuron_gid]['raw_channels']['signal_names'].append(signal_name)
                        receptive_fields[neuron_gid]['raw_channels']['responses'].append(correlation)
                    
                    elif 'band_' in signal_name:
                        # Frequency band responses
                        if 'frequency_bands' not in receptive_fields[neuron_gid]:
                            receptive_fields[neuron_gid]['frequency_bands'] = {
                                'signal_names': [],
                                'responses': []
                            }
                        receptive_fields[neuron_gid]['frequency_bands']['signal_names'].append(signal_name)
                        receptive_fields[neuron_gid]['frequency_bands']['responses'].append(correlation)
                    
                    elif 'derivative' in signal_name:
                        # Temporal derivative responses
                        if 'temporal_derivatives' not in receptive_fields[neuron_gid]:
                            receptive_fields[neuron_gid]['temporal_derivatives'] = {
                                'signal_names': [],
                                'responses': []
                            }
                        receptive_fields[neuron_gid]['temporal_derivatives']['signal_names'].append(signal_name)
                        receptive_fields[neuron_gid]['temporal_derivatives']['responses'].append(correlation)
            
            # Store processed data for this population
            processed_responses[pop_name] = _build_population_response_dict(
                gid_spike_dict, pop_spkts, simulation_duration, correlations, 
                receptive_fields, input_signal, dim_info, feature_data, 
                duration, sample_rate, n_channels, correlation_method, time_bin_ms,
                feature_activities
            )
    
    elif use_binned_features:
        print("Using binned feature correlation approach...")
        print("Computing feature activity time series...")
        feature_activities = compute_feature_activity_timeseries(
            input_signal, feature_data, dim_info, sample_rate, time_bin_ms
        )
        
        processed_responses = {}
        
        for pop_name in include:
            if pop_name not in pop_spk_dict:
                continue
            
            print(f"Processing population: {pop_name}")
            
            pop_spkinds, pop_spkts = pop_spk_dict[pop_name]
            gid_spike_dict = spikedata.make_spike_dict(pop_spkinds, pop_spkts)
            
            # Sample neurons if requested
            if max_gids is not None and len(gid_spike_dict) > max_gids:
                if sample_seed is not None:
                    np.random.seed(sample_seed)
                sampled_gids = np.random.choice(
                    list(gid_spike_dict.keys()), 
                    size=max_gids, 
                    replace=False
                )
                gid_spike_dict = {gid: gid_spike_dict[gid] for gid in sampled_gids}
            
            print("Computing binned feature correlations...")
            correlations, binned_activities, bin_info = compute_binned_feature_correlations(
                feature_activities,
                feature_data,
                dim_info,
                gid_spike_dict,
                simulation_duration,
                time_bin_ms,
                correlation_method,
                n_bins_per_dim
            )
            
            print("Building binned receptive fields...")
            receptive_fields = build_binned_receptive_fields(correlations, bin_info)
            
            # Store processed data for this population (need to flatten correlations for compatibility)
            flattened_correlations = {}
            for neuron_gid, neuron_correlations in correlations.items():
                flattened_correlations[neuron_gid] = {}
                for dim_name, dim_correlations in neuron_correlations.items():
                    for bin_idx, correlation in dim_correlations.items():
                        bin_key = f"{dim_name}_bin_{bin_idx}"
                        flattened_correlations[neuron_gid][bin_key] = correlation
            
            processed_responses[pop_name] = _build_population_response_dict(
                gid_spike_dict, pop_spkts, simulation_duration, flattened_correlations, 
                receptive_fields, input_signal, dim_info, feature_data, 
                duration, sample_rate, n_channels, correlation_method, time_bin_ms,
                feature_activities
            )
            
            # Add binned activity info to the response
            processed_responses[pop_name]['binned_activities'] = binned_activities
            processed_responses[pop_name]['bin_info'] = bin_info
        
    else:
        print("Using detailed feature-based correlation approach...")
        print("Computing feature activity time series...")
        feature_activities = compute_feature_activity_timeseries(
            input_signal, feature_data, dim_info, sample_rate, time_bin_ms
        )
        
        processed_responses = {}
        
        for pop_name in include:
            if pop_name not in pop_spk_dict:
                continue
            
            print(f"Processing population: {pop_name}")
            
            pop_spkinds, pop_spkts = pop_spk_dict[pop_name]
            gid_spike_dict = spikedata.make_spike_dict(pop_spkinds, pop_spkts)
            
            # Sample neurons if requested
            if max_gids is not None and len(gid_spike_dict) > max_gids:
                if sample_seed is not None:
                    np.random.seed(sample_seed)
                sampled_gids = np.random.choice(
                    list(gid_spike_dict.keys()), 
                    size=max_gids, 
                    replace=False
                )
                gid_spike_dict = {gid: gid_spike_dict[gid] for gid in sampled_gids}
            
            print("Computing feature-neuron correlations...")
            correlations = compute_feature_neuron_correlations(
                feature_activities, 
                gid_spike_dict, 
                simulation_duration, 
                time_bin_ms,
                correlation_method
            )
            
            print("Building dimensional receptive fields...")
            receptive_fields = build_dimensional_receptive_fields(
                correlations, 
                feature_data, 
                dim_info
            )
            
            # Store processed data for this population
            processed_responses[pop_name] = _build_population_response_dict(
                gid_spike_dict, pop_spkts, simulation_duration, correlations, 
                receptive_fields, input_signal, dim_info, feature_data, 
                duration, sample_rate, n_channels, correlation_method, time_bin_ms,
                feature_activities
            )
    
    if len(processed_responses) == 1:
        return next(iter(processed_responses.values()))
    else:
        return processed_responses


def _build_population_response_dict(
    gid_spike_dict, pop_spkts, simulation_duration, correlations, 
    receptive_fields, input_signal, dim_info, feature_data, 
    duration, sample_rate, n_channels, correlation_method, time_bin_ms,
    feature_activities
):
    """Helper function to build the population response dictionary."""
    
    # Compute basic cell metrics
    cell_metrics = {}
    for gid, spike_times in gid_spike_dict.items():
        spike_times = np.array(spike_times)
        n_spikes = len(spike_times)
        firing_rate = n_spikes / (simulation_duration / 1000)
        
        # ISI metrics
        if n_spikes > 1:
            isi = np.diff(spike_times)
            cv_isi = np.std(isi) / np.mean(isi) if np.mean(isi) > 0 else 0
            burst_index = np.sum(isi < 10) / len(isi)  # fraction of ISIs < 10ms
        else:
            cv_isi = 0
            burst_index = 0
        
        # Store metrics with correlation-based receptive fields
        cell_metrics[gid] = {
            'firing_rate': firing_rate,
            'n_spikes': n_spikes,
            'spike_times': spike_times,
            'cv_isi': cv_isi,
            'burst_index': burst_index,
            'feature_correlations': correlations.get(gid, {}),
            'receptive_fields': receptive_fields.get(gid, {}),
            'max_correlation': max(correlations.get(gid, {}).values()) if correlations.get(gid) else 0,
            'mean_correlation': np.mean(list(correlations.get(gid, {}).values())) if correlations.get(gid) else 0,
        }
    
    # Population-level metrics
    all_spikes = np.array(sorted(pop_spkts))
    bin_size = 10  # ms
    n_bins = int(simulation_duration / bin_size)
    pop_rate, pop_bin_edges = np.histogram(
        all_spikes, bins=n_bins, range=(0, simulation_duration)
    )
    
    n_cells = len(gid_spike_dict)
    if n_cells > 0:
        pop_rate = pop_rate / (bin_size/1000) / n_cells
    
    return {
        'input_metadata': {
            'signal': input_signal,
            'dimensions': dim_info,
            'feature_data': feature_data,
            'duration': duration,
            'sample_rate': sample_rate,
            'n_channels': n_channels
        },
        'population_metrics': {
            'all_spikes': all_spikes,
            'population_rate': pop_rate,
            'bin_edges': pop_bin_edges,
            'mean_rate': np.mean(pop_rate) if len(pop_rate) > 0 else 0,
            'n_active_cells': sum(1 for metrics in cell_metrics.values() if metrics['n_spikes'] > 0),
            'total_cells': n_cells,
            'correlation_method': correlation_method,
            'time_bin_ms': time_bin_ms,
        },
        'cell_metrics': cell_metrics,
        'feature_activities': feature_activities,  # Store for debugging/analysis
    }

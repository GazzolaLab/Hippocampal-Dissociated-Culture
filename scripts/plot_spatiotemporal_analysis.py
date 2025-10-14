import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Set consistent plotting style
plt.style.use('ggplot')
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


def plot_feature_activities(
    feature_activities: Dict[int, np.ndarray],
    feature_data: Dict,
    dimensions_info: Dict,
    time_bin_ms: float = 1.0,
    max_features: int = 20,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot time series of feature activities to visualize spatio-temporal dynamics.
    
    Parameters:
    -----------
    feature_activities : Dict[int, np.ndarray]
        Feature activity time series from compute_feature_activity_timeseries
    feature_data : Dict
        Feature metadata
    dimensions_info : Dict
        Information about feature dimensions
    time_bin_ms : float
        Time bin size in milliseconds
    max_features : int
        Maximum number of features to plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    # Sample features for visualization
    feature_gids = list(feature_activities.keys())[:max_features]
    
    # Get feature positions for sorting
    gids = feature_data.get('gids', [])
    positions = feature_data.get('positions', [])
    
    feature_positions = {}
    for i, gid in enumerate(gids):
        if i < len(positions) and len(positions[i]) >= 4:
            feature_positions[int(gid)] = positions[i]
    
    # Sort features by temporal frequency for better visualization
    if 'temporal_frequency' in dimensions_info:
        freq_idx = list(dimensions_info.keys()).index('temporal_frequency')
        feature_gids.sort(key=lambda gid: feature_positions.get(gid, [0,0,0,0])[freq_idx])
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
    
    # Main time series plot
    ax_main = fig.add_subplot(gs[0])
    
    n_features = len(feature_gids)
    time_axis = np.arange(len(feature_activities[feature_gids[0]])) * time_bin_ms
    
    # Create colormap for features
    colors = plt.cm.viridis(np.linspace(0, 1, n_features))
    
    for i, gid in enumerate(feature_gids):
        activity = feature_activities[gid]
        ax_main.plot(time_axis, activity + i * 1.2, color=colors[i], 
                    linewidth=1.5, alpha=0.8, label=f'Feature {gid}')
    
    ax_main.set_xlabel('Time (ms)')
    ax_main.set_ylabel('Feature Activity (offset)')
    ax_main.set_title('Feature Activity Time Series')
    ax_main.grid(True, alpha=0.3)
    
    # Summary statistics plots
    # Temporal frequency distribution
    ax_freq = fig.add_subplot(gs[1])
    if 'temporal_frequency' in dimensions_info:
        freq_idx = list(dimensions_info.keys()).index('temporal_frequency')
        frequencies = [feature_positions.get(gid, [0,0,0,0])[freq_idx] for gid in feature_gids]
        ax_freq.hist(frequencies, bins=10, alpha=0.7, color='steelblue')
        ax_freq.set_xlabel('Temporal Frequency (Hz)')
        ax_freq.set_ylabel('Count')
        ax_freq.set_title('Feature Frequency Distribution')
        
        # Log scale if appropriate
        if dimensions_info['temporal_frequency'].get('scale') == 'log':
            ax_freq.set_xscale('log')
    
    # Spatial position distribution
    ax_spatial = fig.add_subplot(gs[2])
    if 'spatial_position' in dimensions_info:
        spatial_idx = list(dimensions_info.keys()).index('spatial_position')
        positions_spatial = [feature_positions.get(gid, [0,0,0,0])[spatial_idx] for gid in feature_gids]
        ax_spatial.hist(positions_spatial, bins=10, alpha=0.7, color='orange')
        ax_spatial.set_xlabel('Spatial Position')
        ax_spatial.set_ylabel('Count')
        ax_spatial.set_title('Feature Spatial Distribution')
    
    # Activity magnitude distribution
    ax_mag = fig.add_subplot(gs[3])
    all_activities = np.concatenate([feature_activities[gid] for gid in feature_gids])
    ax_mag.hist(all_activities, bins=50, alpha=0.7, color='green')
    ax_mag.set_xlabel('Activity Magnitude')
    ax_mag.set_ylabel('Count')
    ax_mag.set_title('Activity Magnitude Distribution')
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    correlations: Dict[int, Dict[int, float]], 
    max_neurons: int = 50,
    max_features: int = 100,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot correlation matrix between neurons and features.
    
    Parameters:
    -----------
    correlations : Dict[int, Dict[int, float]]
        {neuron_gid: {feature_gid: correlation}} from compute_feature_neuron_correlations
    max_neurons : int
        Maximum number of neurons to display
    max_features : int
        Maximum number of features to display
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    # Sample neurons and features
    neuron_gids = list(correlations.keys())[:max_neurons]
    
    # Get all feature GIDs
    all_feature_gids = set()
    for neuron_corrs in correlations.values():
        all_feature_gids.update(neuron_corrs.keys())
    feature_gids = list(all_feature_gids)[:max_features]
    
    # Build correlation matrix
    corr_matrix = np.zeros((len(neuron_gids), len(feature_gids)))
    
    for i, neuron_gid in enumerate(neuron_gids):
        for j, feature_gid in enumerate(feature_gids):
            corr_matrix[i, j] = correlations.get(neuron_gid, {}).get(feature_gid, 0.0)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Main correlation heatmap
    ax_main = axes[0, 0]
    im = ax_main.imshow(corr_matrix, aspect='auto', cmap='RdBu_r', 
                       vmin=-1, vmax=1, interpolation='nearest')
    
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Correlation')
    
    ax_main.set_xlabel('Feature Index')
    ax_main.set_ylabel('Neuron Index')
    ax_main.set_title('Neuron-Feature Correlation Matrix')
    
    # Correlation distribution
    ax_dist = axes[0, 1]
    corr_values = corr_matrix.flatten()
    ax_dist.hist(corr_values, bins=50, alpha=0.7, color='steelblue')
    ax_dist.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax_dist.set_xlabel('Correlation Value')
    ax_dist.set_ylabel('Count')
    ax_dist.set_title('Correlation Distribution')
    
    # Max correlation per neuron
    ax_max = axes[1, 0]
    max_corrs = np.max(np.abs(corr_matrix), axis=1)
    ax_max.bar(range(len(max_corrs)), max_corrs, alpha=0.7, color='orange')
    ax_max.set_xlabel('Neuron Index')
    ax_max.set_ylabel('Max |Correlation|')
    ax_max.set_title('Maximum Correlation per Neuron')
    
    # Mean correlation per feature
    ax_mean = axes[1, 1]
    mean_corrs = np.mean(np.abs(corr_matrix), axis=0)
    ax_mean.bar(range(len(mean_corrs)), mean_corrs, alpha=0.7, color='green')
    ax_mean.set_xlabel('Feature Index')
    ax_mean.set_ylabel('Mean |Correlation|')
    ax_mean.set_title('Mean Correlation per Feature')
    
    plt.tight_layout()
    return fig


def plot_dimensional_receptive_fields(
    processed_data: Dict,
    neuron_gids: Optional[List[int]] = None,
    max_neurons: int = 9,
    figsize: Tuple[int, int] = (15, 12),
    analysis_method: str = 'auto'
) -> plt.Figure:
    """
    Plot dimensional receptive fields for selected neurons.
    Automatically detects and handles different analysis methods.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed data from process_model_spatiotemporal_responses_improved
    neuron_gids : Optional[List[int]]
        Specific neurons to plot, or None for automatic selection
    max_neurons : int
        Maximum number of neurons to plot
    figsize : Tuple[int, int]
        Figure size
    analysis_method : str
        'auto', 'feature_based', 'signal_dimensions', or 'binned_features'
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    cell_metrics = processed_data['cell_metrics']
    
    # Auto-detect analysis method
    if analysis_method == 'auto':
        analysis_method = _detect_analysis_method(processed_data)
    
    # Select neurons to plot
    if neuron_gids is None:
        neurons_with_corrs = [(gid, metrics.get('max_correlation', 0)) 
                             for gid, metrics in cell_metrics.items() 
                             if 'receptive_fields' in metrics and metrics['receptive_fields']]
        neurons_with_corrs.sort(key=lambda x: x[1], reverse=True)
        neuron_gids = [gid for gid, _ in neurons_with_corrs[:max_neurons]]
    
    if not neuron_gids:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No neurons with receptive fields found", 
               ha='center', va='center', fontsize=14)
        return fig
    
    if analysis_method == 'signal_dimensions':
        return _plot_signal_dimension_receptive_fields(processed_data, neuron_gids, figsize)
    elif analysis_method == 'binned_features':
        return _plot_binned_receptive_fields(processed_data, neuron_gids, figsize)
    else:  # feature_based
        return _plot_feature_based_receptive_fields(processed_data, neuron_gids, figsize)


def _detect_analysis_method(processed_data: Dict) -> str:
    """Detect which analysis method was used based on data structure."""
    if 'binned_activities' in processed_data:
        return 'binned_features'
    
    # Check if receptive fields have signal dimension structure
    cell_metrics = processed_data['cell_metrics']
    if cell_metrics:
        sample_neuron = next(iter(cell_metrics.values()))
        rf = sample_neuron.get('receptive_fields', {})
        if any(key in rf for key in ['raw_channels', 'frequency_bands', 'temporal_derivatives']):
            return 'signal_dimensions'
    
    return 'feature_based'


def _plot_feature_based_receptive_fields(processed_data: Dict, neuron_gids: List[int], figsize: Tuple[int, int]) -> plt.Figure:
    """Plot traditional feature-based receptive fields."""
    cell_metrics = processed_data['cell_metrics']
    dimensions_info = processed_data['input_metadata']['dimensions']
    
    n_neurons = len(neuron_gids)
    n_dims = len(dimensions_info)
    
    fig, axes = plt.subplots(n_neurons, n_dims, figsize=figsize, squeeze=False)
    
    dim_names = list(dimensions_info.keys())
    
    for i, neuron_gid in enumerate(neuron_gids):
        metrics = cell_metrics[neuron_gid]
        receptive_fields = metrics.get('receptive_fields', {})
        
        for j, dim_name in enumerate(dim_names):
            ax = axes[i, j]
            
            if dim_name in receptive_fields:
                rf = receptive_fields[dim_name]
                bin_centers = rf['bin_centers']
                responses = rf['responses']
                n_features = rf.get('n_features', np.ones_like(responses))
                
                # Plot tuning curve
                ax.plot(bin_centers, responses, 'b-', linewidth=2, marker='o', markersize=4)
                
                # Add error bars based on number of features
                yerr = responses / np.sqrt(np.maximum(n_features, 1))
                ax.fill_between(bin_centers, responses - yerr, responses + yerr, 
                               alpha=0.3, color='blue')
                
                # Highlight significant responses
                sig_mask = np.abs(responses) > np.std(responses)
                if np.any(sig_mask):
                    ax.scatter(bin_centers[sig_mask], responses[sig_mask], 
                             color='red', s=30, zorder=5)
                
                ax.set_xlabel(dim_name)
                ax.set_ylabel('Correlation')
                
                # Apply log scale if appropriate
                if dimensions_info[dim_name].get('scale') == 'log':
                    ax.set_xscale('log')
                
            else:
                ax.text(0.5, 0.5, f"No {dim_name}\nreceptive field", 
                       ha='center', va='center', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Add neuron info to leftmost plot
            if j == 0:
                firing_rate = metrics.get('firing_rate', 0)
                max_corr = metrics.get('max_correlation', 0)
                ax.set_ylabel(f'Neuron {neuron_gid}\n({firing_rate:.1f} Hz)\nCorr: {max_corr:.2f}')
            
            # Add dimension name to top row
            if i == 0:
                ax.set_title(f'{dim_name}')
    
    plt.suptitle('Feature-Based Receptive Fields', fontsize=16)
    plt.tight_layout()
    return fig


def _plot_signal_dimension_receptive_fields(processed_data: Dict, neuron_gids: List[int], figsize: Tuple[int, int]) -> plt.Figure:
    """Plot signal dimension-based receptive fields."""
    cell_metrics = processed_data['cell_metrics']
    
    n_neurons = len(neuron_gids)
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_neurons, 3, figure=fig)
    
    for i, neuron_gid in enumerate(neuron_gids):
        metrics = cell_metrics[neuron_gid]
        receptive_fields = metrics.get('receptive_fields', {})
        firing_rate = metrics.get('firing_rate', 0)
        max_corr = metrics.get('max_correlation', 0)
        
        # Raw channels
        ax1 = fig.add_subplot(gs[i, 0])
        if 'raw_channels' in receptive_fields:
            rf = receptive_fields['raw_channels']
            signal_names = rf['signal_names']
            responses = rf['responses']
            
            x_pos = np.arange(len(signal_names))
            bars = ax1.bar(x_pos, responses, alpha=0.7)
            
            # Color bars by response strength
            colors = plt.cm.RdBu_r((np.array(responses) + 1) / 2)  # Normalize to [0,1]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([name.replace('channel_', 'Ch') for name in signal_names], rotation=45)
            ax1.set_ylabel('Correlation')
            ax1.set_title('Raw Channels')
        else:
            ax1.text(0.5, 0.5, 'No raw channel data', ha='center', va='center')
        
        # Frequency bands
        ax2 = fig.add_subplot(gs[i, 1])
        if 'frequency_bands' in receptive_fields:
            rf = receptive_fields['frequency_bands']
            signal_names = rf['signal_names']
            responses = rf['responses']
            
            # Extract frequency information from signal names
            freq_info = []
            for name in signal_names:
                if '_band_' in name:
                    freq_part = name.split('_band_')[1]
                    freq_info.append(freq_part.replace('Hz', '').replace('_', '-'))
                else:
                    freq_info.append(name)
            
            x_pos = np.arange(len(freq_info))
            bars = ax2.bar(x_pos, responses, alpha=0.7)
            
            colors = plt.cm.RdBu_r((np.array(responses) + 1) / 2)
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(freq_info, rotation=45)
            ax2.set_ylabel('Correlation')
            ax2.set_title('Frequency Bands')
        else:
            ax2.text(0.5, 0.5, 'No frequency band data', ha='center', va='center')
        
        # Temporal derivatives
        ax3 = fig.add_subplot(gs[i, 2])
        if 'temporal_derivatives' in receptive_fields:
            rf = receptive_fields['temporal_derivatives']
            signal_names = rf['signal_names']
            responses = rf['responses']
            
            # Simplify derivative names
            deriv_names = []
            for name in signal_names:
                if 'second_derivative' in name:
                    deriv_names.append('2nd Deriv')
                elif 'derivative' in name:
                    deriv_names.append('1st Deriv')
                else:
                    deriv_names.append(name)
            
            x_pos = np.arange(len(deriv_names))
            bars = ax3.bar(x_pos, responses, alpha=0.7)
            
            colors = plt.cm.RdBu_r((np.array(responses) + 1) / 2)
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(deriv_names, rotation=45)
            ax3.set_ylabel('Correlation')
            ax3.set_title('Temporal Derivatives')
        else:
            ax3.text(0.5, 0.5, 'No derivative data', ha='center', va='center')
        
        # Add neuron info to leftmost plot
        ax1.text(-0.3, 0.5, f'Neuron {neuron_gid}\n({firing_rate:.1f} Hz)\nCorr: {max_corr:.2f}', 
                transform=ax1.transAxes, rotation=90, va='center', ha='center')
    
    plt.suptitle('Signal Dimension Receptive Fields', fontsize=16)
    plt.tight_layout()
    return fig


def _plot_binned_receptive_fields(processed_data: Dict, neuron_gids: List[int], figsize: Tuple[int, int]) -> plt.Figure:
    """Plot binned feature receptive fields."""
    cell_metrics = processed_data['cell_metrics']
    bin_info = processed_data.get('bin_info', {})
    
    if not bin_info:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No bin information found", ha='center', va='center', fontsize=14)
        return fig
    
    n_neurons = len(neuron_gids)
    n_dims = len(bin_info)
    
    fig, axes = plt.subplots(n_neurons, n_dims, figsize=figsize, squeeze=False)
    
    dim_names = list(bin_info.keys())
    
    for i, neuron_gid in enumerate(neuron_gids):
        metrics = cell_metrics[neuron_gid]
        receptive_fields = metrics.get('receptive_fields', {})
        
        for j, dim_name in enumerate(dim_names):
            ax = axes[i, j]
            
            if dim_name in receptive_fields:
                rf = receptive_fields[dim_name]
                bin_centers = rf['bin_centers']
                responses = rf['responses']
                
                # Plot tuning curve
                ax.plot(bin_centers, responses, 'g-', linewidth=2, marker='s', markersize=6)
                
                # Fill area under curve
                ax.fill_between(bin_centers, responses, alpha=0.3, color='green')
                
                # Highlight significant responses
                sig_mask = np.abs(responses) > np.std(responses)
                if np.any(sig_mask):
                    ax.scatter(bin_centers[sig_mask], responses[sig_mask], 
                             color='red', s=40, zorder=5, marker='*')
                
                ax.set_xlabel(dim_name)
                ax.set_ylabel('Correlation')
                
                # Apply log scale if appropriate
                scale = bin_info[dim_name].get('scale', 'linear')
                if scale == 'log':
                    ax.set_xscale('log')
                
                # Add bin indicators
                bin_edges = rf.get('bin_edges', [])
                if len(bin_edges) > 0:
                    for edge in bin_edges:
                        ax.axvline(x=edge, color='gray', linestyle=':', alpha=0.5)
                
            else:
                ax.text(0.5, 0.5, f"No {dim_name}\nbinned RF", 
                       ha='center', va='center', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Add neuron info to leftmost plot
            if j == 0:
                firing_rate = metrics.get('firing_rate', 0)
                max_corr = metrics.get('max_correlation', 0)
                ax.set_ylabel(f'Neuron {neuron_gid}\n({firing_rate:.1f} Hz)\nCorr: {max_corr:.2f}')
            
            # Add dimension name to top row
            if i == 0:
                n_bins = bin_info[dim_name].get('n_bins', 0)
                ax.set_title(f'{dim_name}\n({n_bins} bins)')
    
    plt.suptitle('Binned Feature Receptive Fields', fontsize=16)
    plt.tight_layout()
    return fig


def plot_receptive_field_heatmaps(
    processed_data: Dict,
    dim_x: str = None,
    dim_y: str = None,
    max_neurons: int = 12,
    figsize: Tuple[int, int] = (16, 12),
    analysis_method: str = 'auto'
) -> plt.Figure:
    """
    Plot 2D receptive field heatmaps for pairs of dimensions.
    Automatically handles different analysis methods.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed data from process_model_spatiotemporal_responses_improved
    dim_x : str
        First dimension name (auto-selected if None)
    dim_y : str
        Second dimension name (auto-selected if None)
    max_neurons : int
        Maximum number of neurons to plot
    figsize : Tuple[int, int]
        Figure size
    analysis_method : str
        'auto', 'feature_based', 'signal_dimensions', or 'binned_features'
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    # Auto-detect analysis method
    if analysis_method == 'auto':
        analysis_method = _detect_analysis_method(processed_data)
    
    if analysis_method == 'signal_dimensions':
        return _plot_signal_dimension_heatmaps(processed_data, max_neurons, figsize)
    elif analysis_method == 'binned_features':
        return _plot_binned_feature_heatmaps(processed_data, dim_x, dim_y, max_neurons, figsize)
    else:  # feature_based
        return _plot_feature_based_heatmaps(processed_data, dim_x, dim_y, max_neurons, figsize)


def _plot_feature_based_heatmaps(processed_data: Dict, dim_x: str, dim_y: str, max_neurons: int, figsize: Tuple[int, int]) -> plt.Figure:
    """Plot 2D heatmaps for feature-based analysis."""
    cell_metrics = processed_data['cell_metrics']
    dimensions_info = processed_data['input_metadata']['dimensions']
    feature_data = processed_data['input_metadata']['feature_data']
    
    # Auto-select dimensions if not provided
    if dim_x is None or dim_y is None:
        dim_names = list(dimensions_info.keys())
        if len(dim_names) >= 2:
            dim_x = dim_x or dim_names[1]  # temporal_frequency
            dim_y = dim_y or dim_names[2]  # spatial_position
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Need at least 2 dimensions for heatmaps", 
                   ha='center', va='center', fontsize=14)
            return fig
    
    # Check dimensions exist
    if dim_x not in dimensions_info or dim_y not in dimensions_info:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Dimensions {dim_x} or {dim_y} not found", 
               ha='center', va='center', fontsize=14)
        return fig
    
    # Get dimension indices
    dim_names = list(dimensions_info.keys())
    x_idx = dim_names.index(dim_x)
    y_idx = dim_names.index(dim_y)
    
    # Select neurons with good receptive fields
    neurons_with_corrs = [(gid, metrics.get('max_correlation', 0)) 
                         for gid, metrics in cell_metrics.items() 
                         if 'feature_correlations' in metrics and metrics['feature_correlations']]
    neurons_with_corrs.sort(key=lambda x: x[1], reverse=True)
    neuron_gids = [gid for gid, _ in neurons_with_corrs[:max_neurons]]
    
    if not neuron_gids:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No neurons with correlations found", 
               ha='center', va='center', fontsize=14)
        return fig
    
    # Prepare feature position data
    gids = feature_data.get('gids', [])
    positions = feature_data.get('positions', [])
    
    feature_positions = {}
    for i, gid in enumerate(gids):
        if i < len(positions) and len(positions[i]) > max(x_idx, y_idx):
            feature_positions[int(gid)] = positions[i]
    
    # Create subplot grid
    n_cols = 4
    n_rows = int(np.ceil(len(neuron_gids) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, neuron_gid in enumerate(neuron_gids):
        ax = axes[i]
        
        correlations = cell_metrics[neuron_gid].get('feature_correlations', {})
        
        if not correlations:
            ax.text(0.5, 0.5, f"No correlations\nfor neuron {neuron_gid}", 
                   ha='center', va='center', fontsize=10)
            continue
        
        # Extract 2D data
        x_vals = []
        y_vals = []
        corr_vals = []
        
        for feature_gid, correlation in correlations.items():
            if feature_gid in feature_positions:
                pos = feature_positions[feature_gid]
                x_vals.append(pos[x_idx])
                y_vals.append(pos[y_idx])
                corr_vals.append(correlation)
        
        if len(x_vals) < 4:  # Need minimum points for interpolation
            ax.text(0.5, 0.5, f"Insufficient data\nfor neuron {neuron_gid}", 
                   ha='center', va='center', fontsize=10)
            continue
        
        # Create interpolated heatmap
        x_range = dimensions_info[dim_x]['range']
        y_range = dimensions_info[dim_y]['range']
        
        # Handle log scaling
        if dimensions_info[dim_x].get('scale') == 'log':
            x_grid = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), 50)
        else:
            x_grid = np.linspace(x_range[0], x_range[1], 50)
            
        if dimensions_info[dim_y].get('scale') == 'log':
            y_grid = np.logspace(np.log10(y_range[0]), np.log10(y_range[1]), 50)
        else:
            y_grid = np.linspace(y_range[0], y_range[1], 50)
        
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate correlation values
        try:
            Z = griddata((x_vals, y_vals), corr_vals, (X, Y), method='cubic', fill_value=0)
        except:
            Z = griddata((x_vals, y_vals), corr_vals, (X, Y), method='linear', fill_value=0)
        
        # Plot heatmap
        vmax = max(np.abs(np.nanmin(Z)), np.abs(np.nanmax(Z)))
        im = ax.pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        
        # Overlay data points
        scatter = ax.scatter(x_vals, y_vals, c=corr_vals, s=20, 
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax, 
                           edgecolors='black', linewidths=0.5)
        
        # Set scales
        if dimensions_info[dim_x].get('scale') == 'log':
            ax.set_xscale('log')
        if dimensions_info[dim_y].get('scale') == 'log':
            ax.set_yscale('log')
        
        ax.set_xlabel(dim_x)
        ax.set_ylabel(dim_y)
        
        firing_rate = cell_metrics[neuron_gid].get('firing_rate', 0)
        max_corr = cell_metrics[neuron_gid].get('max_correlation', 0)
        ax.set_title(f'Neuron {neuron_gid}\n({firing_rate:.1f} Hz, r={max_corr:.2f})', 
                    fontsize=10)
    
    # Hide unused subplots
    for i in range(len(neuron_gids), len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbar
    sm = ScalarMappable(cmap='RdBu_r')
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:len(neuron_gids)], label='Correlation', shrink=0.8)
    
    plt.suptitle(f'Feature-Based 2D Receptive Fields: {dim_x} vs {dim_y}', fontsize=16)
    plt.tight_layout()
    return fig


def _plot_signal_dimension_heatmaps(processed_data: Dict, max_neurons: int, figsize: Tuple[int, int]) -> plt.Figure:
    """Plot signal dimension correlation patterns."""
    cell_metrics = processed_data['cell_metrics']
    
    # Select neurons with good receptive fields
    neurons_with_corrs = [(gid, metrics.get('max_correlation', 0)) 
                         for gid, metrics in cell_metrics.items() 
                         if 'feature_correlations' in metrics and metrics['feature_correlations']]
    neurons_with_corrs.sort(key=lambda x: x[1], reverse=True)
    neuron_gids = [gid for gid, _ in neurons_with_corrs[:max_neurons]]
    
    if not neuron_gids:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No neurons with signal correlations found", 
               ha='center', va='center', fontsize=14)
        return fig
    
    # Collect all signal dimensions
    all_signals = set()
    for gid in neuron_gids:
        correlations = cell_metrics[gid].get('feature_correlations', {})
        all_signals.update(correlations.keys())
    
    signal_list = sorted(list(all_signals))
    
    # Build correlation matrix
    corr_matrix = np.zeros((len(neuron_gids), len(signal_list)))
    
    for i, neuron_gid in enumerate(neuron_gids):
        correlations = cell_metrics[neuron_gid].get('feature_correlations', {})
        for j, signal_name in enumerate(signal_list):
            corr_matrix[i, j] = correlations.get(signal_name, 0.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 1]})
    
    # Main heatmap
    im = ax1.imshow(corr_matrix, aspect='auto', cmap='RdBu_r', 
                   vmin=-1, vmax=1, interpolation='nearest')
    
    ax1.set_xlabel('Signal Dimension')
    ax1.set_ylabel('Neuron')
    ax1.set_title('Signal Dimension Correlations')
    
    # Set ticks
    ax1.set_xticks(range(len(signal_list)))
    ax1.set_xticklabels([s.replace('channel_', 'Ch').replace('_band_', '\n') for s in signal_list], 
                       rotation=45, ha='right')
    ax1.set_yticks(range(len(neuron_gids)))
    ax1.set_yticklabels([f'{gid}' for gid in neuron_gids])
    
    # Colorbar
    plt.colorbar(im, ax=ax1, label='Correlation')
    
    # Summary statistics
    mean_corrs = np.mean(np.abs(corr_matrix), axis=0)
    ax2.barh(range(len(signal_list)), mean_corrs, alpha=0.7)
    ax2.set_ylim(-0.5, len(signal_list) - 0.5)
    ax2.set_xlabel('Mean |Correlation|')
    ax2.set_title('Signal Importance')
    ax2.set_yticks(range(len(signal_list)))
    ax2.set_yticklabels([s.replace('channel_', 'Ch').replace('_', '\n') for s in signal_list])
    
    plt.suptitle('Signal Dimension Analysis', fontsize=16)
    plt.tight_layout()
    return fig


def _plot_binned_feature_heatmaps(processed_data: Dict, dim_x: str, dim_y: str, max_neurons: int, figsize: Tuple[int, int]) -> plt.Figure:
    """Plot 2D heatmaps for binned feature analysis."""
    cell_metrics = processed_data['cell_metrics']
    bin_info = processed_data.get('bin_info', {})
    
    if not bin_info:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No bin information found", ha='center', va='center', fontsize=14)
        return fig
    
    # Auto-select dimensions if not provided
    if dim_x is None or dim_y is None:
        dim_names = list(bin_info.keys())
        if len(dim_names) >= 2:
            dim_x = dim_x or dim_names[1]  # temporal_frequency
            dim_y = dim_y or dim_names[2]  # spatial_position
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Need at least 2 dimensions for heatmaps", 
                   ha='center', va='center', fontsize=14)
            return fig
    
    # Check dimensions exist
    if dim_x not in bin_info or dim_y not in bin_info:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Dimensions {dim_x} or {dim_y} not found in bin info", 
               ha='center', va='center', fontsize=14)
        return fig
    
    # Select neurons with good receptive fields
    neurons_with_corrs = [(gid, metrics.get('max_correlation', 0)) 
                         for gid, metrics in cell_metrics.items() 
                         if 'receptive_fields' in metrics and metrics['receptive_fields']]
    neurons_with_corrs.sort(key=lambda x: x[1], reverse=True)
    neuron_gids = [gid for gid, _ in neurons_with_corrs[:max_neurons]]
    
    if not neuron_gids:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No neurons with receptive fields found", 
               ha='center', va='center', fontsize=14)
        return fig
    
    # Get bin information
    x_bin_centers = bin_info[dim_x]['bin_centers']
    y_bin_centers = bin_info[dim_y]['bin_centers']
    n_x_bins = len(x_bin_centers)
    n_y_bins = len(y_bin_centers)
    
    # Create subplot grid
    n_cols = 4
    n_rows = int(np.ceil(len(neuron_gids) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, neuron_gid in enumerate(neuron_gids):
        ax = axes[i]
        
        receptive_fields = cell_metrics[neuron_gid].get('receptive_fields', {})
        
        if dim_x not in receptive_fields or dim_y not in receptive_fields:
            ax.text(0.5, 0.5, f"Missing RF data\nfor neuron {neuron_gid}", 
                   ha='center', va='center', fontsize=10)
            continue
        
        # Get responses for both dimensions
        x_responses = receptive_fields[dim_x]['responses']
        y_responses = receptive_fields[dim_y]['responses']
        
        # Create 2D response matrix as outer product
        Z = np.outer(y_responses, x_responses)
        
        # Plot heatmap
        vmax = np.max(np.abs(Z))
        im = ax.imshow(Z, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                      extent=[x_bin_centers[0], x_bin_centers[-1], 
                             y_bin_centers[0], y_bin_centers[-1]],
                      origin='lower', interpolation='bilinear')
        
        ax.set_xlabel(dim_x)
        ax.set_ylabel(dim_y)
        
        firing_rate = cell_metrics[neuron_gid].get('firing_rate', 0)
        max_corr = cell_metrics[neuron_gid].get('max_correlation', 0)
        ax.set_title(f'Neuron {neuron_gid}\n({firing_rate:.1f} Hz, r={max_corr:.2f})', 
                    fontsize=10)
        
        # Add bin grid lines
        for x_center in x_bin_centers:
            ax.axvline(x=x_center, color='white', linewidth=0.5, alpha=0.3)
        for y_center in y_bin_centers:
            ax.axhline(y=y_center, color='white', linewidth=0.5, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(neuron_gids), len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbar
    sm = ScalarMappable(cmap='RdBu_r')
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:len(neuron_gids)], label='Correlation', shrink=0.8)
    
    plt.suptitle(f'Binned Feature 2D Receptive Fields: {dim_x} vs {dim_y}', fontsize=16)
    plt.tight_layout()
    return fig


def plot_population_correlation_summary(
    processed_data: Dict,
    figsize: Tuple[int, int] = (15, 10),
    analysis_method = 'auto'
) -> plt.Figure:
    """
    Plot population-level correlation statistics and summaries.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed data from process_model_spatiotemporal_responses_improved
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    cell_metrics = processed_data['cell_metrics']
    dimensions_info = processed_data['input_metadata']['dimensions']
    pop_metrics = processed_data['population_metrics']
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # Extract data
    firing_rates = []
    max_correlations = []
    mean_correlations = []
    n_spikes_list = []
    
    for gid, metrics in cell_metrics.items():
        firing_rates.append(metrics.get('firing_rate', 0))
        max_correlations.append(metrics.get('max_correlation', 0))
        mean_correlations.append(metrics.get('mean_correlation', 0))
        n_spikes_list.append(metrics.get('n_spikes', 0))
    
    # Firing rate vs correlation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(firing_rates, max_correlations, alpha=0.6, s=20)
    
    # Add correlation line
    if len(firing_rates) > 2:
        try:
            r, p = pearsonr(firing_rates, max_correlations)
            ax1.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
    
    ax1.set_xlabel('Firing Rate (Hz)')
    ax1.set_ylabel('Max Correlation')
    ax1.set_title('Firing Rate vs Max Correlation')
    
    # Correlation distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(max_correlations, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(np.mean(max_correlations), color='red', linestyle='--', 
               label=f'Mean: {np.mean(max_correlations):.3f}')
    ax2.set_xlabel('Max Correlation')
    ax2.set_ylabel('Count')
    ax2.set_title('Max Correlation Distribution')
    ax2.legend()
    
    # Active vs silent neurons
    ax3 = fig.add_subplot(gs[0, 2])
    active_neurons = sum(1 for n in n_spikes_list if n > 0)
    silent_neurons = len(n_spikes_list) - active_neurons
    
    ax3.pie([active_neurons, silent_neurons], 
           labels=['Active', 'Silent'], 
           autopct='%1.1f%%',
           colors=['lightgreen', 'lightcoral'])
    ax3.set_title(f'Active vs Silent Neurons\n(Total: {len(n_spikes_list)})')
    
    # Correlation by dimension
    ax4 = fig.add_subplot(gs[1, :])
    
    dim_correlations = {dim: [] for dim in dimensions_info.keys()}
    
    for gid, metrics in cell_metrics.items():
        receptive_fields = metrics.get('receptive_fields', {})
        for dim_name, rf_data in receptive_fields.items():
            if 'responses' in rf_data:
                max_response = np.max(np.abs(rf_data['responses']))
                dim_correlations[dim_name].append(max_response)
    
    # Box plot of correlations by dimension
    dim_names = list(dim_correlations.keys())
    dim_data = [dim_correlations[dim] for dim in dim_names if dim_correlations[dim]]
    
    if dim_data:
        bp = ax4.boxplot(dim_data, labels=dim_names, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    
    ax4.set_ylabel('Max Correlation in Dimension')
    ax4.set_title('Correlation Strength by Dimension')
    ax4.tick_params(axis='x', rotation=45)
    
    # Population response over time
    ax5 = fig.add_subplot(gs[2, :])
    
    pop_rate = pop_metrics['population_rate']
    bin_edges = pop_metrics['bin_edges']
    
    if len(bin_edges) == len(pop_rate) + 1:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    else:
        bin_centers = np.arange(len(pop_rate)) * 10  # 10ms bins
    
    ax5.plot(bin_centers, pop_rate, color='darkblue', linewidth=1.5)
    ax5.fill_between(bin_centers, pop_rate, alpha=0.3, color='lightblue')
    
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Population Rate (Hz/cell)')
    ax5.set_title('Population Response Over Time')
    ax5.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Population Statistics:
Total neurons: {len(cell_metrics)}
Active neurons: {pop_metrics['n_active_cells']}
Mean firing rate: {np.mean(firing_rates):.2f} Hz
Mean max correlation: {np.mean(max_correlations):.3f}
Correlation method: {pop_metrics.get('correlation_method', 'N/A')}
Time bin: {pop_metrics.get('time_bin_ms', 'N/A')} ms"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_temporal_correlation_dynamics(
    processed_data: Dict,
    neuron_gids: Optional[List[int]] = None,
    window_size_ms: float = 100.0,
    step_size_ms: float = 10.0,
    max_neurons: int = 6,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot how correlations change over time using sliding windows.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed data from process_model_spatiotemporal_responses_improved
    neuron_gids : Optional[List[int]]
        Specific neurons to analyze
    window_size_ms : float
        Size of sliding window in milliseconds
    step_size_ms : float
        Step size for sliding window in milliseconds
    max_neurons : int
        Maximum number of neurons to plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    cell_metrics = processed_data['cell_metrics']
    feature_activities = processed_data.get('feature_activities', {})
    duration = processed_data['input_metadata']['duration'] * 1000  # Convert to ms
    time_bin_ms = processed_data['population_metrics'].get('time_bin_ms', 1.0)
    
    # Select neurons
    if neuron_gids is None:
        neurons_with_corrs = [(gid, metrics.get('max_correlation', 0)) 
                             for gid, metrics in cell_metrics.items() 
                             if 'spike_times' in metrics and len(metrics['spike_times']) > 0]
        neurons_with_corrs.sort(key=lambda x: x[1], reverse=True)
        neuron_gids = [gid for gid, _ in neurons_with_corrs[:max_neurons]]
    
    if not neuron_gids or not feature_activities:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Insufficient data for temporal analysis", 
               ha='center', va='center', fontsize=14)
        return fig
    
    # Choose a representative feature for temporal analysis
    feature_gid = list(feature_activities.keys())[0]
    feature_activity = feature_activities[feature_gid]
    
    # Create sliding windows
    window_bins = int(window_size_ms / time_bin_ms)
    step_bins = int(step_size_ms / time_bin_ms)
    
    n_windows = int((len(feature_activity) - window_bins) / step_bins) + 1
    window_times = np.arange(n_windows) * step_size_ms + window_size_ms / 2
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i, neuron_gid in enumerate(neuron_gids):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get neuron spike times and convert to rate
        spike_times = cell_metrics[neuron_gid]['spike_times']
        
        # Convert to binned rate
        from analyze_spatiotemporal_responses import compute_spike_rate_timeseries
        neuron_rate = compute_spike_rate_timeseries(spike_times, duration, time_bin_ms)
        
        # Ensure same length
        min_length = min(len(neuron_rate), len(feature_activity))
        neuron_rate = neuron_rate[:min_length]
        feature_act = feature_activity[:min_length]
        
        # Compute sliding window correlations
        correlations_over_time = []
        
        for w in range(n_windows):
            start_idx = w * step_bins
            end_idx = start_idx + window_bins
            
            if end_idx <= len(neuron_rate):
                neuron_window = neuron_rate[start_idx:end_idx]
                feature_window = feature_act[start_idx:end_idx]
                
                if np.sum(neuron_window) > 0 and np.sum(feature_window) > 0:
                    try:
                        corr, _ = pearsonr(neuron_window, feature_window)
                        correlations_over_time.append(corr if not np.isnan(corr) else 0.0)
                    except:
                        correlations_over_time.append(0.0)
                else:
                    correlations_over_time.append(0.0)
            else:
                correlations_over_time.append(0.0)
        
        # Plot temporal correlation
        ax.plot(window_times[:len(correlations_over_time)], correlations_over_time, 
               'b-', linewidth=2, alpha=0.8)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add smoothed trend
        if len(correlations_over_time) > 5:
            from scipy.signal import savgol_filter
            smoothed = savgol_filter(correlations_over_time, 
                                   min(len(correlations_over_time)//3*2-1, 11), 1)
            ax.plot(window_times[:len(smoothed)], smoothed, 
                   'r-', linewidth=1.5, alpha=0.7, label='Smoothed')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Neuron {neuron_gid}\n({cell_metrics[neuron_gid]["firing_rate"]:.1f} Hz)')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    # Hide unused subplots
    for i in range(len(neuron_gids), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Temporal Correlation Dynamics\n(Window: {window_size_ms}ms, Step: {step_size_ms}ms)', 
                fontsize=16)
    plt.tight_layout()
    return fig


def save_all_plots(
    processed_data: Dict,
    output_dir: str = "./figures",
    file_format: str = "png",
    dpi: int = 300,
    analysis_method = 'auto'
) -> Dict[str, str]:
    """
    Generate and save all visualization plots.
    Automatically detects analysis method and generates appropriate plots.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed data from process_model_spatiotemporal_responses_improved
    output_dir : str
        Directory to save plots
    file_format : str
        File format for saved plots
    dpi : int
        DPI for saved plots
    analysis_method : str
        'auto', 'feature_based', 'signal_dimensions', or 'binned_features'
        
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping plot names to file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect analysis method
    if analysis_method == 'auto':
        analysis_method = _detect_analysis_method(processed_data)
    
    saved_files = {}
    
    # Feature activities plot (only for feature-based methods)
    if analysis_method in ['feature_based', 'binned_features'] and 'feature_activities' in processed_data:
        try:
            fig = plot_feature_activities(
                processed_data['feature_activities'],
                processed_data['input_metadata']['feature_data'],
                processed_data['input_metadata']['dimensions']
            )
            filename = os.path.join(output_dir, f"feature_activities_{analysis_method}.{file_format}")
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            saved_files['feature_activities'] = filename
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Failed to generate feature activities plot: {e}")
    
    # Correlation matrix plot
    try:
        correlations = {gid: metrics.get('feature_correlations', {}) 
                       for gid, metrics in processed_data['cell_metrics'].items()
                       if 'feature_correlations' in metrics}
        
        fig = plot_correlation_matrix(correlations)
        filename = os.path.join(output_dir, f"correlation_matrix_{analysis_method}.{file_format}")
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        saved_files['correlation_matrix'] = filename
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Failed to generate correlation matrix plot: {e}")
    
    # Dimensional receptive fields plot
    try:
        fig = plot_dimensional_receptive_fields(processed_data, analysis_method=analysis_method)
        filename = os.path.join(output_dir, f"dimensional_receptive_fields_{analysis_method}.{file_format}")
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        saved_files['dimensional_receptive_fields'] = filename
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Failed to generate dimensional receptive fields plot: {e}")
    
    # Population summary plot
    try:
        fig = plot_population_correlation_summary(processed_data, analysis_method=analysis_method)
        filename = os.path.join(output_dir, f"population_summary_{analysis_method}.{file_format}")
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        saved_files['population_summary'] = filename
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Failed to generate population summary plot: {e}")
    
    # 2D receptive field heatmaps
    try:
        fig = plot_receptive_field_heatmaps(processed_data, analysis_method=analysis_method)
        filename = os.path.join(output_dir, f"rf_heatmaps_{analysis_method}.{file_format}")
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        saved_files['rf_heatmaps'] = filename
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Failed to generate receptive field heatmaps: {e}")
    
    # Method-specific additional plots
    if analysis_method == 'binned_features' and 'binned_activities' in processed_data:
        try:
            fig = plot_binned_activities_overview(processed_data)
            filename = os.path.join(output_dir, f"binned_activities_overview.{file_format}")
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            saved_files['binned_activities'] = filename
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Failed to generate binned activities plot: {e}")
    
    return saved_files


def plot_binned_activities_overview(processed_data: Dict, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot overview of binned feature activities for binned feature analysis.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed data from binned feature analysis
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Generated figure
    """
    binned_activities = processed_data.get('binned_activities', {})
    bin_info = processed_data.get('bin_info', {})
    
    if not binned_activities or not bin_info:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No binned activities data found", 
               ha='center', va='center', fontsize=14)
        return fig
    
    n_dims = len(binned_activities)
    fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, (dim_name, dim_activities) in enumerate(binned_activities.items()):
        if i >= 4:  # Only plot first 4 dimensions
            break
            
        ax = axes[i]
        
        n_bins = len(dim_activities)
        bin_centers = bin_info[dim_name]['bin_centers']
        
        # Plot average activity for each bin
        avg_activities = []
        for bin_idx in range(n_bins):
            activity = dim_activities[bin_idx]
            avg_activities.append(np.mean(activity))
        
        bars = ax.bar(range(n_bins), avg_activities, alpha=0.7)
        
        # Color bars by activity level
        colors = plt.cm.viridis(np.array(avg_activities) / max(avg_activities))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Set bin center labels
        ax.set_xticks(range(n_bins))
        bin_labels = [f'{center:.2f}' for center in bin_centers]
        ax.set_xticklabels(bin_labels, rotation=45)
        
        ax.set_xlabel(f'{dim_name}')
        ax.set_ylabel('Mean Activity')
        ax.set_title(f'Binned Activities: {dim_name}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(binned_activities), 4):
        axes[i].set_visible(False)
    
    plt.suptitle('Binned Feature Activities Overview', fontsize=16)
    plt.tight_layout()
    return fig

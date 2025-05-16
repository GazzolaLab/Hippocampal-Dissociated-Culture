import os
import sys
from typing import List, Optional, Tuple
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

def process_model_temporal_responses(
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

    Parameters:
    -----------
    simulation_duration : float
        Duration of simulation in ms
    populations : List, optional
        List of populations to analyze (default: all)
        
    Returns:
    --------
    Dict
        Processed responses with metrics
    """

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
        
        input_signal = signal_group['data'][:]
        time = signal_group['time'][:]
        frequencies = signal_group['frequencies'][:]
        segment_starts = signal_group['segment_starts'][:]
        segment_ends = signal_group['segment_ends'][:]
        
        duration = signal_group.attrs.get('duration', 10.0)  # seconds
        sample_rate = signal_group.attrs.get('sample_rate', 1000)  # Hz
        sample_dt_ms = signal_group.attrs.get('sample_dt_ms', 1.0)  # ms
    
    # Create segment info
    segments = []
    for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        segments.append({
            'start_idx': start,
            'end_idx': end,
            'frequency': frequencies[i],
            'start_time': time[start],
            'end_time': time[end-1] if end < len(time) else time[-1],
            'duration': (time[end-1] if end < len(time) else time[-1]) - time[start]
        })
            
    
    time_range = [tmin, tmax]
    
    processed_responses = {}
    
    # Process each population
    for pop_name in include:
        if pop_name not in pop_spk_dict:
            continue
        
        pop_spkinds, pop_spkts = pop_spk_dict[pop_name]
        gid_spike_dict = spikedata.make_spike_dict(pop_spkinds, pop_spkts)
        
        all_spikes = pop_spkts
        cell_metrics = {}

        for gid, spike_times in gid_spike_dict.items():
            if len(spike_times) == 0:
                cell_metrics[gid] = {
                    'firing_rate': 0,
                    'n_spikes': 0,
                    'spike_times': [],
                    'isi': [],
                    'cv_isi': 0,
                    'burst_index': 0,
                    'segment_responses': []
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

            # Compute response to each frequency segment
            segment_responses = []
            for segment in segments:
                segment_start_ms = segment['start_time'] * 1000  # ms
                segment_end_ms = segment['end_time'] * 1000

                # count spikes in this segment
                segment_spikes = spike_times[(spike_times >= segment_start_ms) & 
                                            (spike_times <= segment_end_ms)]

                segment_duration = (segment_end_ms - segment_start_ms) / 1000  # seconds
                segment_rate = len(segment_spikes) / segment_duration if segment_duration > 0 else 0

                segment_responses.append({
                    'frequency': segment['frequency'],
                    'n_spikes': len(segment_spikes),
                    'firing_rate': segment_rate,
                    'spikes': segment_spikes
                })

            cell_metrics[gid] = {
                'firing_rate': firing_rate,
                'n_spikes': n_spikes,
                'spike_times': spike_times,
                'isi': isi,
                'cv_isi': cv_isi,
                'burst_index': burst_index,
                'segment_responses': segment_responses
            }

        all_spikes = np.array(sorted(all_spikes))

        # population spike rate histogram (10ms bins)
        bin_size = 10  # ms
        n_bins = int(simulation_duration / bin_size)
        pop_rate, bin_edges = np.histogram(
            all_spikes, bins=n_bins, range=(0, simulation_duration)
        )

        # Normalize to spikes/s per cell
        n_cells = len(gid_spike_dict)
        if n_cells > 0:
            pop_rate = pop_rate / (bin_size/1000) / n_cells  # Hz per cell

        # Compute population response to each frequency segment
        segment_population_responses = []
        for segment in segments:
            segment_start_ms = segment['start_time'] * 1000  # convert to ms
            segment_end_ms = segment['end_time'] * 1000

            segment_bin_start = int(segment_start_ms / bin_size)
            segment_bin_end = int(segment_end_ms / bin_size)
            segment_pop_rate = pop_rate[segment_bin_start:segment_bin_end]

            # spectral analysis of population rate during this segment
            if len(segment_pop_rate) > 50:  # Ensure enough data for PSD
                fs = 1000 / bin_size  # Hz
                freqs, psd = signal.welch(segment_pop_rate, fs=fs, nperseg=min(256, len(segment_pop_rate)))

                # power in physiological frequency bands
                freq_bands = {
                    'delta': (1, 4),
                    'theta': (4, 12),
                    'beta': (12, 30),
                    'gamma': (30, 100)
                }

                band_powers = {}
                for band, (low, high) in freq_bands.items():
                    mask = (freqs >= low) & (freqs <= high)
                    if np.any(mask):
                        band_powers[f'{band}_power'] = np.mean(psd[mask])
                    else:
                        band_powers[f'{band}_power'] = 0
            else:
                freqs, psd = np.array([]), np.array([])
                band_powers = {f'{band}_power': 0 for band in ['delta', 'theta', 'beta', 'gamma']}

            segment_duration = (segment_end_ms - segment_start_ms) / 1000  # seconds
            segment_total_spikes = sum(len(cell_metrics[gid]['segment_responses'][len(segment_population_responses)]['spikes']) 
                                      for gid in cell_metrics)
            segment_mean_rate = segment_total_spikes / (segment_duration * n_cells) if segment_duration > 0 and n_cells > 0 else 0

            segment_population_responses.append({
                'frequency': segment['frequency'],
                'start_time': segment_start_ms,
                'end_time': segment_end_ms,
                'mean_rate': segment_mean_rate,
                'population_rate': segment_pop_rate,
                'psd_freqs': freqs,
                'psd': psd,
                'band_powers': band_powers
            })

        processed_data = {
            'input_metadata': {
                'signal': input_signal,
                'time': time,
                'segments': segments,
                'duration': duration,
                'sample_rate': sample_rate,
                'sample_dt_ms': sample_dt_ms
            },
            'population_metrics': {
                'all_spikes': all_spikes,
                'population_rate': pop_rate,
                'bin_edges': bin_edges,
                'mean_rate': np.mean(pop_rate) if len(pop_rate) > 0 else 0,
                'n_active_cells': sum(1 for gid, metrics in cell_metrics.items() if metrics['n_spikes'] > 0),
                'total_cells': n_cells,
                'segment_responses': segment_population_responses
            },
            'cell_metrics': cell_metrics
        }

        return processed_data

def plot_input_output_feature_transformation(processed_data):
    """
    Plot of the input transformation performed by the model network.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed responses from process_model_temporal_responses
    """
    input_metadata = processed_data['input_metadata']
    population_metrics = processed_data['population_metrics']
    
    input_signal = input_metadata['signal']
    time = input_metadata['time']
    segments = input_metadata['segments']
    
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[3, 1])
    
    # 1. Plot input signal
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.plot(time, input_signal)
    ax_input.set_title('Input Signal with Changing Frequencies')
    ax_input.set_xlabel('Time (s)')
    ax_input.set_ylabel('Amplitude')
    
    # Add segment markers
    for segment in segments:
        freq = segment['frequency']
        start_time = segment['start_time']
        end_time = segment['end_time']
        ax_input.axvspan(start_time, end_time, alpha=0.2, color='r')
        ax_input.text((start_time + end_time)/2, 0.8, f"{freq} Hz", 
                     ha='center', va='center', fontweight='bold')
    
    # 2. Plot population spike rate
    bin_edges = population_metrics['bin_edges']
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 / 1000  # convert to seconds
    pop_rate = population_metrics['population_rate']
    
    ax_rate = fig.add_subplot(gs[1, 0])
    ax_rate.plot(bin_centers, pop_rate)
    ax_rate.set_title('Population Firing Rate')
    ax_rate.set_xlabel('Time (s)')
    ax_rate.set_ylabel('Firing Rate (Hz/cell)')
    
    # Add segment markers
    for segment in segments:
        freq = segment['frequency']
        start_time = segment['start_time'] / 1000  # convert to seconds
        end_time = segment['end_time'] / 1000
        ax_rate.axvspan(start_time, end_time, alpha=0.2, color='r')
        
    # 3. Plot frequency-specific metrics
    frequencies = [segment['frequency'] for segment in segments]
    mean_rates = [segment['mean_rate'] for segment in population_metrics['segment_responses']]
    
    # Power in the input frequency band
    band_powers = []
    for segment in population_metrics['segment_responses']:
        if 'psd_freqs' in segment and len(segment['psd_freqs']) > 0:
            # Find power at the input frequency
            freq = segment['frequency']
            freqs = segment['psd_freqs']
            psd = segment['psd']
            
            # Get closest frequency bin
            idx = np.argmin(np.abs(freqs - freq))
            if idx < len(psd):
                band_powers.append(psd[idx])
            else:
                band_powers.append(0)
        else:
            band_powers.append(0)
    
    # 4. Plot spike raster (for a subset of cells)
    ax_raster = fig.add_subplot(gs[2, 0])
    
    cell_metrics = processed_data['cell_metrics']
    gids = list(cell_metrics.keys())
    
    # Sort cells by overall firing rate
    sorted_gids = sorted(gids, 
                         key=lambda g: cell_metrics[g]['firing_rate'], 
                         reverse=True)
    
    # Plot top 50 most active cells
    max_cells = min(50, len(sorted_gids))
    for i, gid in enumerate(sorted_gids[:max_cells]):
        spike_times = cell_metrics[gid]['spike_times'] / 1000  # convert to seconds
        ax_raster.plot(spike_times, np.ones_like(spike_times) * i, '|', markersize=3)
    
    ax_raster.set_title(f'Spike Raster (Top {max_cells} Cells)')
    ax_raster.set_xlabel('Time (s)')
    ax_raster.set_ylabel('Cell #')
    
    # Add segment markers
    for segment in segments:
        start_time = segment['start_time'] / 1000  # convert to seconds
        end_time = segment['end_time'] / 1000
        ax_raster.axvspan(start_time, end_time, alpha=0.2, color='r')
    
    # 5. Plot frequency tuning curve
    ax_tuning = fig.add_subplot(gs[3, 0])
    ax_tuning.plot(frequencies, mean_rates, 'o-', linewidth=2, markersize=10)
    ax_tuning.set_title('Population Frequency Response')
    ax_tuning.set_xlabel('Input Frequency (Hz)')
    ax_tuning.set_ylabel('Mean Firing Rate (Hz)')
    ax_tuning.set_xscale('log')
    ax_tuning.grid(True)
    
    # 6. Plot power at input frequency
    ax_power = fig.add_subplot(gs[3, 1])
    ax_power.bar(range(len(frequencies)), band_powers, color='cornflowerblue')
    ax_power.set_xticks(range(len(frequencies)))
    ax_power.set_xticklabels([f"{f} Hz" for f in frequencies])
    ax_power.set_title('Power at Input Frequency')
    ax_power.set_ylabel('Power (a.u.)')
    
    # 7. Plot single-cell frequency tuning (for a few example cells)
    ax_cell_tuning = fig.add_subplot(gs[0:2, 1])
    
    # Select a few cells with good responses
    example_cells = sorted_gids[:5] if len(sorted_gids) >= 5 else sorted_gids
    
    for gid in example_cells:
        cell_firing_rates = [segment_response['firing_rate'] 
                            for segment_response in cell_metrics[gid]['segment_responses']]
        ax_cell_tuning.plot(frequencies, cell_firing_rates, 'o-', label=f'Cell {gid}')
    
    ax_cell_tuning.set_title('Single Cell Frequency Tuning')
    ax_cell_tuning.set_xlabel('Input Frequency (Hz)')
    ax_cell_tuning.set_ylabel('Firing Rate (Hz)')
    ax_cell_tuning.set_xscale('log')
    ax_cell_tuning.legend()
    
    # 8. Plot population statistics
    ax_stats = fig.add_subplot(gs[2, 1])
    
    # Calculate proportion of active cells per segment
    active_fractions = []
    for i, segment in enumerate(segments):
        n_active = sum(1 for gid in cell_metrics if 
                       len(cell_metrics[gid]['segment_responses'][i]['spikes']) > 0)
        active_fractions.append(n_active / len(cell_metrics) if len(cell_metrics) > 0 else 0)
    
    ax_stats.bar(range(len(frequencies)), active_fractions, color='lightgreen')
    ax_stats.set_xticks(range(len(frequencies)))
    ax_stats.set_xticklabels([f"{f} Hz" for f in frequencies])
    ax_stats.set_title('Fraction of Active Cells')
    ax_stats.set_ylabel('Active Fraction')
    
    plt.tight_layout()
    return fig    



def plot_transfer_function_analysis(processed_data):
    """
    Analyze how model transforms input frequency information.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed responses from process_model_temporal_responses
    """
    input_metadata = processed_data['input_metadata']
    population_metrics = processed_data['population_metrics']
    
    segments = input_metadata['segments']
    segment_responses = population_metrics['segment_responses']
    
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2)
    
    # 1. Input amplitude for each segment
    input_signal = input_metadata['signal']
    input_amplitudes = []
    
    for segment in segments:
        start_idx = segment['start_idx']
        end_idx = segment['end_idx']
        
        segment_signal = input_signal[start_idx:end_idx]
        amplitude = np.std(segment_signal)
        input_amplitudes.append(amplitude)
    
    # 2. Output amplitude (spike rate stdev)
    output_amplitudes = []
    
    for segment in segment_responses:
        if 'population_rate' in segment and len(segment['population_rate']) > 0:
            # Use standard deviation of the population rate as amplitude
            amplitude = np.std(segment['population_rate'])
            output_amplitudes.append(amplitude)
        else:
            output_amplitudes.append(0)
    
    # 3. Gain (output/input ratio)
    gains = []
    for out_amp, in_amp in zip(output_amplitudes, input_amplitudes):
        if in_amp > 0:
            gains.append(out_amp / in_amp)
        else:
            gains.append(0)
    
    # 4. Plot frequency response (gain)
    frequencies = [segment['frequency'] for segment in segments]
    
    ax_gain = fig.add_subplot(gs[0, 0])
    ax_gain.plot(frequencies, gains, 'o-', linewidth=2, markersize=10)
    ax_gain.set_title('Population Frequency Response (Gain)')
    ax_gain.set_xlabel('Input Frequency (Hz)')
    ax_gain.set_ylabel('Gain (Output/Input Amplitude Ratio)')
    ax_gain.set_xscale('log')
    ax_gain.grid(True)
    
    # 5. Plot input-output relationship for each frequency
    ax_io = fig.add_subplot(gs[0, 1])
    
    for i, freq in enumerate(frequencies):
        ax_io.plot(input_amplitudes[i], output_amplitudes[i], 'o', 
                  markersize=12, label=f'{freq} Hz')
    
    if len(input_amplitudes) > 1:
        
        slope, intercept, r_value, p_value, std_err = linregress(input_amplitudes, output_amplitudes)
        
        x_range = np.linspace(min(input_amplitudes), max(input_amplitudes), 100)
        y_fit = slope * x_range + intercept
        
        ax_io.plot(x_range, y_fit, 'k--', alpha=0.7, 
                  label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_value**2:.2f}')
    
    ax_io.set_title('Input-Output Relationship')
    ax_io.set_xlabel('Input Amplitude')
    ax_io.set_ylabel('Output Amplitude (Rate SD)')
    ax_io.legend()
    ax_io.grid(True)
    
    # 6. Plot population rate response vs. input frequency
    ax_rate = fig.add_subplot(gs[1, 0])
    
    mean_rates = [segment['mean_rate'] for segment in segment_responses]
    
    ax_rate.plot(frequencies, mean_rates, 'o-', linewidth=2, markersize=10)
    ax_rate.set_title('Mean Firing Rate vs. Input Frequency')
    ax_rate.set_xlabel('Input Frequency (Hz)')
    ax_rate.set_ylabel('Mean Firing Rate (Hz)')
    ax_rate.set_xscale('log')
    ax_rate.grid(True)
    
    # 7. Plot synchrony/reliability metrics
    ax_sync = fig.add_subplot(gs[1, 1])
    
    # Calculate firing synchrony/reliability across cells for each frequency
    
    cv_across_cells = []
    
    for i, segment in enumerate(segments):
        rates = []
        
        for gid in processed_data['cell_metrics']:
            segment_response = processed_data['cell_metrics'][gid]['segment_responses'][i]
            rates.append(segment_response['firing_rate'])
        
        # CV of firing rates across cells (higher = more variable = less synchronous)
        # Coefficient of variation = std/mean
        if np.mean(rates) > 0:
            cv = variation(rates)
        else:
            cv = 0
            
        cv_across_cells.append(cv)
    
    ax_sync.plot(frequencies, cv_across_cells, 'o-', linewidth=2, markersize=10)
    ax_sync.set_title('Variability of Firing Rates Across Cells')
    ax_sync.set_xlabel('Input Frequency (Hz)')
    ax_sync.set_ylabel('Coefficient of Variation Across Cells')
    ax_sync.set_xscale('log')
    ax_sync.grid(True)
    
    plt.tight_layout()
    return fig


def plot_information_theoretic_analysis(processed_data):
    """
    Information-theoretic analysis of model responses to temporal features.
    
    Parameters:
    -----------
    processed_data : Dict
        Processed responses from process_model_temporal_responses
    """
    
    input_metadata = processed_data['input_metadata']
    cell_metrics = processed_data['cell_metrics']
    
    segments = input_metadata['segments']
    
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2)
    
    # 1. Calculate mutual information between input frequency and cell firing rates
    frequencies = np.array([segment['frequency'] for segment in segments])
    n_segments = len(frequencies)
    
    # Get active cells
    gids = list(cell_metrics.keys())
    active_gids = [gid for gid in gids if cell_metrics[gid]['n_spikes'] > 0]
    
    if len(active_gids) < 5:
        plt.text(0.5, 0.5, "Not enough active cells for information analysis", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Calculate mutual information for each cell
    cell_mi_values = []
    
    for gid in tqdm(active_gids):
        # Get firing rates for each segment
        rates = np.array([cell_metrics[gid]['segment_responses'][i]['firing_rate'] 
                         for i in range(n_segments)])
        
        # Discretize rates and frequencies for MI calculation
        n_bins = min(5, n_segments)
        
        # Skip cells with no variation in firing rate
        if np.std(rates) == 0:
            cell_mi_values.append(0)
            continue
        
        # Calculate mutual information using scikit-learn
        try:
            mi = mutual_info_regression(frequencies.reshape(-1, 1), rates)[0]
            cell_mi_values.append(mi)
        except:
            # Fallback to discretized calculation if regression fails
            rate_bins = np.linspace(min(rates), max(rates), n_bins+1)
            freq_bins = np.linspace(min(frequencies), max(frequencies), n_bins+1)
            
            rate_discrete = np.digitize(rates, rate_bins)
            freq_discrete = np.digitize(frequencies, freq_bins)
            
            mi = mutual_info_score(freq_discrete, rate_discrete)
            cell_mi_values.append(mi)
    
    # Plot distribution of MI values
    ax_mi_dist = fig.add_subplot(gs[0, 0])
    
    ax_mi_dist.hist(cell_mi_values, bins=20, color='cornflowerblue', alpha=0.7)
    ax_mi_dist.set_title('Distribution of Mutual Information Values')
    ax_mi_dist.set_xlabel('Mutual Information (bits)')
    ax_mi_dist.set_ylabel('Number of Cells')
    
    # Calculate informative cells (above threshold)
    mi_threshold = np.mean(cell_mi_values) + np.std(cell_mi_values)
    informative_cells = [gid for i, gid in enumerate(active_gids) 
                         if i < len(cell_mi_values) and cell_mi_values[i] > mi_threshold]
    
    # 2. Plot firing rate vs. frequency for most informative cells
    ax_info_cells = fig.add_subplot(gs[0, 1])
    
    # Select top 5 most informative cells
    if cell_mi_values:
        sorted_indices = np.argsort(cell_mi_values)[::-1]
        top_indices = sorted_indices[:5]
        top_cells = [active_gids[i] for i in top_indices]
        
        for gid in top_cells:
            rates = [cell_metrics[gid]['segment_responses'][i]['firing_rate'] 
                    for i in range(n_segments)]
            ax_info_cells.plot(frequencies, rates, 'o-', label=f'Cell {gid}')
        
        ax_info_cells.set_title('Frequency Tuning of Most Informative Cells')
        ax_info_cells.set_xlabel('Input Frequency (Hz)')
        ax_info_cells.set_ylabel('Firing Rate (Hz)')
        ax_info_cells.set_xscale('log')
        ax_info_cells.legend()
        ax_info_cells.grid(True)
    else:
        ax_info_cells.text(0.5, 0.5, "No cells with mutual information data", 
                          ha='center', va='center')
    
    # 3. Mutual information as a function of firing rate
    ax_mi_vs_rate = fig.add_subplot(gs[1, 0])
    
    mean_rates = [np.mean([cell_metrics[gid]['segment_responses'][i]['firing_rate'] 
                          for i in range(n_segments)]) 
                 for gid in active_gids]
    
    ax_mi_vs_rate.scatter(mean_rates, cell_mi_values, alpha=0.7, s=50)
    ax_mi_vs_rate.set_title('Mutual Information vs. Mean Firing Rate')
    ax_mi_vs_rate.set_xlabel('Mean Firing Rate (Hz)')
    ax_mi_vs_rate.set_ylabel('Mutual Information (bits)')
    ax_mi_vs_rate.grid(True)
    
    # Add regression line
    if len(mean_rates) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(mean_rates, cell_mi_values)
        
        x_range = np.linspace(min(mean_rates), max(mean_rates), 100)
        y_fit = slope * x_range + intercept
        
        ax_mi_vs_rate.plot(x_range, y_fit, 'r--', 
                          label=f'Fit: R²={r_value**2:.2f}, p={p_value:.4f}')
        ax_mi_vs_rate.legend()
    
    # 4. Calculate and plot population-level mutual information estimates
    ax_pop_mi = fig.add_subplot(gs[1, 1])
    
    # Calculate summary statistics for segment responses
    summary_stats = []
    for i in range(n_segments):
        freq = frequencies[i]
        rates = [cell_metrics[gid]['segment_responses'][i]['firing_rate'] for gid in active_gids]
        
        # Create summary statistics
        summary_stats.append({
            'frequency': freq,
            'mean_rate': np.mean(rates) if rates else 0,
            'max_rate': np.max(rates) if rates else 0,
            'cv': variation(rates) if rates and np.mean(rates) > 0 else 0,
            'fraction_active': sum(r > 0 for r in rates) / len(rates) if rates else 0
        })
    
    # Calculate MI between frequency and each statistic
    stats_to_analyze = ['mean_rate', 'max_rate', 'cv', 'fraction_active']
    mi_values = []
    
    for stat in stats_to_analyze:
        stat_values = np.array([summary_stats[i][stat] for i in range(n_segments)])
        
        if np.std(stat_values) == 0:
            mi_values.append(0)
            continue
            
        try:
            mi = mutual_info_regression(frequencies.reshape(-1, 1), stat_values)[0]
            mi_values.append(mi)
        except:
            # Fallback method if regression fails
            n_bins = min(5, n_segments)
            
            stat_bins = np.linspace(min(stat_values), max(stat_values), n_bins+1)
            freq_bins = np.linspace(min(frequencies), max(frequencies), n_bins+1)
            
            stat_discrete = np.digitize(stat_values, stat_bins)
            freq_discrete = np.digitize(frequencies, freq_bins)
            
            mi = mutual_info_score(freq_discrete, stat_discrete)
            mi_values.append(mi)
    
    # Create bar chart
    ax_pop_mi.bar(stats_to_analyze, mi_values, color='lightgreen')
    ax_pop_mi.set_title('Mutual Information by Population Statistic')
    ax_pop_mi.set_xlabel('Population Statistic')
    ax_pop_mi.set_ylabel('Mutual Information (bits)')
    ax_pop_mi.set_xticklabels([s.replace('_', ' ').title() for s in stats_to_analyze])
    
    plt.tight_layout()
    return fig


def analyze_temporal_responses(model_output_path,
                               model_output_namespace_id,
                               input_features_path,
                               input_signal_id,
                               populations: Optional[List] = None,
                               time_range=None,
                               time_variable="t",
                               include_artificial=True,
                               output_dir=None,
                               analyses=['feature_transformation',
                                         'transfer_function',
                                         'information_theory']):
    """
    Analysis of model responses to temporal feature stimuli.
    
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
    output_dir : str, optional
        Directory to save output figures.
    
    Returns:
    --------
    Dict
        Dictionary containing processed data and figures
    """
    # 1. Preprocess the responses
    processed_data = process_model_temporal_responses(
        model_output_path = model_output_path,
        model_output_namespace_id = model_output_namespace_id,
        input_features_path = input_features_path,
        input_signal_id = input_signal_id,
        populations = populations,
        time_range=time_range,
        time_variable=time_variable,
        include_artificial=include_artificial
    )
    
    # 2. Generate all the analysis figures
    figures = {}

    if 'feature_transformation' in analyses:
        print("Generating input-output feature transformation plot...")
        figures['feature_transformation'] = plot_input_output_feature_transformation(processed_data)
    
    if 'transfer_function' in analyses:
        print("Generating transfer function analysis plot...")
        figures['transfer_function'] = plot_transfer_function_analysis(processed_data)
    
    if 'information_theory' in analyses:
        print("Generating information theoretic analysis plot...")
        figures['information_theory'] = plot_information_theoretic_analysis(processed_data)
    
    # 3. Save figures if output directory is provided
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
    
    analyze_temporal_responses(#model_output_path = "./results/CA1_Slice_100_Temporal_Input_10s_20250513.h5",
        model_output_path = "./results/Full_Scale_Temporal_Features_7106299/Full_Scale_results.h5",
        model_output_namespace_id = "Spike Events",
        input_features_path = "./input/EC_temporal_input_spike_trains_10s.h5",
        input_signal_id = "test_temporal_features_20240510",
        populations = ["PYR"],
        include_artificial = False,
        output_dir="figures/full_scale",
        analyses=['feature_transformation'])

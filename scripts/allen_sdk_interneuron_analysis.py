#!/usr/bin/env python3
"""
Allen Brain Observatory GABAergic Interneuron Spike Train Analysis

This script uses the AllenSDK to extract and analyze spike trains from
identified GABAergic interneurons for CA1 model optimization.

"""

import gc
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml

# Allen SDK imports
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys import ecephys_session
from allensdk.core.brain_observatory_cache import BrainObservatoryCache


def compute_spike_statistics_per_time_bin(
    spike_data: Dict,
    bin_size: float = 0.50,  # 50ms ~ gamma cycle
    interneuron_type: Optional[str] = None,
    brain_area: Optional[str] = None,
    min_spikes_threshold: int = 1
) -> Dict:
    """
    Compute spike statistics per time bin

    Parameters:
   -----------
    spike_data : Dict
        Nested spike data structure or spike_times_dict
    bin_size : float
        Time bin size in seconds (default: 0.125s for theta cycle)
    interneuron_type : str, optional
        Specific interneuron type to analyze
    brain_area : str, optional
        Specific brain area to analyze
    min_spikes_threshold : int
        Minimum spikes in bin to consider neuron "active"

    Returns:
    --------
    Dict
        Time bin analysis results
    """

    # Extract relevant spike times
    all_spike_times = []
    neuron_spike_times = {}
    neuron_labels = {}

    # Handle different input formats
    if isinstance(spike_data, dict) and 'spike_times' in str(spike_data):
        # Direct spike_times_dict format
        spike_times_dict = spike_data
        print(f"spike_times_dict = {spike_times_dict}")
        for unit_id, unit_data in spike_times_dict.items():
            spike_times = unit_data.get('spike_times', np.array([]))
            neuron_spike_times[unit_id] = spike_times
            if len(spike_times) > 0:
                all_spike_times.extend(spike_times)
            neuron_labels[unit_id] = {'type': 'unknown', 'area': 'unknown'}

    else:
        # Nested spike_data format
        for itype, areas in spike_data.items():
            # Filter by interneuron type if specified
            if interneuron_type and itype != interneuron_type:
                continue

            for area, units in areas.items():
                # Filter by brain area if specified
                if brain_area and area != brain_area:
                    continue

                for unit_id, unit_data in units.items():
                    spike_times = unit_data.get('spike_times', np.array([]))
                    print(f"unit {unit_id}: unit_data = {unit_data} spike_times = {spike_times}")
                    neuron_spike_times[unit_id] = spike_times
                    if len(spike_times) > 0:
                        all_spike_times.extend(spike_times)
                    neuron_labels[unit_id] = {'type': itype, 'area': area}

    if not neuron_spike_times:
        return {'error': 'No spike data found for specified criteria'}

    if not all_spike_times:
        return {'error': 'No spikes found in data'}

    all_spike_times = np.asarray(all_spike_times)

    recording_start = np.min(all_spike_times)
    recording_end = np.max(all_spike_times)
    recording_duration = recording_end - recording_start

    n_bins = int(np.ceil(recording_duration / bin_size))
    bin_edges = np.linspace(recording_start, recording_start + n_bins * bin_size, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    n_neurons = len(neuron_spike_times)
    fraction_active_per_bin = np.zeros(n_bins)
    active_count_per_bin = np.zeros(n_bins)

    for bin_idx in range(n_bins):
        bin_start = bin_edges[bin_idx]
        bin_end = bin_edges[bin_idx + 1]

        active_count = 0

        for unit_id, spike_times in neuron_spike_times.items():

            spikes_in_bin = np.sum((spike_times >= bin_start) & (spike_times < bin_end))

            if spikes_in_bin >= min_spikes_threshold:
                active_count += 1

        active_count_per_bin[bin_idx] = active_count
        fraction_active_per_bin[bin_idx] = active_count / n_neurons if n_neurons > 0 else 0.0

    results = {
        'bin_centers': bin_centers,
        'bin_edges': bin_edges,
        'bin_size': bin_size,
        'fraction_active_per_bin': fraction_active_per_bin,
        'active_count_per_bin': active_count_per_bin,
        'n_neurons': n_neurons,
        'n_bins': n_bins,
        'recording_duration': recording_duration,
        'recording_start': recording_start,
        'recording_end': recording_end,

        'mean_fraction_active': np.mean(fraction_active_per_bin),
        'std_fraction_active': np.std(fraction_active_per_bin),
        'median_fraction_active': np.median(fraction_active_per_bin),
        'min_fraction_active': np.min(fraction_active_per_bin),
        'max_fraction_active': np.max(fraction_active_per_bin),

        # Distribution analysis
        'fraction_active_percentiles': {
            '25th': np.percentile(fraction_active_per_bin, 25),
            '75th': np.percentile(fraction_active_per_bin, 75),
            '90th': np.percentile(fraction_active_per_bin, 90),
            '95th': np.percentile(fraction_active_per_bin, 95)
        },

        # Burst detection (high activity periods)
        'high_activity_threshold': np.mean(fraction_active_per_bin) + 2 * np.std(fraction_active_per_bin),
        'quiet_period_threshold': np.mean(fraction_active_per_bin) - np.std(fraction_active_per_bin),

        # Additional metadata
        'criteria': {
            'interneuron_type': interneuron_type,
            'brain_area': brain_area,
            'min_spikes_threshold': min_spikes_threshold
        }
    }

    high_activity_bins = fraction_active_per_bin > results['high_activity_threshold']
    quiet_period_bins = fraction_active_per_bin < results['quiet_period_threshold']

    results['burst_analysis'] = {
        'n_high_activity_bins': np.sum(high_activity_bins),
        'fraction_time_high_activity': np.mean(high_activity_bins),
        'n_quiet_bins': np.sum(quiet_period_bins),
        'fraction_time_quiet': np.mean(quiet_period_bins),
        'high_activity_bin_indices': np.where(high_activity_bins)[0],
        'quiet_period_bin_indices': np.where(quiet_period_bins)[0]
    }

    return results


    
def is_optotagged(unit_id: int,
                  session,
                  response_window: float = 0.002,  # 2ms response window
                  min_trials: int = 3,
                  p_threshold: float = 0.1) -> bool:
    """
    Check if a unit responds to optogenetic stimulation

    Parameters:
    -----------
    unit_id : int
        Unit ID to check
    session : EcephysSession
        Allen session object
    response_window : float
        Time window after stimulus onset to check for response (seconds)
    min_trials : int
        Minimum number of trials required for statistical test
    p_threshold : float
        P-value threshold for significance
            
    Returns:
    --------
    bool : True if unit is light-responsive
    """
    try:
        # Get optogenetic stimulation timestamps
        optostim_epochs = session.optogenetic_stimulation_epochs

        # only choose epochs with 10ms duration, as those provide a
        # long enough artifact-free window to observe light-evoked
        # spikes, but do not last long enough to be contaminated by
        # visually driven activity
        
        optostim_epochs = optostim_epochs[(optostim_epochs.duration > 0.009) & \
            (optostim_epochs.duration < 0.02)]

        if optostim_epochs is None or len(optostim_epochs) == 0:
            return False

        # Get spike times for this unit
        spike_times = session.spike_times[unit_id]

        # Extract all stimulation start times and durations
        stim_starts = optostim_epochs['start_time'].values
        stim_ends = optostim_epochs['stop_time'].values
        
        # spike counting for stimulation periods
        stim_spike_counts = count_spikes_in_windows(
            spike_times, stim_starts + response_window, stim_ends
        )

        # spike counting for baseline periods  
        baseline_starts = stim_starts - optostim_epochs.duration
        baseline_ends = stim_starts  # Use pre-stimulus period
        baseline_spike_counts = count_spikes_in_windows(
            spike_times, baseline_starts, baseline_ends
        )

        print(f"Unit {unit_id} has {np.sum(baseline_spike_counts)} baseline spikes, {np.sum(stim_spike_counts)} stim spikes")
        # Statistical test
        if len(stim_spike_counts) >= min_trials:
            if np.sum(stim_spike_counts) == 0 and np.sum(baseline_spike_counts) == 0:
                return False  # No spikes in either condition
            
            # Use Wilcoxon signed-rank test or simple comparison
            if np.sum(stim_spike_counts != baseline_spike_counts) >= min_trials // 2:
                try:
                    _, p_value = stats.wilcoxon(
                        stim_spike_counts, baseline_spike_counts,
                        alternative='greater', zero_method='zsplit'
                    )
                    print(f"Unit {unit_id} p-value is {p_value}")
                    return p_value < p_threshold
                except ValueError:
                    # Fallback to simple threshold if statistical test fails
                    return np.mean(stim_spike_counts) > 2 * np.mean(baseline_spike_counts)
            else:
                return False
        else:
            # Simple threshold for few trials
            return np.mean(stim_spike_counts) > 2 * np.mean(baseline_spike_counts)

        return False

    except Exception as e:
        print(f"Error checking optotagging for unit {unit_id}: {e}")
        return False

def count_spikes_in_windows(
        spike_times: np.ndarray,
        window_starts: np.ndarray,
        window_ends: np.ndarray
) -> np.ndarray:
    """
    Count spikes using searchsorted
    """
    
    # Sort spike times if not already sorted
    if not np.all(spike_times[:-1] <= spike_times[1:]):
        spike_times = np.sort(spike_times)
        
    # Find indices where windows would be inserted in sorted spike array
    start_indices = np.searchsorted(spike_times, window_starts, side='left')
    end_indices = np.searchsorted(spike_times, window_ends, side='left')
    
    # Count spikes in each window
    spike_counts = end_indices - start_indices
    
    return spike_counts

    
def analyze_waveform(unit_row: pd.Series) -> Dict:
    """
    Analyze spike waveform characteristics

    Parameters:
    -----------
    unit_row : pd.Series
        Row from units table

    Returns:
    --------
    Dict : Waveform analysis results
    """
    try:
        # Get waveform duration (trough-to-peak time)
        waveform_duration = unit_row.get('waveform_duration', np.nan)

        # PV+ interneurons typically have narrow waveforms (<0.5 ms)
        is_narrow = waveform_duration < 0.5 if not np.isnan(waveform_duration) else False

        return {
            'is_narrow_waveform': is_narrow,
            'waveform_duration': waveform_duration,
            'amplitude': unit_row.get('amplitude', np.nan)
        }

    except Exception:
        return {
            'is_narrow_waveform': False,
            'waveform_duration': np.nan,
            'amplitude': np.nan
        }


def compute_population_statistics(spike_data: Dict, bin_size: float = 50.) -> Dict:
    """
    Compute population statistics for model optimization

    Parameters:
    -----------
    spike_data : Dict
        Spike train data from extract_spike_trains()

    Returns:
    --------
    Dict : Population statistics by interneuron type
    """
    stats = {}

    for interneuron_type, areas in spike_data.items():

        if not interneuron_type in ['PV+', 'SST+', 'VIP+']:
            continue
        
        print(f"Computing {interneuron_type} population statistics...")
        type_stats = {
            'n_units': 0,
            'firing_rates': [],
            'fraction_active': 0.0,
            'mean_firing_rate': 0.0,
            'std_firing_rate': 0.0,
            'areas': {}
        }

        all_firing_rates = []
        active_units = 0
        total_units = 0

        for area, units in areas.items():
            area_firing_rates = []

            print(f"Processing {len(units)} {interneuron_type} units...")
            for unit_id, unit_data in units.items():
                firing_rate = unit_data['firing_rate']
                if not np.isnan(firing_rate):
                    all_firing_rates.append(firing_rate)
                    area_firing_rates.append(firing_rate)

                    if firing_rate > 0.1:  # Active threshold: >0.1 Hz
                        active_units += 1
                    total_units += 1

            type_stats['areas'][area] = {
                'n_units': len(area_firing_rates),
                'mean_firing_rate': np.mean(area_firing_rates) if area_firing_rates else 0.0,
                'std_firing_rate': np.std(area_firing_rates) if area_firing_rates else 0.0
            }

        # Overall statistics
        type_stats['n_units'] = total_units
        type_stats['firing_rates'] = all_firing_rates
        type_stats['fraction_active'] = active_units / total_units if total_units > 0 else 0.0
        type_stats['mean_firing_rate'] = np.mean(all_firing_rates) if all_firing_rates else 0.0
        type_stats['std_firing_rate'] = np.std(all_firing_rates) if all_firing_rates else 0.0

        # Area-specific time bin analysis
        for area in areas.keys():
            area_time_bin_results = compute_spike_statistics_per_time_bin(
                areas[area],
                bin_size=bin_size,
                interneuron_type=interneuron_type,
                brain_area=area
            )

            if 'error' not in area_time_bin_results:
                type_stats['areas'][area]['time_bin_analysis'] = {
                    'bin_size': bin_size,
                    'mean_fraction_active_per_bin': area_time_bin_results['mean_fraction_active'],
                    'std_fraction_active_per_bin': area_time_bin_results['std_fraction_active'],
                    'burst_fraction_time': area_time_bin_results['burst_analysis']['fraction_time_high_activity']
                }

        stats[interneuron_type] = type_stats

    return stats


def plot_firing_rate_distributions(spike_data: Dict):

    for interneuron_type, areas in spike_data.items():

        interneuron_dict = spike_data[interneuron_type]
        interneuron_records = []

        # Iterate through brain areas
        for brain_area, units in interneuron_dict.items():
            # Iterate through units in this brain area
            for unit_id, unit_data in units.items():

                # obtain summary information about unit
                record = {
                    'unit_id': unit_id,
                    'brain_area': brain_area,
                    'interneuron_type': interneuron_type,
                    'confidence': unit_data['confidence'],
                    'n_spikes': unit_data['n_spikes'],
                    'firing_rate': unit_data['firing_rate'],
                    'waveform_duration': unit_data['waveform_duration']
                }

                interneuron_records.append(record)

        # Convert to DataFrame
        interneuron_df = pd.DataFrame(interneuron_records)
        if interneuron_df.empty:
            continue

        
        # Set appropriate data types
        if not interneuron_df.empty:
            interneuron_df = interneuron_df.astype({
                'unit_id': 'int64',
                'brain_area': 'category',
                'interneuron_type': 'category'
            })

        # Set numeric columns
        numeric_columns = ['firing_rate', 'n_spikes', 'confidence', 'waveform_duration']
        for col in numeric_columns:
            if col in interneuron_df.columns:
                interneuron_df[col] = pd.to_numeric(interneuron_df[col], errors='coerce')

        print(interneuron_df)

        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.hist(interneuron_df['firing_rate'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Firing Rate (Hz)')
        plt.ylabel('Count')
        plt.title(f'{interneuron_type} Interneuron Firing Rate Distribution')

        plt.subplot(1, 2, 2)
        for area in interneuron_df['brain_area'].unique():
            area_data = interneuron_df[interneuron_df['brain_area'] == area]
            plt.hist(area_data['firing_rate'], bins=15, alpha=0.6, label=area)
        plt.xlabel('Firing Rate (Hz)')
        plt.ylabel('Count')
        plt.title('Firing Rates by Brain Area')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{interneuron_type}_firing_rates.png', dpi=300, bbox_inches='tight')
        plt.show()

def export_optimization_targets(stats: Dict, filepath: str) -> None:
    """
    Save firing rate and fraction active targets for model optimization

    Parameters:
    -----------
    stats : Dict
        Population statistics from compute_population_statistics()
    filepath : str
        Path to save optimization targets
    """
    targets = {}

    for interneuron_type, type_stats in stats.items():
        targets[interneuron_type] = {}
        for area, area_stats in type_stats['areas'].items():
            area_targets = {
                'mean_firing_rate_target': float(area_stats['mean_firing_rate']),
                'std_firing_rate_target': float(area_stats['std_firing_rate']),
                'bin_size': float(area_stats['time_bin_analysis']['bin_size']),
                'mean_fraction_active_per_bin': float(area_stats['time_bin_analysis']['mean_fraction_active_per_bin']),
                'std_fraction_active_per_bin': float(area_stats['time_bin_analysis']['std_fraction_active_per_bin'])
            }
            targets[interneuron_type][area] = area_targets

    # Save as both YAML and CSV for different use cases
    with open(f"{filepath}.yaml", 'w') as f:
        yaml.dump(targets, f)

    # Also save as DataFrame for easy analysis
    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.to_csv(f"{filepath}.csv")

    print(f"Optimization targets saved to {filepath}.yaml and {filepath}.csv")
    
        
class GABAergicInterneuronAnalyzer:
    """
    Analyzer for GABAergic interneuron activity from Allen Neuropixels data
    """
    
    def __init__(self, manifest_path: str = "./ecephys_cache"):
        """
        Initialize the Allen SDK cache and analyzer
        
        Parameters:
        -----------
        manifest_path : str
            Path to store the Allen SDK cache
        """
        self.cache = EcephysProjectCache.from_warehouse(manifest=Path(manifest_path) / "manifest.json")
        self.sessions = None
        self.units_table = None
        
    def get_sessions_with_interneurons(self) -> pd.DataFrame:
        """
        Get all sessions that contain optotagged GABAergic interneurons
        
        Returns:
        --------
        pd.DataFrame : Sessions with interneuron data
        """
        # Get all sessions
        print("Getting recording sessions...")
        self.sessions = self.cache.get_session_table()
        print(f"Found {len(self.sessions)} sessions.")
        
        # Filter for sessions with transgenic lines that label interneurons
        interneuron_sessions = self.sessions[self.sessions['full_genotype'].str.contains('Pvalb|Sst|Vip', na=False)]
        
        print(f"Found {len(interneuron_sessions)} sessions with interneuron optotagging")
        print("\nGenotype distribution:")
        print(interneuron_sessions['full_genotype'].value_counts())
        
        return interneuron_sessions
    
    def get_hippocampal_interneurons(self, session_id: int) -> pd.DataFrame:
        """
        Extract GABAergic interneurons from hippocampal regions
        
        Parameters:
        -----------
        session_id : int
            Allen session ID
            
        Returns:
        --------
        pd.DataFrame : Units table filtered for hippocampal interneurons
        """
        # Get session data
        session = self.cache.get_session_data(session_id)
        
        # Get units table for this session
        units = session.units
        
        # Filter for hippocampal areas (CA1, CA3, DG)
        hippocampal_areas = ['CA1', 'CA3', 'DG']
        hippocampal_units = units[units['ecephys_structure_acronym'].isin(hippocampal_areas)]
        print(f"Session {session_id}: Found {len(hippocampal_units)} hippocampal units")
        
        # Filter for putative interneurons based on:
        # 1. Optotagging (light-responsive units)
        # 2. Narrow waveforms (fast-spiking for PV+ cells)
        # 3. High firing rates (typical of interneurons)
        
        interneurons = self._identify_interneurons(hippocampal_units, session)
        
        print(f"Session {session_id}: Found {len(interneurons)} hippocampal GABAergic interneurons")
        print(f"Brain areas: {interneurons['ecephys_structure_acronym'].value_counts().to_dict()}")
        
        return interneurons
    
    def _identify_interneurons(self, units: pd.DataFrame, session) -> pd.DataFrame:
        """
        Identify GABAergic interneurons using multiple criteria
        
        Parameters:
        -----------
        units : pd.DataFrame
            Units table from session
        session : EcephysSession
            Allen session object
            
        Returns:
        --------
        pd.DataFrame : Identified interneurons
        """
        interneurons = []
        
        # Get optotagging results
        optotag_table = session.optogenetic_stimulation_epochs
        session_genotype = self._get_interneuron_type_from_genotype(session)
        
        for unit_id, unit_row in units.iterrows():
            is_interneuron = False
            interneuron_type = "unknown"
            confidence = 0.0

            print(f"Identifying unit {unit_id} optotagging response...")
            print(f"session_genotype is {session_genotype}")
            
            # Optotagging response criterion
            if is_optotagged(unit_id, session):
                is_interneuron = True
                interneuron_type = session_genotype
                confidence = max(confidence, 0.8)  # High confidence from optotagging
            
            print(f"Identifying unit {unit_id} waveform characteristics ...")
            # Waveform characteristics (fast-spiking) criterion
            waveform_features = analyze_waveform(unit_row)
            if waveform_features['is_narrow_waveform']:
                if interneuron_type == "unknown":
                    if unit_row.get('firing_rate', 0) > 5.0:  # >5 Hz typical of interneurons
                        is_interneuron = True
                        interneuron_type = "PV+"  # Narrow waveforms typical of PV+ cells
                        confidence = max(confidence, 0.6)  # Medium confidence from waveform
            
            # High firing rate
            #if unit_row.get('firing_rate', 0) > 5.0:  # >5 Hz typical of interneurons
            #    is_interneuron = True
            #    if interneuron_type == "unknown":
            #        confidence = max(confidence, 0.4)  # Lower confidence from firing rate alone
            #    else:
            #        confidence = min(confidence + 0.25, 0.9)
            #print(f"Identifying unit {unit_id} firing rate: {unit_row.get('firing_rate')} Hz")
            
            # Add to interneurons list if sufficient confidence threshold
            if is_interneuron and confidence > 0.5:
                unit_data = unit_row.copy()
                unit_data['interneuron_type'] = interneuron_type
                unit_data['identification_confidence'] = confidence
                interneurons.append(unit_data)
        
        return pd.DataFrame(interneurons) if interneurons else pd.DataFrame()
    
    
    def _get_interneuron_type_from_genotype(self, session) -> str:
        """
        Determine interneuron type from session genotype
        """
        session_data = self.cache.get_session_data(session.ecephys_session_id)
        genotype = session_data.metadata.get('full_genotype', '')

        print(f"session genotype is {genotype}")
        if 'Pvalb' in genotype:
            return 'PV+'
        elif 'Sst' in genotype:
            return 'SST+'
        elif 'Vip' in genotype:
            return 'VIP+'
        else:
            return 'unknown'
    
    def extract_spike_trains(self, session_id: int, interneurons: pd.DataFrame) -> Dict:
        """
        Extract spike trains for identified interneurons
        
        Parameters:
        -----------
        session_id : int
            Allen session ID
        interneurons : pd.DataFrame
            Identified interneurons from get_hippocampal_interneurons()
            
        Returns:
        --------
        Dict : Spike trains by interneuron type and unit
        """
        session = self.cache.get_session_data(session_id)
        spike_data = {}
        
        for unit_id, unit_row in interneurons.iterrows():
            # Get spike times for this unit
            spike_times = session.spike_times[unit_id]
            
            interneuron_type = unit_row['interneuron_type']
            brain_area = unit_row['ecephys_structure_acronym']
            
            # Organize by type and area
            if interneuron_type not in spike_data:
                spike_data[interneuron_type] = {}
            if brain_area not in spike_data[interneuron_type]:
                spike_data[interneuron_type][brain_area] = {}

            session_duration = session.session_duration if hasattr(session, 'session_duration') else (spike_times.max() - spike_times.min()) if len(spike_times) > 0 else np.nan
            spike_data[interneuron_type][brain_area][unit_id] = {
                'spike_times': np.asarray(spike_times, dtype=np.float32),
                'n_spikes': len(spike_times),
                'firing_rate': len(spike_times) / session_duration,
                'confidence': unit_row['identification_confidence'],
                'waveform_duration': unit_row.get('waveform_duration', np.nan)
            }
        
        return spike_data
    

def main():
    """
    Main procedure of the GABAergic interneuron analyzer
    """
    # Initialize analyzer
    analyzer = GABAergicInterneuronAnalyzer()
    
    # Get sessions with interneuron data
    interneuron_sessions = analyzer.get_sessions_with_interneurons()
    
    # Process multiple sessions for robust statistics
    all_spike_data = {}
    all_stats = {}
    
    n_units = 0
    for session_id in interneuron_sessions.index:
        try:
            print(f"\nProcessing session {session_id}...")
            
            interneurons = analyzer.get_hippocampal_interneurons(session_id)

            print(f"interneurons: {interneurons[['interneuron_type', 'identification_confidence']]}")
            
            if len(interneurons) > 0:
                spike_data = analyzer.extract_spike_trains(session_id, interneurons)

                print(f"spike_data: {spike_data}")
                # Merge with overall data
                for itype, areas in spike_data.items():
                    if itype not in all_spike_data:
                        all_spike_data[itype] = {}
                    for area, units in areas.items():
                        if area not in all_spike_data[itype]:
                            all_spike_data[itype][area] = {}
                        all_spike_data[itype][area].update(units)
                        n_units += len(units)

            del(interneurons)
                        
            gc.collect()
            
        except Exception as e:
            print(f"Error processing session {session_id}: {e}")
            continue

    print(f"Processed {n_units} units\n")

    print(f"all_spike_data: {all_spike_data}")
    # Compute overall population statistics
    if all_spike_data:
        all_stats = compute_population_statistics(all_spike_data)
        print(f"all_stats: {all_stats}")
        
        for itype, stats in all_stats.items():
            print(f"\n{itype} Interneurons:")
            print("="*60)
            print(f"  Number of units: {stats['n_units']}")
            print(f"  Fraction active: {stats['fraction_active']:.3f}")
            print(f"  Mean firing rate: {stats['mean_firing_rate']:.2f} +/- {stats['std_firing_rate']:.2f} Hz")
            
            for area, area_stats in stats['areas'].items():
                print(f"    {area}: {area_stats['n_units']} units, "
                      f"{area_stats['mean_firing_rate']:.2f} +/- {area_stats['std_firing_rate']:.2f} Hz")

        # Save optimization targets
        export_optimization_targets(all_stats, "HC_interneuron_targets")

        plot_firing_rate_distributions(spike_data)

    else:
        print("No interneuron data found in processed sessions")

if __name__ == "__main__":
    main()

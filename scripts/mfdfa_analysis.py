import numpy as np
from scipy import signal as sp_signal
from mpi4py import MPI
from typing import Dict, List, Tuple, Optional, Callable
import warnings
from functools import partial
from itertools import repeat
import logging

logger = logging.getLogger("mfdfa_analysis")

def mfdfa_detrend_segment(segment: np.ndarray, order: int = 1) -> float:
    """
    Detrend a single segment using polynomial fitting.
    
    Parameters:
    -----------
    segment : np.ndarray
        Time series segment
    order : int
        Polynomial order for detrending (1=linear, 2=quadratic, etc.)
        
    Returns:
    --------
    float
        Root mean square fluctuation of detrended segment
    """
    if len(segment) < order + 1:
        return 0.0
    
    try:
        x = np.arange(len(segment))
        # Fit polynomial
        coeffs = np.polyfit(x, segment, order)
        trend = np.polyval(coeffs, x)
        
        # Calculate fluctuation
        detrended = segment - trend
        fluctuation = np.sqrt(np.mean(detrended**2))
        
        return fluctuation
    except (np.linalg.LinAlgError, ValueError):
        # Handle singular matrix or other numerical issues
        return 0.0


def compute_fluctuation_function(
    profile: np.ndarray, 
    scale: int, 
    q_values: np.ndarray,
    order: int = 1,
    overlap: bool = False
) -> np.ndarray:
    """
    Compute fluctuation function F(scale, q) for given scale and q values.
    
    Parameters:
    -----------
    profile : np.ndarray
        Integrated time series (profile)
    scale : int
        Segment length
    q_values : np.ndarray
        Array of q values for multifractal analysis
    order : int
        Polynomial order for detrending
    overlap : bool
        Whether to use overlapping segments (increases computation but may improve accuracy)
        
    Returns:
    --------
    np.ndarray
        F(scale, q) values for each q
    """
    n = len(profile)
    
    if scale >= n:
        return np.zeros(len(q_values))
    
    # Determine segmentation strategy
    if overlap:
        # Overlapping segments with 50% overlap
        step = max(1, scale // 2)
        n_segments = (n - scale) // step + 1
        segments = [profile[i:i+scale] for i in range(0, n - scale + 1, step)]
    else:
        # Non-overlapping segments
        n_segments = n // scale
        segments = [profile[i*scale:(i+1)*scale] for i in range(n_segments)]
    
    if n_segments == 0:
        return np.zeros(len(q_values))
    
    # Compute fluctuations for all segments
    fluctuations = []
    for segment in segments:
        if len(segment) == scale:  # Ensure segment is complete
            fluct = mfdfa_detrend_segment(segment, order)
            if fluct > 0:  # Only include non-zero fluctuations
                fluctuations.append(fluct)
    
    if len(fluctuations) == 0:
        return np.zeros(len(q_values))
    
    fluctuations = np.array(fluctuations)
    
    # Compute F(scale, q) for each q value
    f_sq = np.zeros(len(q_values))
    
    for i, q in enumerate(q_values):
        if q == 0:
            # Special case: q=0 requires logarithmic averaging
            log_fluct = np.log(fluctuations[fluctuations > 0])
            if len(log_fluct) > 0:
                f_sq[i] = np.exp(np.mean(log_fluct))
            else:
                f_sq[i] = 0.0
        else:
            # General case: F(s,q) = [mean(f^q)]^(1/q)
            if np.any(fluctuations > 0):
                mean_fluct_q = np.mean(fluctuations[fluctuations > 0]**q)
                if mean_fluct_q > 0:
                    f_sq[i] = mean_fluct_q**(1.0/q)
                else:
                    f_sq[i] = 0.0
            else:
                f_sq[i] = 0.0
    
    return f_sq


def compute_mfdfa_scaling(
    time_series: np.ndarray,
    q_values: np.ndarray = None,
    scales: np.ndarray = None,
    order: int = 1,
    overlap: bool = False,
    min_scale: int = 10,
    max_scale_ratio: float = 0.25
) -> Dict[str, np.ndarray]:
    """
    Compute MFDFA scaling analysis for a time series.
    
    Parameters:
    -----------
    time_series : np.ndarray
        Input time series
    q_values : np.ndarray, optional
        Array of q values (default: -5 to 5)
    scales : np.ndarray, optional
        Array of scales to analyze (default: logarithmically spaced)
    order : int
        Polynomial order for detrending
    overlap : bool
        Whether to use overlapping segments
    min_scale : int
        Minimum scale length
    max_scale_ratio : float
        Maximum scale as ratio of time series length
        
    Returns:
    --------
    dict
        Dictionary containing scaling results
    """
    if len(time_series) < min_scale * 2:
        return {
            'scales': np.array([]),
            'q_values': np.array([]),
            'fluctuation_function': np.array([]),
            'scaling_exponents': np.array([]),
            'multifractal_spectrum': {'alpha': np.array([]), 'f_alpha': np.array([])},
            'hurst_exponent': 0.0,
            'multifractality_strength': 0.0
        }
    
    # Default q values covering positive, negative, and zero
    if q_values is None:
        q_values = np.concatenate([
            np.linspace(-5, -0.5, 10),
            [0],
            np.linspace(0.5, 5, 10)
        ])
    
    # Default scales - logarithmically spaced
    if scales is None:
        max_scale = int(len(time_series) * max_scale_ratio)
        max_scale = max(min_scale, max_scale)
        n_scales = min(20, max_scale // min_scale)
        scales = np.unique(np.logspace(
            np.log10(min_scale), 
            np.log10(max_scale), 
            n_scales
        ).astype(int))
    
    # Remove mean and compute profile (cumulative sum)
    centered_series = time_series - np.mean(time_series)
    profile = np.cumsum(centered_series)
    
    # Compute fluctuation function for all scales and q values
    fluctuation_matrix = np.zeros((len(scales), len(q_values)))
    
    for i, scale in enumerate(scales):
        f_sq = compute_fluctuation_function(profile, scale, q_values, order, overlap)
        fluctuation_matrix[i, :] = f_sq
    
    # Compute scaling exponents h(q) by fitting log-log relationship
    scaling_exponents = np.zeros(len(q_values))
    
    for j, q in enumerate(q_values):
        f_values = fluctuation_matrix[:, j]
        
        # Only use scales where fluctuation > 0
        valid_mask = (f_values > 0) & (scales > 0)
        
        if np.sum(valid_mask) >= 3:  # Need at least 3 points for fitting
            log_scales = np.log10(scales[valid_mask])
            log_f = np.log10(f_values[valid_mask])
            
            try:
                # Linear fit in log-log space
                slope, intercept = np.polyfit(log_scales, log_f, 1)
                scaling_exponents[j] = slope
            except (np.linalg.LinAlgError, ValueError):
                scaling_exponents[j] = 0.0
        else:
            scaling_exponents[j] = 0.0
    
    # Compute multifractal spectrum using Legendre transform
    multifractal_spectrum = compute_multifractal_spectrum(q_values, scaling_exponents)
    
    # Extract key parameters
    hurst_exponent = scaling_exponents[q_values == 2.0]
    hurst_exponent = hurst_exponent[0] if len(hurst_exponent) > 0 else scaling_exponents[len(scaling_exponents)//2]
    
    # Multifractality strength (width of h(q))
    valid_h = scaling_exponents[scaling_exponents != 0]
    multifractality_strength = np.max(valid_h) - np.min(valid_h) if len(valid_h) > 1 else 0.0
    
    return {
        'scales': scales,
        'q_values': q_values,
        'fluctuation_function': fluctuation_matrix,
        'scaling_exponents': scaling_exponents,
        'multifractal_spectrum': multifractal_spectrum,
        'hurst_exponent': hurst_exponent,
        'multifractality_strength': multifractality_strength,
        'profile': profile,
        'mean_fluctuation': np.mean(fluctuation_matrix[fluctuation_matrix > 0]) if np.any(fluctuation_matrix > 0) else 0.0
    }


def compute_multifractal_spectrum(q_values: np.ndarray, h_q: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute multifractal spectrum f(α) from scaling exponents h(q) using Legendre transform.
    
    Parameters:
    -----------
    q_values : np.ndarray
        Array of q values
    h_q : np.ndarray
        Scaling exponents h(q)
        
    Returns:
    --------
    dict
        Dictionary with 'alpha' and 'f_alpha' arrays
    """
    # Remove invalid values
    valid_mask = (h_q != 0) & np.isfinite(h_q) & np.isfinite(q_values)
    
    if np.sum(valid_mask) < 3:
        return {'alpha': np.array([]), 'f_alpha': np.array([])}
    
    q_valid = q_values[valid_mask]
    h_valid = h_q[valid_mask]
    
    # Compute derivatives dh/dq numerically
    if len(q_valid) < 3:
        return {'alpha': np.array([]), 'f_alpha': np.array([])}
    
    # Use central differences for interior points
    dh_dq = np.zeros_like(h_valid)
    
    # Forward difference for first point
    if len(h_valid) > 1:
        dh_dq[0] = (h_valid[1] - h_valid[0]) / (q_valid[1] - q_valid[0])
    
    # Central difference for interior points
    for i in range(1, len(h_valid) - 1):
        dq = q_valid[i+1] - q_valid[i-1]
        if dq != 0:
            dh_dq[i] = (h_valid[i+1] - h_valid[i-1]) / dq
    
    # Backward difference for last point
    if len(h_valid) > 1:
        dh_dq[-1] = (h_valid[-1] - h_valid[-2]) / (q_valid[-1] - q_valid[-2])
    
    # Legendre transform: α = h(q) + q * dh/dq, f(α) = q * α - τ(q)
    # where τ(q) = q * h(q) - 1
    alpha = h_valid + q_valid * dh_dq
    tau_q = q_valid * h_valid - 1
    f_alpha = q_valid * alpha - tau_q
    
    # Remove non-physical values
    physical_mask = np.isfinite(alpha) & np.isfinite(f_alpha) & (f_alpha >= 0)
    
    return {
        'alpha': alpha[physical_mask],
        'f_alpha': f_alpha[physical_mask]
    }


def compute_spike_mfdfa_analysis(
    spike_times: np.ndarray,
    duration_ms: float,
    time_bin_ms: float = 50.0,
    q_values: np.ndarray = None,
    scales: np.ndarray = None,
    order: int = 1,
    overlap: bool = False,
    analysis_type: str = 'rate'
) -> Dict[str, any]:
    """
    Compute MFDFA analysis for spike train data.
    
    Parameters:
    -----------
    spike_times : np.ndarray
        Array of spike times in milliseconds
    duration_ms : float
        Total duration in milliseconds
    time_bin_ms : float
        Time bin size for rate computation
    q_values : np.ndarray, optional
        Array of q values for multifractal analysis
    scales : np.ndarray, optional
        Array of scales to analyze
    order : int
        Polynomial order for detrending
    overlap : bool
        Whether to use overlapping segments
    analysis_type : str
        Type of analysis: 'rate', 'isi', or 'count'
        
    Returns:
    --------
    dict
        MFDFA analysis results
    """
    if len(spike_times) == 0:
        return {
            'analysis_type': analysis_type,
            'valid_analysis': False,
            'n_spikes': 0,
            'scales': np.array([]),
            'q_values': np.array([]),
            'fluctuation_function': np.array([]),
            'scaling_exponents': np.array([]),
            'multifractal_spectrum': {'alpha': np.array([]), 'f_alpha': np.array([])},
            'hurst_exponent': 0.0,
            'multifractality_strength': 0.0
        }
    
    # Prepare time series based on analysis type
    if analysis_type == 'rate':
        # Convert to firing rate time series
        n_bins = int(duration_ms / time_bin_ms)
        bin_edges = np.linspace(0, duration_ms, n_bins + 1)
        spike_counts, _ = np.histogram(spike_times, bins=bin_edges)
        time_series = spike_counts / (time_bin_ms / 1000.0)  # Convert to Hz
        
    elif analysis_type == 'isi':
        # Use interspike intervals
        if len(spike_times) < 2:
            time_series = np.array([])
        else:
            time_series = np.diff(spike_times)
            
    elif analysis_type == 'count':
        # Use spike counts (without rate conversion)
        n_bins = int(duration_ms / time_bin_ms)
        bin_edges = np.linspace(0, duration_ms, n_bins + 1)
        spike_counts, _ = np.histogram(spike_times, bins=bin_edges)
        time_series = spike_counts
        
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")
    
    if len(time_series) < 20:  # Minimum requirement for meaningful MFDFA
        return {
            'analysis_type': analysis_type,
            'valid_analysis': False,
            'n_spikes': len(spike_times),
            'time_series_length': len(time_series),
            'scales': np.array([]),
            'q_values': np.array([]),
            'fluctuation_function': np.array([]),
            'scaling_exponents': np.array([]),
            'multifractal_spectrum': {'alpha': np.array([]), 'f_alpha': np.array([])},
            'hurst_exponent': 0.0,
            'multifractality_strength': 0.0
        }
    
    # Perform MFDFA analysis
    mfdfa_results = compute_mfdfa_scaling(
        time_series, q_values, scales, order, overlap
    )
    
    # Add metadata
    mfdfa_results['analysis_type'] = analysis_type
    mfdfa_results['valid_analysis'] = True
    mfdfa_results['n_spikes'] = len(spike_times)
    mfdfa_results['time_series_length'] = len(time_series)
    mfdfa_results['time_bin_ms'] = time_bin_ms
    
    return mfdfa_results


def compute_population_mfdfa_analysis(
    gid_spike_dict: Dict[int, np.ndarray],
    duration_ms: float,
    time_bin_ms: float = 50.0,
    q_values: np.ndarray = None,
    scales: np.ndarray = None,
    order: int = 1,
    overlap: bool = False,
    analysis_types: List[str] = ['rate'],
    comm=None,
    root: int = 0,
    progress_interval: int = 1000
) -> Dict[int, Dict[str, any]]:
    """
    Compute MFDFA analysis for all neurons in a population.
    
    Parameters:
    -----------
    gid_spike_dict : dict
        {neuron_gid: spike_times_array}
    duration_ms : float
        Total simulation duration in milliseconds
    time_bin_ms : float
        Time bin size for rate computation
    q_values : np.ndarray, optional
        Array of q values for multifractal analysis
    scales : np.ndarray, optional
        Array of scales to analyze
    order : int
        Polynomial order for detrending
    overlap : bool
        Whether to use overlapping segments
    analysis_types : List[str]
        Types of analysis to perform: 'rate', 'isi', 'count'
    comm : mpi4py communicator object, optional
        MPI communicator for distributed operation
    root : int
        Root MPI rank
    progress_interval : int
        Interval for progress reporting
        
    Returns:
    --------
    dict
        {neuron_gid: {analysis_type: mfdfa_results}}
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    
    rank = comm.rank
    size = comm.size
    
    local_total = len(gid_spike_dict)
    total_gids = comm.allreduce(local_total, op=MPI.SUM)
    
    if rank == root:
        logger.info(f"Computing MFDFA analysis for {total_gids} neurons across {size} ranks...")
    
    mfdfa_results = {}
    
    for i, (gid, spike_times) in enumerate(gid_spike_dict.items()):
        mfdfa_results[gid] = {}
        
        for analysis_type in analysis_types:
            try:
                mfdfa_results[gid][analysis_type] = compute_spike_mfdfa_analysis(
                    spike_times, duration_ms, time_bin_ms, q_values, scales, 
                    order, overlap, analysis_type
                )
                
            except Exception as e:
                warnings.warn(f"MFDFA analysis failed for neuron {gid}, type {analysis_type}: {e}")
                # Create empty result
                mfdfa_results[gid][analysis_type] = {
                    'analysis_type': analysis_type,
                    'valid_analysis': False,
                    'error': str(e),
                    'n_spikes': len(spike_times),
                    'scales': np.array([]),
                    'q_values': np.array([]),
                    'fluctuation_function': np.array([]),
                    'scaling_exponents': np.array([]),
                    'multifractal_spectrum': {'alpha': np.array([]), 'f_alpha': np.array([])},
                    'hurst_exponent': 0.0,
                    'multifractality_strength': 0.0
                }
        
        if rank == root and (i + 1) % progress_interval == 0:
            estimated_global = (i + 1) * size
            progress_pct = min(100, (estimated_global / total_gids) * 100)
            logger.info(f"MFDFA Progress: ~{progress_pct:.1f}% (rank 0: {i + 1} / {local_total})")
    
    return mfdfa_results


def compute_population_mfdfa_summary(
    mfdfa_results: Dict[int, Dict[str, any]],
    analysis_types: List[str] = ['rate']
) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics of MFDFA analysis across a population.
    
    Parameters:
    -----------
    mfdfa_results : dict
        Results from compute_population_mfdfa_analysis
    analysis_types : List[str]
        Analysis types to summarize
        
    Returns:
    --------
    dict
        Summary statistics for each analysis type
    """
    summary = {}
    
    for analysis_type in analysis_types:
        # Collect valid results
        hurst_values = []
        multifractality_values = []
        n_valid = 0
        n_total = 0
        
        for gid, gid_results in mfdfa_results.items():
            if analysis_type in gid_results:
                n_total += 1
                result = gid_results[analysis_type]
                
                if result.get('valid_analysis', False):
                    n_valid += 1
                    hurst_values.append(result['hurst_exponent'])
                    multifractality_values.append(result['multifractality_strength'])
        
        # Compute statistics
        if len(hurst_values) > 0:
            summary[analysis_type] = {
                'n_total': n_total,
                'n_valid': n_valid,
                'fraction_valid': n_valid / n_total if n_total > 0 else 0.0,
                'hurst_mean': np.mean(hurst_values),
                'hurst_std': np.std(hurst_values),
                'hurst_median': np.median(hurst_values),
                'multifractality_mean': np.mean(multifractality_values),
                'multifractality_std': np.std(multifractality_values),
                'multifractality_median': np.median(multifractality_values),
                'fraction_multifractal': np.sum(np.array(multifractality_values) > 0.1) / len(multifractality_values)
            }
        else:
            summary[analysis_type] = {
                'n_total': n_total,
                'n_valid': 0,
                'fraction_valid': 0.0,
                'hurst_mean': 0.0,
                'hurst_std': 0.0,
                'hurst_median': 0.0,
                'multifractality_mean': 0.0,
                'multifractality_std': 0.0,
                'multifractality_median': 0.0,
                'fraction_multifractal': 0.0
            }
    
    return summary


def add_mfdfa_to_population_response(
    processed_responses: Dict[str, Dict],
    time_bin_ms: float = 50.0,
    q_values: np.ndarray = None,
    scales: np.ndarray = None,
    order: int = 1,
    overlap: bool = False,
    analysis_types: List[str] = ['rate'],
    comm=None,
    root: int = 0
) -> Dict[str, Dict]:
    """
    Add MFDFA analysis to existing processed responses.
    
    Parameters:
    -----------
    processed_responses : dict
        Existing processed responses from process_model_spatiotemporal_responses
    time_bin_ms : float
        Time bin size for MFDFA analysis
    q_values : np.ndarray, optional
        Array of q values for multifractal analysis
    scales : np.ndarray, optional
        Array of scales to analyze
    order : int
        Polynomial order for detrending
    overlap : bool
        Whether to use overlapping segments
    analysis_types : List[str]
        Types of analysis to perform
    comm : mpi4py communicator object, optional
        MPI communicator
    root : int
        Root MPI rank
        
    Returns:
    --------
    dict
        Updated processed responses with MFDFA analysis
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    
    rank = comm.rank
    
    if processed_responses is None:
        return None
    
    for pop_name, pop_data in processed_responses.items():
        if rank == root:
            logger.info(f"Computing MFDFA analysis for population: {pop_name}")
        
        # Extract spike data
        gid_spike_dict = {}
        for gid, cell_metrics in pop_data['cell_metrics'].items():
            if 'spike_times' in cell_metrics:
                gid_spike_dict[gid] = cell_metrics['spike_times']
        
        # Get simulation duration
        duration_ms = None
        if 'input_metadata' in pop_data and 'duration' in pop_data['input_metadata']:
            duration_ms = pop_data['input_metadata']['duration']
        else:
            # Estimate from spike data
            all_spikes = pop_data['population_metrics'].get('all_spikes', [])
            if len(all_spikes) > 0:
                duration_ms = np.max(all_spikes)
            else:
                duration_ms = 1000.0  # Default 1 second
        
        # Compute MFDFA analysis
        mfdfa_results = compute_population_mfdfa_analysis(
            gid_spike_dict, duration_ms, time_bin_ms, q_values, scales,
            order, overlap, analysis_types, comm, root
        )
        
        # Add MFDFA results to cell metrics
        for gid, mfdfa_data in mfdfa_results.items():
            if gid in pop_data['cell_metrics']:
                pop_data['cell_metrics'][gid]['mfdfa_analysis'] = mfdfa_data
        
        # Compute population-level MFDFA summary
        mfdfa_summary = compute_population_mfdfa_summary(mfdfa_results, analysis_types)
        pop_data['population_metrics']['mfdfa_summary'] = mfdfa_summary
        
        if rank == root:
            n_valid_total = sum(summary['n_valid'] for summary in mfdfa_summary.values())
            logger.info(f"MFDFA analysis complete for {pop_name}: {n_valid_total} valid analyses")
    
    return processed_responses

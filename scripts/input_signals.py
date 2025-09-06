#!/usr/bin/env python3
"""
Support for reading and writing input signals to and from HDF5
files.  This provides helper functions to read and write signals generating by 
the DRC script.
"""

import h5py
import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any


def list_available_signals(h5_file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    List all available signals in an HDF5 file with their metadata.
    
    Args:
        h5_file_path: Path to the HDF5 file
        
    Returns:
        Dictionary mapping signal_ids to their metadata
    """
    signals_info = {}
    
    try:
        with h5py.File(h5_file_path, "r") as f:
            if "Signals" not in f:
                logging.warning(f"No 'Signals' group found in {h5_file_path}")
                return signals_info
            
            signals_group = f["Signals"]
            
            for signal_id in signals_group.keys():
                signal_group = signals_group[signal_id]
                
                info = {
                    'signal_id': signal_id,
                    'has_stimulus': 'stimulus' in signal_group,
                    'stimulus_shape': None,
                    'stimulus_dtype': None,
                    'attributes': dict(signal_group.attrs)
                }
                
                if info['has_stimulus']:
                    stimulus_dataset = signal_group['stimulus']
                    info['stimulus_shape'] = stimulus_dataset.shape
                    info['stimulus_dtype'] = stimulus_dataset.dtype
                
                # Check for dimension information
                if 'dimensions' in signal_group:
                    info['has_dimensions'] = True
                    info['n_dimensions'] = len(signal_group['dimensions'])
                else:
                    info['has_dimensions'] = False
                    info['n_dimensions'] = None
                
                signals_info[signal_id] = info
                
    except Exception as e:
        logging.error(f"Error reading signals from {h5_file_path}: {e}")
        
    return signals_info


def write_signal(h5_file_path: str,
                 name: str,
                 dimensions: List[Dict[str, Any]],
                 signal_id: str,
                 stimulus: np.ndarray):

    assert validate_signal(stimulus)

    _, n_features = stimulus.shape
    
    with h5py.File(output_path, "a") as f:
            # Create a group for signals if it doesn't exist
            if "Signals" not in f:
                signals_group = f.create_group("Signals")
            else:
                signals_group = f["Signals"]

            # Create a group for the specific signal
            if signal_id in signals_group:
                del signals_group[signal_id]
            signal_group = signals_group.create_group(signal_id)

            # Store the signal data
            signal_group.create_dataset("stimulus", data=stimulus, compression="gzip")

            # Store simple attributes directly
            signal_group.attrs['name'] = name
            signal_group.attrs['n_features'] = n_features

            # Store dimensions as a dataset
            dims_data = []
            for dim in self.dimensions:
                dims_data.append({
                    'name': dim['name'], 
                    'range_min': dim['range'][0],
                    'range_max': dim['range'][1],
                    'scale': dim.get('scale', 'linear'),
                    'priority': dim.get('priority', 1.0)
                })

            # Create a structured dtype for dimensions
            dim_dtype = np.dtype([
                ('name', 'S64'),
                ('range_min', float),
                ('range_max', float),
                ('scale', 'S16'),
                ('priority', float)
            ])

            # Create a structured array for dimensions
            dim_array = np.zeros(len(dims_data), dtype=dim_dtype)
            for i, dim in enumerate(dims_data):
                dim_array[i] = (
                    dim['name'].encode('utf-8') if isinstance(dim['name'], str) else dim['name'],
                    dim['range_min'],
                    dim['range_max'],
                    dim['scale'].encode('utf-8') if isinstance(dim['scale'], str) else dim['scale'],
                    dim['priority']
                )

            signal_group.create_dataset('dimensions', data=dim_array)

            # Store dimension_stats as a group with datasets
            dim_stats_group = signal_group.create_group('dimension_stats')
            for dim_name, stats in self.dimension_stats.items():
                dim_group = dim_stats_group.create_group(dim_name)
                for stat_name, stat_value in stats.items():
                    if stat_name == 'values' and stat_value:
                        # Save values as a dataset
                        dim_group.create_dataset('values', data=np.array(stat_value))
                    elif stat_value is not None:
                        # Save other stats as attributes
                        dim_group.attrs[stat_name] = stat_value

    

def read_signal(h5_file_path: str, 
                signal_id: str,
                sample_rate: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Read a signal from an HDF5 file created by the DRC system.
    
    Args:
        h5_file_path: Path to the HDF5 file
        signal_id: Identifier of the signal to read
        sample_rate: Expected sample rate (Hz). If provided, will generate time vector
        
    Returns:
        Tuple of (time_vector, stimulus_array, metadata_dict)
        time_vector will be None if sample_rate is not provided
    """
    try:
        with h5py.File(h5_file_path, "r") as f:
            if "Signals" not in f:
                raise ValueError(f"No 'Signals' group found in {h5_file_path}")
            
            signals_group = f["Signals"]
            
            if signal_id not in signals_group:
                available_signals = list(signals_group.keys())
                raise ValueError(f"Signal '{signal_id}' not found. Available signals: {available_signals}")
            
            signal_group = signals_group[signal_id]
            
            if "stimulus" not in signal_group:
                raise ValueError(f"No 'stimulus' dataset found for signal '{signal_id}'")
            
            # Read the stimulus data
            stimulus = signal_group["stimulus"][:]
            
            # Read metadata from attributes
            metadata = dict(signal_group.attrs)
            
            # Read dimension information if available
            if 'dimensions' in signal_group:
                dimensions_data = signal_group['dimensions'][:]
                metadata['dimensions'] = dimensions_data
            
            # Read dimension statistics if available
            if 'dimension_stats' in signal_group:
                dim_stats = {}
                dim_stats_group = signal_group['dimension_stats']
                for dim_name in dim_stats_group.keys():
                    dim_group = dim_stats_group[dim_name]
                    dim_info = dict(dim_group.attrs)
                    if 'values' in dim_group:
                        dim_info['values'] = dim_group['values'][:]
                    dim_stats[dim_name] = dim_info
                metadata['dimension_stats'] = dim_stats
            
            # Generate time vector if sample rate is provided
            time_vector = None
            if sample_rate is not None:
                n_samples = stimulus.shape[0]
                duration = n_samples / sample_rate
                time_vector = np.linspace(0, duration, n_samples, endpoint=False)
            
            logging.info(f"Successfully read signal '{signal_id}' with shape {stimulus.shape}")
            
            return time_vector, stimulus, metadata
            
    except Exception as e:
        logging.error(f"Error reading signal from {h5_file_path}: {e}")
        raise


def validate_signal(stimulus: np.ndarray, 
                    expected_duration: Optional[float] = None,
                    expected_sample_rate: Optional[float] = None,
                    min_dimensions: int = 1) -> bool:
    """
    Validate that a signal is suitable for interneuron processing.
    
    Args:
        stimulus: The stimulus array
        expected_duration: Expected duration in seconds
        expected_sample_rate: Expected sample rate in Hz
        min_dimensions: Minimum number of dimensions required
        
    Returns:
        True if signal is valid for interneuron processing
    """
    if len(stimulus.shape) < 1 or len(stimulus.shape) > 2:
        logging.warning(f"Invalid stimulus shape: {stimulus.shape}. Expected 1D or 2D array.")
        return False
    
    if len(stimulus.shape) == 2 and stimulus.shape[1] < min_dimensions:
        logging.warning(f"Signal has {stimulus.shape[1]} dimensions, but {min_dimensions} required.")
        return False
    
    if expected_duration is not None and expected_sample_rate is not None:
        expected_samples = int(expected_duration * expected_sample_rate)
        actual_samples = stimulus.shape[0]
        if abs(actual_samples - expected_samples) > expected_sample_rate * 0.1:  # 100ms tolerance
            logging.warning(f"Signal length mismatch. Expected ~{expected_samples} samples, got {actual_samples}")
            return False
    
    # Check for reasonable signal values
    if np.all(stimulus == 0):
        logging.warning("Signal is all zeros")
        return False
    
    if np.any(np.isnan(stimulus)) or np.any(np.isinf(stimulus)):
        logging.warning("Signal contains NaN or infinite values")
        return False
    
    logging.info(f"Signal validation passed. Shape: {stimulus.shape}, "
                f"Range: [{np.min(stimulus):.3f}, {np.max(stimulus):.3f}]")
    return True


# ============================================================================
# MODIFIED MAIN SECTION LOGIC
# ============================================================================

if __name__ == "__main__":
    # Configuration flags
    dry_run = False  # Set to True to skip actual spike generation
    plot = True     # Set to True to generate plots
    register_population = True  # Set to True to register with environment
    
    # NEW: Signal input configuration
    input_h5_file = None  # Set to path of HDF5 file to read signal from
    input_signal_id = None  # Set to signal ID to read (if None, will list available signals)
    use_generated_signal = True  # Set to False to force reading from file
    
    # Example usage:
    # input_h5_file = "./PYR_dynamical_response_spike_trains_10s.h5"
    # input_signal_id = "drc_features_20240514" 
    # use_generated_signal = False
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    logging.basicConfig(level=logging.INFO)
    
    # Population configurations (unchanged)
    population_config = {
        'PV_interneurons': {
            'interneuron_type': 'PV',
            'n_features': 100,
            'fraction_active_stats': {'mean': 0.35, 'std': 0.12}
        },
        'SST_interneurons': {
            'interneuron_type': 'SST', 
            'n_features': 60,
            'fraction_active_stats': {'mean': 0.18, 'std': 0.06}
        },
        'VIP_interneurons': {
            'interneuron_type': 'VIP',
            'n_features': 40,
            'fraction_active_stats': {'mean': 0.25, 'std': 0.09}
        }
    }
    
    # Environment setup (unchanged)
    output_prefix = "."
    config = {}
    params = dict(locals())
    params["config_prefix"] = "./config"
    
    try:
        env = Env(**params)
    except:
        if rank == 0:
            logging.warning("Could not initialize Env, proceeding without environment registration")
        env = None
    
    # Default signal parameters
    sample_dt_ms = 1.0
    sample_rate = 1000.0 / sample_dt_ms  # Sample rate [Hz]
    duration = 10.0  # Overall signal duration [s]
    n_dimensions = 8  # Number of input signal dimensions
    
    # ========================================================================
    # NEW: Signal Reading/Generation Logic
    # ========================================================================
    
    stimulus = None
    t = None
    signal_metadata = {}
    
    # Check if we should read from file
    if not use_generated_signal and input_h5_file is not None:
        if rank == 0:
            logging.info(f"Attempting to read signal from {input_h5_file}")
            
            # List available signals if signal_id not specified
            if input_signal_id is None:
                available_signals = list_available_signals(input_h5_file)
                if available_signals:
                    logging.info("Available signals:")
                    for sig_id, info in available_signals.items():
                        logging.info(f"  - {sig_id}: shape={info['stimulus_shape']}, "
                                   f"attrs={info['attributes']}")
                    
                    # Use the first available signal
                    input_signal_id = list(available_signals.keys())[0]
                    logging.info(f"Using signal: {input_signal_id}")
                else:
                    logging.error("No signals found in file")
                    use_generated_signal = True
            
            # Try to read the signal
            if input_signal_id is not None:
                try:
                    t, stimulus, signal_metadata = read_signal_from_h5(
                        input_h5_file, 
                        input_signal_id, 
                        sample_rate=sample_rate
                    )
                    
                    # Validate the signal
                    if not validate_signal_for_interneurons(
                        stimulus, 
                        expected_duration=duration,
                        expected_sample_rate=sample_rate,
                        min_dimensions=1
                    ):
                        logging.warning("Signal validation failed, falling back to generated signal")
                        stimulus = None
                        t = None
                    else:
                        # Update parameters based on read signal
                        if len(stimulus.shape) == 1:
                            n_dimensions = 1
                        else:
                            n_dimensions = stimulus.shape[1]
                        
                        duration = stimulus.shape[0] / sample_rate
                        
                        logging.info(f"Successfully loaded signal: duration={duration:.1f}s, "
                                   f"dimensions={n_dimensions}, sample_rate={sample_rate}Hz")
                        
                except Exception as e:
                    logging.error(f"Failed to read signal: {e}")
                    stimulus = None
                    t = None
    
    # Broadcast signal reading results to all ranks
    if comm.size > 1:
        signal_data = comm.bcast((stimulus, t, signal_metadata, duration, n_dimensions), root=0)
        stimulus, t, signal_metadata, duration, n_dimensions = signal_data
    
    # Fall back to generated signal if reading failed or not requested
    if stimulus is None:
        if rank == 0:
            if not use_generated_signal:
                logging.info("Falling back to generated signal")
            else:
                logging.info("Generating new test signal")
        
        # Create test signal (original logic)
        t, stimulus = create_test_multidimensional_signal(
            duration=duration,
            sample_rate=sample_rate, 
            n_dimensions=n_dimensions,
            signal_type="mixed"
        )
        
        signal_metadata = {
            'source': 'generated',
            'signal_type': 'mixed',
            'duration': duration,
            'sample_rate': sample_rate,
            'n_dimensions': n_dimensions
        }
    
    # ========================================================================
    # Continue with original processing logic...
    # ========================================================================
    
    # Create the interneuron modality
    interneuron_modality = InterneuronModality(
        name="interneuron",
        input_shape=(int(duration * sample_rate), n_dimensions),
        norm_type='l2',
        temporal_smoothing=0.1
    )
    
    # Process the stimulus using the modality
    processed_stimulus = interneuron_modality.preprocess_signal(stimulus)
    
    if rank == 0:
        logging.info(f"Signal source: {signal_metadata.get('source', 'unknown')}")
        logging.info(f"Created stimulus with shape {stimulus.shape}")
        logging.info(f"Processed stimulus shape: {processed_stimulus.shape}")
        logging.info(f"Processed magnitude range: {np.min(processed_stimulus):.3f} - {np.max(processed_stimulus):.3f}")
        
        if 'signal_id' in signal_metadata:
            logging.info(f"Original signal ID: {signal_metadata.get('signal_id', 'unknown')}")
    
    # Rest of the processing continues as before...
    # (Create feature space, populations, generate features, etc.)
    
    # The remainder of the main section continues unchanged from the original script

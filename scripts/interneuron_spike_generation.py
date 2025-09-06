#!/usr/bin/env python3
"""
Interneuron spike train generation script integrated with MiV simulator framework.

This script generates spike trains for interneuron populations based on 
multidimensional input signals using physiologically-grounded f-I curves.

Based on the temporal_spike_trains.py framework but adapted for interneuron
population dynamics rather than selective temporal features.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib as mpl
from typing import Tuple, List, Callable, Optional, Dict, Any
from miv_simulator.env import Env
from miv_simulator.input_features import (
    EncoderTimeConfig,
    CoordinateSystemConfig,
    InputModality,
    InputFeature,
    InputFeaturePopulation,
    FeatureSpace,
)
from miv_simulator.input_spike_trains import generate_input_spike_trains
from mpi4py import MPI
import h5py

sys.path.append('.')
from interneuron_features import (
    InterneuronModality,
    InterneuronFeaturePopulation, 
    InterneuronFeature,
    InterneuronResponseModel,
    INTERNEURON_FI_PARAMS
)
from input_signals import read_signal, validate_signal

# Define a custom reduction operation that concatenates lists
def response_concat(a, b, datatype):
    """Concatenate dictionaries of spike responses"""
    if a is None:
        return b
    if b is None:
        return a
    d = a.copy()
    for k in b:
        if k in d:
            d[k] = d[k] + b[k]
        else:
            d[k] = b[k]
    return d


# Create MPI concatenation operation
response_concat_op = MPI.Op.Create(response_concat, commute=True)


def mpi_excepthook(type, value, traceback):
    """MPI exception handler."""
    sys_excepthook(type, value, traceback)
    sys.stderr.flush()
    sys.stdout.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


def create_test_multidimensional_signal(duration: float, 
                                        sample_rate: float,
                                        n_dimensions: int = 10,
                                        signal_type: str = "oscillatory") -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a test multidimensional signal for interneuron stimulation.
    
    Args:
        duration: Signal duration in seconds
        sample_rate: Sample rate in Hz
        n_dimensions: Number of signal dimensions
        signal_type: Type of signal ('oscillatory', 'noise', 'mixed')
    
    Returns:
        Tuple of (time_vector, signal_array)
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = np.zeros((n_samples, n_dimensions))
    
    if signal_type == "oscillatory":
        # Create oscillatory signals with different frequencies in each dimension
        base_frequencies = np.logspace(np.log10(2), np.log10(50), n_dimensions)  # 2-50 Hz
        
        for dim in range(n_dimensions):
            freq = base_frequencies[dim]
            amplitude = 0.5 + 0.5 * np.random.rand()  # Random amplitude 0.5-1.0
            phase = 2 * np.pi * np.random.rand()  # Random phase
            signal[:, dim] = amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Add a global modulation envelope
        envelope_freq = 0.5  # 0.5 Hz modulation
        envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * envelope_freq * t))
        signal = signal * envelope.reshape(-1, 1)
        
    elif signal_type == "noise":
        # Colored noise with different temporal correlations
        for dim in range(n_dimensions):
            # Generate colored noise with different correlation times
            correlation_time = 0.01 + 0.1 * dim / n_dimensions  # 10-110 ms
            alpha = np.exp(-1.0 / (sample_rate * correlation_time))
            
            white_noise = np.random.randn(n_samples)
            colored_noise = np.zeros_like(white_noise)
            colored_noise[0] = white_noise[0]
            
            for i in range(1, n_samples):
                colored_noise[i] = alpha * colored_noise[i-1] + np.sqrt(1 - alpha**2) * white_noise[i]
            
            signal[:, dim] = colored_noise
    
    elif signal_type == "mixed":
        # Mix of oscillatory and noise components
        # First half dimensions: oscillatory
        for dim in range(n_dimensions // 2):
            freq = 5 + 10 * dim  # 5, 15, 25, ... Hz
            amplitude = 0.7
            signal[:, dim] = amplitude * np.sin(2 * np.pi * freq * t)
        
        # Second half dimensions: noise
        for dim in range(n_dimensions // 2, n_dimensions):
            signal[:, dim] = 0.5 * np.random.randn(n_samples)
        
        # Add global amplitude modulation
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz modulation
        signal = signal * modulation.reshape(-1, 1)
    
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")
    
    return t, signal

def run_interneuron_spike_generation(signal_id,
                                    dataset_prefix = "./datasets",
                                    output_prefix = ".",
                                    config_prefix = "./config",
                                    config = "Dynamical_Response_Features.yaml",
                                    population_name = "IN",
                                    neuron_type = None,
                                    fraction_active = { 'mean': 0.9, 'std': 0.2 },
                                    input_signal_file = None,  # Set to path of HDF5 file to read signal from
                                    register_population = True,
                                    stimulus_duration = 10,
                                    n_features = 150,
                                    sample_dt_ms=1.0,
                                    random_seed = 42,
                                    n_channels = 10,
                                    dry_run = False,
                                    plot = True,
                                    comm = None,
                                    io_kwargs = {
                                        'io_size': 1,
                                        'write_size': 1,
                                        'chunk_size': 10000,
                                        'value_chunk_size': 20000,
                                    },
                                ):
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.rank
    
    logging.basicConfig(level=logging.INFO)
    
    
    # Environment setup
    params = dict(locals())
    params["config_prefix"] = config_prefix
    params["Model Name"] = "interneuron_features"
    # np.seterr(all="raise")
    
    # Try to create environment (may need config files)
    env = Env(**params)
    try:
        env = Env(**params)
    except:
        if rank == 0:
            logging.warning("Could not initialize Env, proceeding without environment registration")
        env = None

    fi_parameters_config = INTERNEURON_FI_PARAMS
    mean_response_config = {}
    if env is not None:
        if "Stimulus" in env.model_config:
            mean_response_config = env.model_config["Stimulus"].get("Mean Response", {})
            fi_parameters_config = env.model_config["Stimulus"].get("f-I Parameters", INTERNEURON_FI_PARAMS)
            
    if population_name in mean_response_config:
        mean_response = mean_response_config[population_name]
        fraction_active = { 'mean': mean_response['mean fraction active per time bin'],
                            'std': mean_response['std fraction active per time bin'] }

    # Signal input configuration
    use_generated_signal = input_signal_file is None
        
    # Generated signal parameters
    sample_dt_ms = 1.0
    sample_rate = 1000.0 / sample_dt_ms  # Sample rate [Hz]
    n_dimensions = 8  # Number of input signal dimensions
    
    # Create the interneuron modality
    interneuron_modality = InterneuronModality(
        name="interneuron",
        input_shape=(int(stimulus_duration * sample_rate), n_dimensions),
        norm_type='l2',
        temporal_smoothing=0.1
    )
    
    # Create a feature space and register the modality
    feature_space = FeatureSpace(name="interneuron_feature_space")
    feature_space.register_modality(interneuron_modality)
    
    # Create interneuron populations
    populations = {}
    current_start_gid = 0
    
    if env is not None and register_population:
        # Register populations with environment if available
        max_pop_enum = 0
        if hasattr(env, 'Populations'):
            for _, pop_enum in env.Populations.items():
                max_pop_enum = max(pop_enum, max_pop_enum)


    # Population configurations
    population_config = {
        population_name: {
            'neuron_type': population_name if neuron_type is None else neuron_type,
            'n_features': n_features,
            'fraction_active_stats': fraction_active
        }
    }
    
    for pop_name, pop_config in population_config.items():
        neuron_type = pop_config.get('neuron_type', pop_name)
        n_features = pop_config['n_features']
        
        # Override from environment if available
        if env is not None and pop_name in env.celltypes:
            current_start_gid = env.celltypes[pop_name]["start"]
            n_features = env.celltypes[pop_name]["num"]
            
        # Create population
        population = InterneuronFeaturePopulation(
            name=pop_name,
            feature_space=feature_space,
            n_features=n_features,
            modality=interneuron_modality,
            neuron_type=neuron_type,
            fraction_active_stats=pop_config.get('fraction_active_stats')
        )
        
        # Register with environment if available
        if env is not None and register_population:
            pop_id = max_pop_enum + 1
            env.Populations[pop_name] = pop_id
            max_pop_enum = pop_id
            
            # Add to cell distribution
            cell_distribution = {}
            if hasattr(env, 'geometry') and "Cell Distribution" in env.geometry:
                cell_distribution = env.geometry["Cell Distribution"]
            else:
                if hasattr(env, 'geometry'):
                    env.geometry["Cell Distribution"] = cell_distribution
            cell_distribution[pop_name] = {"All": n_features}
        
        populations[pop_name] = population
        current_start_gid += n_features
    
    # Generate features for each population
    all_features = {}
    for pop_name, population in populations.items():
        start_gid = 0
        if env is not None and pop_name in env.celltypes:
            start_gid = env.celltypes[pop_name]["start"]
        
        features = population.generate_features(
            start_gid=start_gid, 
            rank=comm.rank, 
            size=comm.size
        )
        all_features[pop_name] = features
        
        if rank == 0:
            logging.info(f"Generated {len(features)} features for {pop_name} on rank {rank}")

    stimulus = None
    t = None
    signal_metadata = {}

    # Check if we should read from file
    if rank == 0:
        if not use_generated_signal and input_signal_file is not None:
            logging.info(f"Reading signal from {input_signal_file}...")

            # List available signals if signal_id not specified
            if input_signal_id is None:
                available_signals = list_available_signals(input_signal_file)
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
                    t, stimulus, signal_metadata = read_signal(
                        input_signal_file, 
                        input_signal_id, 
                        sample_rate=sample_rate
                    )

                    # Validate the signal
                    if not validate_signal(
                        stimulus, 
                        expected_duration=stimulus_duration,
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

                        stimulus_duration = stimulus.shape[0] / sample_rate

                        logging.info(f"Successfully loaded signal: duration={stimulus_duration:.1f}s, "
                                     f"dimensions={n_dimensions}, sample_rate={sample_rate}Hz")

                except Exception as e:
                    logging.error(f"Failed to read signal: {e}")
                    stimulus = None
                    t = None
    comm.barrier()
    
    # Broadcast signal reading results to all ranks
    if comm.size > 1:
        signal_data = comm.bcast((stimulus, t, signal_metadata, stimulus_duration, n_dimensions), root=0)
        stimulus, t, signal_metadata, stimulus_duration, n_dimensions = signal_data
    
    # Fall back to generated signal if reading failed or not requested
    if stimulus is None:
        if rank == 0:
            logging.info("Generating new test signal")
        
            # Create test signal
            t, stimulus = create_test_multidimensional_signal(
                duration=stimulus_duration,
                sample_rate=sample_rate, 
                n_dimensions=n_dimensions,
                signal_type="mixed"
            )
        
            signal_metadata = {
                'source': 'generated',
                'signal_type': 'mixed',
                'duration': stimulus_duration,
                'sample_rate': sample_rate,
                'n_dimensions': n_dimensions
            }
        signal_data = comm.bcast((stimulus, t, signal_metadata, stimulus_duration, n_dimensions), root=0)
        stimulus, t, signal_metadata, stimulus_duration, n_dimensions = signal_data
        
    
    # Process the stimulus using the modality
    processed_stimulus = interneuron_modality.preprocess_signal(stimulus)
    
    if rank == 0:
        logging.info(f"Created stimulus with shape {stimulus.shape}")
        logging.info(f"Processed stimulus shape: {processed_stimulus.shape}")
        logging.info(f"Processed magnitude range: {np.min(processed_stimulus):.3f} - {np.max(processed_stimulus):.3f}")
    
    # Initialize encoders and get responses for plotting
    dt_ms = 1.0  # Encoder timestep [ms]
    sample_duration_ms = dt_ms  # Duration of one sample [ms]
    time_config = EncoderTimeConfig(duration_ms=sample_duration_ms, dt_ms=dt_ms)
    
    # Collect responses for plotting (if enabled)
    local_responses = {}
    local_population_stats = {}
    
    if plot and rank == 0:
        for pop_name, population in populations.items():
            # Initialize encoders
            for feature in population.features.values():
                feature.initialize_encoder(time_config)
            
            # Get responses from a subset of neurons for plotting
            pop_responses = []
            for i, (gid, feature) in enumerate(population.features.items()):
                if i >= 5:  # Limit to first 5 neurons per population for plotting
                    break
                response = feature.get_response(processed_stimulus)
                pop_responses.append(response)
            
            local_responses[pop_name] = pop_responses
            
            # Get population statistics
            avg_magnitude = np.mean(processed_stimulus)
            pop_stats = population.generate_population_response(avg_magnitude)
            local_population_stats[pop_name] = pop_stats

    comm.barrier()

    # Gather responses across all ranks for plotting
    if plot:
        all_responses = comm.reduce(local_responses, op=response_concat_op, root=0)
        all_pop_stats = comm.reduce(local_population_stats, op=response_concat_op, root=0)
    else:
        all_responses = None
        all_pop_stats = None
    
    # Generate plots on rank 0
    if plot and (rank == 0):
        # Set up plotting parameters
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
        
        # Create comprehensive plot
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), 
                               gridspec_kw={"height_ratios": [2, 2, 2, 2, 3]})
        
        # Original multidimensional signal (first few dimensions)
        n_dims_to_plot = min(4, n_dimensions)
        for dim in range(n_dims_to_plot):
            axes[0].plot(t, stimulus[:, dim], label=f'Dim {dim+1}', alpha=0.7)
        axes[0].set_title('Original Multidimensional Signal (First 4 Dimensions)')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Signal magnitude across all dimensions
        signal_magnitude = np.linalg.norm(stimulus, axis=1)
        axes[1].plot(t, signal_magnitude, 'k-', linewidth=2, label='L2 Magnitude')
        axes[1].set_title('Signal Magnitude Across All Dimensions')
        axes[1].set_ylabel('Magnitude')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Processed (normalized) signal magnitude
        axes[2].plot(t, processed_stimulus, 'r-', linewidth=2, label='Normalized Magnitude')
        axes[2].set_title('Processed Signal Magnitude (Input to Interneurons)')
        axes[2].set_ylabel('Normalized Magnitude')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # f-I curves for all interneuron types
        test_inputs = np.linspace(0, 1, 100)
        axes[3].set_title('f-I Curves for Different Interneuron Types')
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, (neuron_type, params) in enumerate(fi_parameters_config.items()):
            response_model = InterneuronResponseModel(neuron_type)
            firing_rates = [response_model.compute_firing_rate(inp) for inp in test_inputs]
            
            color = colors[i % len(colors)]
            axes[3].plot(test_inputs, firing_rates, label=f'{neuron_type}', 
                        linewidth=2, color=color)
            
            # Add threshold line
            threshold = params['threshold']
            axes[3].axvline(x=threshold, color=color, linestyle='--', alpha=0.5)
        
        axes[3].set_xlabel('Normalized Input Magnitude')
        axes[3].set_ylabel('Firing Rate (Hz)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # Example spike rasters
        axes[4].set_title('Example Spike Responses from Each Population')
        y_offset = 0
        
        for i, (pop_name, pop_responses) in enumerate(all_responses.items()):
            color = colors[i % len(colors)]
            neuron_type = populations[pop_name].neuron_type
            
            for j, response in enumerate(pop_responses[:5]):  # Show first 5 neurons
                # Extract spike times
                spike_times = response[:, 0] / 1000.  # First sample, first neuron, convert ms to s.
                    
                # Plot as vertical lines
                for spike_time in spike_times:
                    axes[4].plot([spike_time, spike_time], 
                                 [y_offset - 0.4, y_offset + 0.4], 
                                 color=color, linewidth=1.5)
                
                y_offset += 1
            
            # Add population label
            mid_y = y_offset - len(pop_responses[:3]) / 2
            axes[4].text(-0.5, mid_y, f'{neuron_type}', 
                        rotation=90, verticalalignment='center',
                        fontweight='bold', color=color)
        
        axes[4].set_xlabel('Time (s)')
        axes[4].set_ylabel('Neuron')
        axes[4].set_xlim(0, stimulus_duration)
        axes[4].set_ylim(-1, y_offset)
        
        plt.tight_layout()
        plt.savefig("interneuron_population_responses.svg", dpi=600, bbox_inches='tight')
        plt.show()
        
        # Print population statistics
        print("\n" + "="*60)
        print("Interneuron population summary")
        print("="*60)
        
        for pop_name, pop_stats in all_pop_stats.items():
            population = populations[pop_name]
            print(f"\n{pop_name.upper()}:")
            print(f"  Type: {population.neuron_type}")
            print(f"  Total neurons: {population.n_features}")
            print(f"  Expected fraction active: {population.fraction_active_stats}")
            print(f"  Response to mean signal: {pop_stats}")
            
            # Show f-I parameters
            fi_params = fi_parameters_config[population.neuron_type]
            print(f"  f-I parameters: threshold={fi_params['threshold']:.2f}, "
                  f"slope={fi_params['slope']:.1f}, max_rate={fi_params['max_rate']:.0f} Hz")
    
    comm.barrier()
    
    # Generate actual spike trains (if not dry run)
    if not dry_run:
        for pop_name, population in populations.items():
            output_path = os.path.join(
                output_prefix, 
                f"{pop_name}_{signal_id}_interneuron_spikes.h5"
            )
            
            if rank == 0:
                logging.info(f"Generating spike trains for {pop_name}...")
            
            generate_input_spike_trains(
                env,
                population,
                signal=stimulus,
                signal_id=signal_id,
                coords_path=None,
                output_path=output_path,
                output_spikes_namespace="Interneuron Spikes",
                output_spike_train_attr_name="Spike Train",
                io_size=4,
                write_size=50000,
                chunk_size=10000,
                value_chunk_size=200000,
            )
            print(f"\nSpike trains saved to: {output_path}")
            
            # Save signal and metadata
            if rank == 0:
                logging.info(f"Saving signal metadata to {output_path}")
                with h5py.File(output_path, "a") as f:
                    # Create signals group
                    if "Signals" not in f:
                        signals_group = f.create_group("Signals")
                    else:
                        signals_group = f["Signals"]
                    
                    # Create signal group
                    if signal_id in signals_group:
                        del signals_group[signal_id]
                    signal_group = signals_group.create_group(signal_id)
                    
                    # Save signal data
                    signal_group.create_dataset("data", data=stimulus, compression="gzip")
                    signal_group.create_dataset("processed_data", data=processed_stimulus, compression="gzip")
                    signal_group.create_dataset("time", data=t, compression="gzip")
                    
                    # Save metadata
                    signal_group.attrs["duration"] = stimulus_duration
                    signal_group.attrs["sample_rate"] = sample_rate
                    signal_group.attrs["sample_dt_ms"] = sample_dt_ms
                    signal_group.attrs["n_dimensions"] = n_dimensions
                    signal_group.attrs["norm_type"] = interneuron_modality.norm_type
                    signal_group.attrs["temporal_smoothing"] = interneuron_modality.temporal_smoothing
                    signal_group.attrs["description"] = (
                        f"Multidimensional signal for interneuron population stimulation. "
                        f"Signal type: mixed oscillatory and noise. "
                        f"Dimensions: {n_dimensions}. Duration: {stimulus_duration}s."
                    )
                    
                    # Save population-specific metadata
                    pop_group = signal_group.create_group(f"population_{pop_name}")
                    pop_group.attrs["neuron_type"] = population.neuron_type
                    pop_group.attrs["n_features"] = population.n_features
                    
                    # Save f-I parameters
                    fi_params = fi_parameters_config[population.neuron_type]
                    for param_name, param_value in fi_params.items():
                        if isinstance(param_value, (int, float)):
                            pop_group.attrs[param_name] = param_value
                        else:
                            pop_group.attrs[param_name] = param_value
            
            comm.barrier()
    
    if rank == 0:
        print(f"Processed {len(populations)} populations:")
        for pop_name, pop in populations.items():
            print(f"  - {pop_name}: {pop.n_features} {pop.neuron_type} neurons")
        
        print(f"Signal duration: {stimulus_duration} s")
        print(f"Signal dimensions: {n_dimensions}")
        print(f"Total time points: {len(processed_stimulus)}")

        
if __name__ == "__main__":
    run_interneuron_spike_generation(signal_id = "test_interneuron_features_20250905",
                                     neuron_type = "PV",
                                     population_name = "PVBC",
                                     config = "Network_Clamp_PYR_gid_48041.yaml",
                                     dataset_prefix = "datasets",
                                     stimulus_duration = 10,
                                     output_prefix = "datasets",
                                     register_population = False,
                                     dry_run = False)
                                     

    
    # run_interneuron_spike_generation(signal_id = "drc_features_20250905",
    #                                  stimulus_duration = 10,
    #                                  population_name = "PYR",
    #                                  register_population = False,
    #                                  config = "Full_Scale_Dynamic_Response_Features.yaml",
    #                                  output_path = "PYR_dynamical_response_spike_trains_10s.h5",
    #                                  dataset_prefix = "/scratch1/03320/iraikov/striped2/MiV",
    #                                  output_prefix = "/scratch1/03320/iraikov/striped2/MiV/results/livn",
    #                                  plot=False,
    #                                  io_kwargs={'io_size': 4,
    #                                             'write_size': 50000,
    #                                             'chunk_size': 10000,
    #                                             'value_chunk_size': 100000,
    #                                             }
    #                                  )

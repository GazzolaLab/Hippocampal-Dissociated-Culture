import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable, Optional, Dict, Any
from dataclasses import dataclass
from scipy.signal import lfiltic, lfilter
from miv_simulator.input_features import (
    InputModality, 
    InputFeature, 
    InputFeaturePopulation, 
    FeatureEncoding,
    CoordinateSystemConfig,
    FeatureSpace
)
from spike_encoder import EncoderTimeConfig, PoissonSpikeGenerator, EncodingPipeline


def ewma_linear_filter(array, alpha):
    b = [alpha]
    a = [1, alpha-1]
    zi = lfiltic(b, a, array[0:1], [0])
    return lfilter(b, a, array, zi=zi)[0]


# Physiologically-grounded f-I parameters for interneuron types
INTERNEURON_FI_PARAMS = {
    'PV': {
        'threshold': 0.2,        # Normalized input threshold (0-1)
        'slope': 80.0,           # Firing rate gain (Hz per normalized input unit)
        'max_rate': 150.0,       # Saturation firing rate (Hz)
        'circuit_type': 'feedforward',  # Primary circuit role
        'description': 'Parvalbumin-positive fast-spiking interneurons'
    },
    'SST': {
        'threshold': 0.15,       # Lower threshold
        'slope': 60.0,           # Gentler slope  
        'max_rate': 80.0,        # Lower max rate
        'circuit_type': 'feedback',
        'description': 'Somatostatin-positive interneurons'
    },
    'VIP': {
        'threshold': 0.18,
        'slope': 30.0,
        'max_rate': 100.0,
        'circuit_type': 'feedforward',
        'description': 'Vasoactive intestinal peptide interneurons'
    },
    'CCK': {
        'threshold': 0.12,
        'slope': 30.0,
        'max_rate': 90.0,
        'circuit_type': 'feedback',
        'description': 'Cholecystokinin-positive interneurons'
    },
    'NPY': {
        'threshold': 0.15,
        'slope': 30.0,
        'max_rate': 40.0,
        'circuit_type': 'feedforward',
        'description': 'Neuropeptide Y expressing interneurons'
    }
}


def apply_circuit_dynamics(activity_magnitude: float, 
                           circuit_type: str, 
                           previous_activity: float = 0.0,
                           temporal_weight: float = 0.7) -> float:
    """
    Apply circuit-specific dynamics to input signal.
    
    Args:
        activity_magnitude: Current normalized activity magnitude (0-1)
        circuit_type: 'feedforward' or 'feedback'
        previous_activity: Previous processed signal value
        temporal_weight: Weight for current vs previous activity (feedback only)
    
    Returns:
        Processed signal magnitude
    """
    if circuit_type == 'feedforward':
        # Feedforward: responds directly to input changes
        return activity_magnitude
        
    elif circuit_type == 'feedback':
        # Feedback: integrates current input with recent network activity
        return (temporal_weight * activity_magnitude + 
                (1 - temporal_weight) * previous_activity)
    
    else:
        raise ValueError(f"Unknown circuit_type: {circuit_type}")


def fi_curve_response(input_magnitude: float, 
                     threshold: float, 
                     slope: float, 
                     max_rate: float) -> float:
    """
    Approximation of f-I curve: linear above threshold with saturation.
    
    Args:
        input_magnitude: Processed input magnitude (0-1)
        threshold: Input threshold for spiking (0-1)
        slope: Slope of f-I curve (Hz per unit input)
        max_rate: Maximum firing rate (Hz)
    
    Returns:
        Firing rate (Hz)
    """
    if input_magnitude < threshold:
        return 0.0
    
    # Linear region above threshold
    linear_response = slope * (input_magnitude - threshold)
    
    # Apply saturation
    return min(linear_response, max_rate)


class InterneuronModality(InputModality):
    """
    Modality for interneuron population responses to multidimensional signals.
    
    Converts multidimensional input signals to normalized magnitude and 
    generates population-level interneuron responses based on f-I curves
    and circuit dynamics.
    """
    
    def __init__(self,
                 name: str = "interneuron",
                 input_shape: Tuple[int, ...] = (1000, 10),  # (time_samples, dimensions)
                 norm_type: str = 'l2',
                 temporal_smoothing: float = 0.1):
        """
        Initialize interneuron modality.
        
        Args:
            name: Modality name
            input_shape: Shape of input signal (time_samples, n_dimensions)
            norm_type: Type of norm to compute ('l1', 'l2', 'max')
            temporal_smoothing: Temporal smoothing factor for magnitude
        """
        
        # 1D feature coordinate system: just interneuron type
        feature_coordinate_system = CoordinateSystemConfig(
            dimensions=1,
            bounds=[(0, len(INTERNEURON_FI_PARAMS) - 1)],  # Interneuron type index
            units=["neuron_type"],
        )
        
        super().__init__(
            name,
            "interneuron",
            feature_coordinate_system,
            input_shape=input_shape
        )
        
        self.norm_type = norm_type
        self.temporal_smoothing = temporal_smoothing
        
        # Create type index mapping
        self.type_names = list(INTERNEURON_FI_PARAMS.keys())
        self.type_to_index = {name: i for i, name in enumerate(self.type_names)}
        self.index_to_type = {i: name for i, name in enumerate(self.type_names)}
    
    def preprocess_signal(self, stimulus: np.ndarray) -> np.ndarray:
        """
        Preprocess multidimensional signal to normalized magnitude.
        
        Args:
            stimulus: Input signal of shape (time_samples, n_dimensions) or (time_samples,)
        
        Returns:
            Normalized magnitude signal of shape (time_samples,)
        """
        # Ensure input is 2D
        if len(stimulus.shape) == 1:
            # Single dimension - reshape to (time, 1)
            signal = stimulus.reshape(-1, 1)
        elif len(stimulus.shape) == 2:
            signal = stimulus.copy()
        else:
            raise ValueError(f"Expected 1D or 2D signal, got shape {stimulus.shape}")
        
        time_samples, n_dimensions = signal.shape

        # Compute magnitude using specified norm
        if self.norm_type == 'l1':
            magnitude = np.sum(np.abs(signal), axis=1)
        elif self.norm_type == 'l2':
            magnitude = np.sqrt(np.sum(signal**2, axis=1))
        elif self.norm_type == 'max':
            magnitude = np.max(np.abs(signal), axis=1)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")
        
        # Normalize to [0, 1] range
        if np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        
        # Apply temporal smoothing if specified
        if self.temporal_smoothing > 0:
            magnitude = self._apply_temporal_smoothing(magnitude)
        
        return magnitude
    
    def _apply_temporal_smoothing(self, magnitude: np.ndarray) -> np.ndarray:
        """Apply exponential temporal smoothing to magnitude signal."""

        alpha = self.temporal_smoothing
        smoothed = ewma_linear_filter(magnitude, alpha)

        return smoothed
    
    def to_feature_coordinates(self, modality_coordinates: np.ndarray) -> np.ndarray:
        """Convert interneuron type to feature space coordinates."""
        if len(modality_coordinates) != 1:
            raise ValueError(f"Expected 1 coordinate, got {len(modality_coordinates)}")
        return modality_coordinates
    
    def from_feature_coordinates(self, feature_coordinates: np.ndarray) -> np.ndarray:
        """Convert feature space coordinates to modality-specific coordinates."""
        if len(feature_coordinates) != 1:
            raise ValueError(f"Expected 1D feature coordinates, got {len(feature_coordinates)}")
        return feature_coordinates
    
    def get_neuron_type_name(self, type_index: int) -> str:
        """Get neuron type name from index."""
        return self.index_to_type.get(type_index, 'Unknown')


class InterneuronResponseModel:
    """
    Models the response of a single interneuron based on f-I curve and circuit dynamics.
    """
    
    def __init__(self, neuron_type: str):
        """
        Initialize response model for specific interneuron type.
        
        Args:
            neuron_type: Type of interneuron ('PV', 'SST', 'VIP', 'CCK')
        """
        if neuron_type not in INTERNEURON_FI_PARAMS:
            raise ValueError(f"Unknown interneuron type: {neuron_type}")
        
        self.neuron_type = neuron_type
        self.params = INTERNEURON_FI_PARAMS[neuron_type]
        self.previous_activity = 0.0
    
    def compute_firing_rate(self, activity_magnitude: float) -> float:
        """
        Convert normalized signal magnitude to firing rate.
        
        Args:
            activity_magnitude: Normalized activity magnitude (0-1)
        
        Returns:
            Firing rate (Hz)
        """
        # Apply circuit-specific dynamics
        processed_input = apply_circuit_dynamics(
            activity_magnitude, 
            self.params['circuit_type'], 
            self.previous_activity
        )
        
        # Apply f-I curve
        firing_rate = fi_curve_response(
            processed_input,
            self.params['threshold'],
            self.params['slope'], 
            self.params['max_rate']
        )
        
        # Update history for feedback circuits
        self.previous_activity = processed_input
        
        return firing_rate
    
    def reset_history(self):
        """Reset the activity history (useful between trials)."""
        self.previous_activity = 0.0


class InterneuronEncoder:
    """
    Encoder that converts signal magnitude to spike trains using interneuron f-I curves.
    """
    
    def __init__(self,
                 time_config: EncoderTimeConfig,
                 neuron_type: str,
                 local_random: Optional[np.random.RandomState] = None):
        """
        Initialize interneuron encoder.
        
        Args:
            time_config: Encoder time configuration
            neuron_type: Type of interneuron
            local_random: Random state for spike generation
        """
        self.time_config = time_config
        self.neuron_type = neuron_type
        self.response_model = InterneuronResponseModel(neuron_type)
        
        if local_random is None:
            local_random = np.random.RandomState()
        
        # Create Poisson spike generator
        self.spike_generator = PoissonSpikeGenerator(
            time_config=time_config,
            random_seed=local_random
        )
    
    def encode(self, activity_magnitude: np.ndarray) -> np.ndarray:
        """
        Encode activity magnitude as spike trains.
        
        Args:
            activity_magnitude: Array of normalized magnitudes over time
        
        Returns:
            Spike train array
        """
        # Convert each magnitude value to firing rate
        firing_rates = np.array([
            self.response_model.compute_firing_rate(mag) 
            for mag in activity_magnitude
        ]).reshape((-1,1))
        
        # Generate spikes using Poisson process
        spikes = np.asarray(self.spike_generator.encode(firing_rates, return_times=True)[0][0])
        
        return spikes
    
    def get_response(self, activity_magnitude: np.ndarray) -> np.ndarray:
        """Get response (for compatibility with existing interface)."""
        return self.encode(activity_magnitude)


class InterneuronFeature(InputFeature):
    """
    Individual interneuron feature with f-I curve properties.
    """
    
    def __init__(self,
                 gid: int,
                 neuron_type: str,
                 position: Optional[np.ndarray] = None,
                 kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize interneuron feature.
        
        Args:
            gid: Global ID
            neuron_type: Type of interneuron
            position: Position in feature space (type index)
            kwargs: Additional parameters
        """
        self.neuron_type = neuron_type
        
        # Create position if not provided
        if position is None:
            type_index = list(INTERNEURON_FI_PARAMS.keys()).index(neuron_type)
            position = np.array([type_index], dtype=np.float32)
        
        # Create encoding specification
        encoding = FeatureEncoding(
            feature_type="interneuron",
            encoder_params={
                "neuron_type": neuron_type,
                **INTERNEURON_FI_PARAMS[neuron_type]
            }
        )
        
        super().__init__(
            gid=gid,
            position=position,
            encoding=encoding,
            kwargs=kwargs or {}
        )
        
        self._encoder = None
    
    def initialize_encoder(self,
                          time_config: EncoderTimeConfig,
                          local_random: Optional[np.random.RandomState] = None):
        """Initialize the encoder for this feature."""
        if self._encoder is None:
            self._encoder = InterneuronEncoder(
                time_config=time_config,
                neuron_type=self.neuron_type,
                local_random=local_random
            )
    
    def get_response(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Get the feature's response to a stimulus."""
        if self._encoder is None:
            raise RuntimeError("Encoder not initialized. Call initialize_encoder first.")
        
        return self._encoder.get_response(signal, **kwargs)
    
    def to_attribute_dict(self) -> Dict[str, np.ndarray]:
        """Convert to attribute dictionary for storage."""
        attr_dict = {
            "Feature Type": np.array([self.encoding.feature_type], dtype='U20'),
            "Position": self.position.astype(np.float32),
            "Interneuron Type": np.array([self.neuron_type], dtype='U10'),
        }
        
        # Add f-I curve parameters
        fi_params = INTERNEURON_FI_PARAMS[self.neuron_type]
        for param_name, param_value in fi_params.items():
            if isinstance(param_value, (int, float)):
                attr_dict[param_name] = np.array([param_value], dtype=np.float32)
            else:
                attr_dict[param_name] = np.array([param_value], dtype='U20')
        
        return attr_dict


class InterneuronFeaturePopulation(InputFeaturePopulation):
    """
    Population of interneuron features with population-level statistics.
    """
    
    def __init__(self,
                 name: str,
                 feature_space: FeatureSpace,
                 n_features: int,
                 modality: InterneuronModality,
                 neuron_type: str,
                 fraction_active_stats: Optional[Dict[str, float]] = None):
        """
        Initialize interneuron population.
        
        Args:
            name: Population name
            feature_space: Feature space container
            n_features: Number of interneurons
            modality: Interneuron modality
            neuron_type: Type of interneurons in this population
            fraction_active_stats: Statistics for fraction active per time bin
        """
        
        super().__init__(
            name=name,
            feature_space=feature_space,
            n_features=n_features,
            modality=modality
        )
        
        self.neuron_type = neuron_type
        
        # Default fraction active statistics if not provided
        if fraction_active_stats is None:
            # Default values based on typical interneuron activity
            default_stats = {
                'PV':  {'mean': 0.9, 'std': 0.2},
                'SST': {'mean': 0.9, 'std': 0.1},
                'VIP': {'mean': 0.9, 'std': 0.05},
                'CCK': {'mean': 0.9, 'std': 0.2}
            }
            fraction_active_stats = default_stats.get(neuron_type, {'mean': 0.9, 'std': 0.1})
        
        self.fraction_active_stats = fraction_active_stats
    
    def generate_features(self,
                          start_gid: int = 0,
                          n_features: Optional[int] = None,
                          local_random: Optional[np.random.RandomState] = None,
                          rank: Optional[int] = None,
                          size: Optional[int] = None) -> List[InterneuronFeature]:
        """Generate interneuron features for this population."""
        
        if local_random is None:
            local_random = np.random.RandomState()
        
        if n_features is None:
            n_features = self.n_features
        else:
            n_features = min(n_features, self.n_features)
        
        # Determine which features this rank should generate
        if (rank is not None) and (size is not None):
            feature_indices = list(range(rank, n_features, size))
        else:
            feature_indices = list(range(n_features))
        
        features = []
        type_index = self.modality.type_to_index[self.neuron_type]
        
        for i in feature_indices:
            gid = start_gid + i
            
            # All interneurons in this population have the same position in feature space
            position = np.array([type_index], dtype=np.float32)
            
            feature = InterneuronFeature(
                gid=gid,
                neuron_type=self.neuron_type,
                position=position
            )
            
            features.append(feature)
            self.features[gid] = feature
        
        return features
    
    def generate_population_response(self, activity_magnitude: float) -> Dict[str, Any]:
        """
        Generate population-level response statistics for a single time bin.
        
        Args:
            activity_magnitude: Normalized input magnitude (0-1)
        
        Returns:
            Dictionary with population response statistics
        """
        # Get base firing rate from f-I curve
        response_model = InterneuronResponseModel(self.neuron_type)
        base_firing_rate = response_model.compute_firing_rate(activity_magnitude)
        
        # Determine fraction active based on input magnitude and empirical statistics
        mean_frac = self.fraction_active_stats['mean']
        std_frac = self.fraction_active_stats['std']
        
        # Add noise based on empirical statistics
        actual_fraction = np.random.normal(mean_frac, std_frac)
        actual_fraction = np.clip(actual_fraction, 0.0, 1.0)
        
        n_active = int(actual_fraction * self.n_features)
        
        return {
            'n_active': n_active,
            'firing_rate': base_firing_rate,
            'fraction_active': actual_fraction,
            'activity_magnitude': activity_magnitude,
            'neuron_type': self.neuron_type
        }


def create_interneuron_populations(feature_space: FeatureSpace,
                                   modality: InterneuronModality,
                                   population_sizes: Dict[str, int],
                                   fraction_active_stats: Optional[Dict[str, Dict[str, float]]] = None,
                                   start_gid: int = 0) -> Dict[str, InterneuronFeaturePopulation]:
    """
    Convenience function to create multiple interneuron populations.
    
    Args:
        feature_space: Feature space to add populations to
        modality: Interneuron modality
        population_sizes: Dictionary mapping interneuron types to population sizes
        fraction_active_stats: Optional statistics for each type
        start_gid: Starting GID for populations
    
    Returns:
        Dictionary mapping population names to populations
    """
    populations = {}
    current_gid = start_gid
    
    for neuron_type, n_neurons in population_sizes.items():
        if neuron_type not in INTERNEURON_FI_PARAMS:
            raise ValueError(f"Unknown interneuron type: {neuron_type}")
        
        # Get fraction active stats for this type
        type_stats = None
        if fraction_active_stats is not None:
            type_stats = fraction_active_stats.get(neuron_type)
        
        pop_name = f"{neuron_type}_interneurons"
        population = InterneuronFeaturePopulation(
            name=pop_name,
            feature_space=feature_space,
            n_features=n_neurons,
            modality=modality,
            neuron_type=neuron_type,
            fraction_active_stats=type_stats
        )
        
        population.generate_features(start_gid=current_gid)
        
        populations[pop_name] = population
        current_gid += n_neurons
    
    return populations



def test_interneuron_features():
    """Test function to demonstrate interneuron feature functionality."""
    
    # Create test signal (multidimensional with varying magnitude)
    duration = 2.0  # seconds
    sample_rate = 1000  # Hz
    n_dimensions = 5
    n_samples = int(duration * sample_rate)
    
    # Create a signal with varying magnitude over time
    t = np.linspace(0, duration, n_samples)
    signal = np.zeros((n_samples, n_dimensions))
    
    # Different frequency components in each dimension
    for dim in range(n_dimensions):
        freq = 2 + dim * 3  # 2, 5, 8, 11, 14 Hz
        amplitude = 0.5
        phase = dim * np.pi / 2
        signal[:, dim] = amplitude * np.sin(2 * np.pi * freq * t - phase)


    # Create interneuron modality
    modality = InterneuronModality(
        name="interneuron",
        input_shape=(n_samples, n_dimensions),
        norm_type='l2',
        temporal_smoothing=0.01
    )
    
    # Create feature space and register modality
    feature_space = FeatureSpace(name="test_space")
    feature_space.register_modality(modality)
    
    # Create populations for different interneuron types
    population_sizes = {'PV': 50, 'SST': 30, 'VIP': 25}
    populations = create_interneuron_populations(
        feature_space=feature_space,
        modality=modality,
        population_sizes=population_sizes
    )
    
    processed_signal = modality.preprocess_signal(signal)
    
    print(f"Original signal shape: {signal.shape}")
    print(f"Processed signal shape: {processed_signal.shape}")
    print(f"Signal magnitude range: {np.min(processed_signal):.3f} - {np.max(processed_signal):.3f}")
    
    time_config = EncoderTimeConfig(duration_ms=1.0, dt_ms=1.0)
    
    responses = {}
    for pop_name, population in populations.items():
        print(f"\nProcessing {pop_name} population...")
        
        for feature in population.features.values():
            feature.initialize_encoder(time_config)
        
        # Get responses from first few neurons as examples
        pop_responses = []
        for i, (gid, feature) in enumerate(population.features.items()):
            if i >= 3:  # Just test first 3 neurons
                break
            response = feature.get_response(processed_signal)
            pop_responses.append(response)
        
        responses[pop_name] = pop_responses
        
        pop_stats = population.generate_population_response(np.mean(processed_signal))
        print(f"  Population stats: {pop_stats}")
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # original signal magnitude across dimensions
    axes[0].plot(t, np.linalg.norm(signal, axis=1))
    axes[0].set_title('Original Signal Magnitude (L2 norm across dimensions)')
    axes[0].set_ylabel('Magnitude')
    
    # processed signal
    axes[1].plot(t, processed_signal)
    axes[1].set_title('Normalized Signal Magnitude')
    axes[1].set_ylabel('Normalized Magnitude')
    
    # f-I curves for different interneuron types
    test_inputs = np.linspace(0, 1, 100)
    axes[2].set_title('f-I Curves for Different Interneuron Types')
    
    for neuron_type in ['PV', 'SST', 'VIP']:
        response_model = InterneuronResponseModel(neuron_type)
        firing_rates = [response_model.compute_firing_rate(inp) for inp in test_inputs]
        axes[2].plot(test_inputs, firing_rates, label=f'{neuron_type}', linewidth=2)
    
    axes[2].set_xlabel('Normalized Input Magnitude')
    axes[2].set_ylabel('Firing Rate (Hz)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    axes[3].set_title('Sample Spike Responses')
    y_offset = 0
    colors = ['red', 'blue', 'green']
    
    for i, (pop_name, pop_responses) in enumerate(responses.items()):
        color = colors[i % len(colors)]
        for j, response in enumerate(pop_responses):
            
            spike_times = response.reshape((-1,)) / 1000. # convert to s

            for spike_time in spike_times:
                axes[3].plot([spike_time, spike_time], 
                             [y_offset - 0.4, y_offset + 0.4], 
                             color=color, linewidth=1)
                
            y_offset += 1
    
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Neuron')
    axes[3].set_xlim(0, duration)
    
    plt.tight_layout()
    plt.show()
    
    return populations, processed_signal, responses


if __name__ == "__main__":
    populations, signal, responses = test_interneuron_features()
    
    print("Available interneuron types:", list(INTERNEURON_FI_PARAMS.keys()))
    for pop_name, pop in populations.items():
        print(f"{pop_name}: {pop.n_features} neurons of type {pop.neuron_type}")

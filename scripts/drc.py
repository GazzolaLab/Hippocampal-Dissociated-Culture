import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from dataclasses import dataclass, field
from miv_simulator.env import Env
from miv_simulator.input_features import (
    CoordinateSystemConfig,
    FeatureSpace,
    FeatureEncoding,
    InputModality,
    InputFeaturePopulation,
    InputFeature,
    EncoderTimeConfig
    )
from miv_simulator.input_spike_trains import generate_input_spike_trains
from input_signals import write_signal, read_signal, list_available_signals, create_multidimensional_signal
from mpi4py import MPI
import h5py
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('drc')


# Define custom reduction operations that concatenate lists and merge dictionaries
def list_concat(a, b, datatype):
    """Concatenate two lists of lists"""
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def dict_merge(a, b, datatype):
    """Merge two dictionaries"""
    d = {}
    if a is not None:
        d.update(a)
    if b is not None:
        d.update(b)
    return d

# Create MPI concatenation operation
list_concat_op = MPI.Op.Create(list_concat, commute=True)
dict_merge_op = MPI.Op.Create(dict_merge, commute=True)

def mpi_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    sys.stderr.flush()
    sys.stdout.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook

class SpatioTemporalModality(InputModality):
    """
    Modality for processing spatio-temporal patterns in neural networks.
    Combines spatial, spectral, and temporal dimensions.
    """
    
    def __init__(
        self,
        name: str = "spatio_temporal",
        input_shape: Tuple[int, ...] = (1000, 10),  # (time_samples, channels)
        temporal_bounds: Tuple[float, float] = (0, 1),  # Normalized time
        frequency_bounds: Tuple[float, float] = (1, 100),  # Hz
        spatial_bounds: Tuple[float, float] = (0, 1),  # Normalized space
        spatial_scale: str = "normalized",  # How spatial coordinates are scaled
        sample_rate: int = 1000,  # Hz
    ):
        # Define a 4D feature coordinate system:
        # 1. Temporal position (when in the signal this feature responds)
        # 2. Frequency preference (what frequency this feature detects)
        # 3. Spatial position (where in space this feature responds)
        # 4. Spatial extent (how broad/narrow the spatial tuning is)
        feature_coordinate_system = CoordinateSystemConfig(
            dimensions=4,
            bounds=[
                temporal_bounds,          # Time position (normalized 0-1)
                frequency_bounds,         # Frequency preference (Hz)
                spatial_bounds,           # Spatial position (normalized 0-1)
                (0.05, 0.5),              # Spatial tuning width (normalized)
            ],
            units=["normalized_time", "Hz", spatial_scale, "width"],
        )

        super().__init__(
            name,
            "spatio_temporal",
            feature_coordinate_system,
            input_shape=input_shape,  # (time_samples, channels)
        )

        self.sample_rate = sample_rate
        self.spatial_scale = spatial_scale

    def preprocess_signal(self, stimulus: np.ndarray) -> np.ndarray:
        """
        Preprocess a spatio-temporal signal.
        
        Parameters:
        -----------
        stimulus : np.ndarray
            Input stimulus of shape (time, channels) or (time,)
            
        Returns:
        --------
        np.ndarray
            Processed stimulus
        """
        # Ensure input is at least 2D (time, channels)
        if len(stimulus.shape) == 1:
            processed = stimulus.reshape(-1, 1)
        elif len(stimulus.shape) == 2:
            processed = stimulus.copy()
        else:
            raise ValueError(f"Expected 1D or 2D stimulus, got shape {stimulus.shape}")

        # Normalize amplitude to [-1, 1]
        if processed.max() > 1.0 or processed.min() < -1.0:
            processed = processed / np.max(np.abs(processed))

        return processed

    def to_feature_coordinates(self, modality_coordinates: np.ndarray) -> np.ndarray:
        """Convert modality-specific coordinates to feature space coordinates."""
        return modality_coordinates

    def from_feature_coordinates(self, feature_coordinates: np.ndarray) -> np.ndarray:
        """Convert feature space coordinates to modality-specific coordinates."""
        return feature_coordinates

    def create_input_filter(self, position: np.ndarray) -> Callable:
        """
        Create a spatio-temporal filter based on feature coordinates.
        
        Parameters:
        -----------
        position : np.ndarray
            Position in feature space [time_pos, freq, space_pos, space_width]
            
        Returns:
        --------
        Callable
            Filter function
        """
        time_pos, preferred_freq, space_pos, space_width = position

        def spatio_temporal_filter(signal: np.ndarray) -> np.ndarray:
            """
            Apply spatio-temporal filtering to the input signal.
            
            Parameters:
            -----------
            signal : np.ndarray
                Input signal of shape (time, channels)
                
            Returns:
            --------
            np.ndarray
                Filtered signal
            """
            original_shape = signal.shape
            if len(original_shape) == 1:
                signal_reshaped = signal.reshape(-1, 1)
            else:
                signal_reshaped = signal

            time_points, num_channels = signal_reshaped.shape
            filtered_signal = np.zeros_like(signal_reshaped, dtype=np.float32)

            # 1. Apply temporal filter (bandpass around preferred frequency)
            nyquist = self.sample_rate / 2
            lowcut = max(0.5, preferred_freq * 0.7) / nyquist
            highcut = min(preferred_freq * 1.3, nyquist * 0.95) / nyquist
            
            b, a = sp_signal.butter(2, [lowcut, highcut], btype='band')
            
            for ch in range(num_channels):
                filtered_signal[:, ch] = sp_signal.filtfilt(b, a, signal_reshaped[:, ch])
            
            # 2. Apply spatial filter (Gaussian weighting by channel position)
            # Assuming channels are evenly distributed in space [0, 1]
            channel_positions = np.linspace(0, 1, num_channels)
            
            # Compute Gaussian weights based on distance to preferred position
            spatial_weights = np.exp(
                -0.5 * ((channel_positions - space_pos) / space_width) ** 2
            )
            
            # Apply spatial weights to all timepoints
            for ch in range(num_channels):
                filtered_signal[:, ch] *= spatial_weights[ch]
            
            # 3. Apply temporal position filter (Gaussian window in time)
            if 0 <= time_pos <= 1:
                time_indices = np.linspace(0, 1, time_points)
                time_window = np.exp(-0.5 * ((time_indices - time_pos) / 0.1) ** 2)
                
                # Apply time window to all channels
                for ch in range(num_channels):
                    filtered_signal[:, ch] *= time_window
            
            # 4. Compute overall response (RMS energy or envelope)
            energy = np.sqrt(np.mean(filtered_signal ** 2, axis=1))
            
            # Normalize to [0, 1] range
            if np.max(energy) > 0:
                energy = energy / np.max(energy)
                
            if len(original_shape) == 1:
                return energy
            else:
                return energy.reshape(-1, 1)

        return spatio_temporal_filter
    
    def generate_feature_distribution(
        self, n_features: int, local_random: Optional[np.random.RandomState] = None
    ) -> List[np.ndarray]:
        """
        Generate a distribution of feature coordinates for this modality.
        
        Parameters:
        -----------
        n_features : int
            Number of features to generate
        local_random : np.random.RandomState, optional
            Random number generator
            
        Returns:
        --------
        List[np.ndarray]
            List of feature positions
        """
        if local_random is None:
            local_random = np.random.RandomState()

        positions = []

        t_min, t_max = self.feature_coordinate_system.bounds[0]  # temporal
        f_min, f_max = self.feature_coordinate_system.bounds[1]  # frequency
        s_min, s_max = self.feature_coordinate_system.bounds[2]  # spatial
        w_min, w_max = self.feature_coordinate_system.bounds[3]  # width

        # Calculate sample counts for each dimension based on n_features
        n_time = max(4, int(n_features ** 0.25))
        n_freq = max(5, int(n_features ** 0.3))
        n_space = max(4, int(n_features ** 0.25))
        n_width = max(2, int(n_features ** 0.2))
        
        time_points = np.linspace(t_min, t_max, n_time)
        # Log spacing for frequencies
        freq_points = np.logspace(np.log10(max(1.0, f_min)), np.log10(f_max), n_freq)
        space_points = np.linspace(s_min, s_max, n_space)
        width_points = np.linspace(w_min, w_max, n_width)
        
        # Generate all combinations (grid)
        for t in time_points:
            for f in freq_points:
                for s in space_points:
                    for w in width_points:
                        # Add small jitter to avoid perfect grid alignment
                        jitter_t = local_random.uniform(-0.5, 0.5) * (t_max - t_min) / (2 * n_time)
                        jitter_f = local_random.uniform(-0.5, 0.5) * f * 0.1  # 10% jitter
                        jitter_s = local_random.uniform(-0.5, 0.5) * (s_max - s_min) / (2 * n_space)
                        jitter_w = local_random.uniform(-0.5, 0.5) * (w_max - w_min) / (2 * n_width)
                        
                        positions.append(np.array([
                            t + jitter_t, 
                            f + jitter_f,
                            s + jitter_s,
                            w + jitter_w
                        ]))
        
        # Shuffle and limit to requested number
        local_random.shuffle(positions)
        return positions[:n_features]
    

class DynamicalResponsePopulation(InputFeaturePopulation):
    """
    Population of input features specialized for characterizing
    dynamical responses of biophysical networks to diverse
    spatio-temporal patterns.

    """

    def __init__(
        self,
        name: str,
        feature_space: "FeatureSpace",
        modality: InputModality,
        n_features: int,
        dimensions: List[Dict[str, Any]],
        sampling_strategy: str = "grid",
        density_function: Optional[Callable] = None,
        encoding_distribution: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize a DynamicalResponsePopulation.
        
        Parameters:
        -----------
        name : str
            Name of the population
        feature_space : FeatureSpace
            Parent feature space
        modality : InputModality
            The modality this population belongs to
        n_features : int
            Number of features to generate
        dimensions : List[Dict[str, Any]]
            Specification of dimensions to sample, each with:
            - name: dimension name
            - range: (min, max) values
            - scale: 'linear', 'log', or 'logistic'
            - priority: importance weight for sampling (higher = more samples)
        sampling_strategy : str
            How to sample the feature space: 'grid', 'random', 'stratified'
        density_function : Callable, optional
            Custom density function for sampling
        encoding_distribution : Dict[str, Any], optional
            Parameters for feature encoding
        random_seed : int, optional
            Random seed for reproducibility
        """
        super().__init__(
            name=name,
            feature_space=feature_space,
            n_features=n_features,
            modality=modality,
            density_function=density_function,
            encoding_distribution=encoding_distribution or {},
        )
        
        self.dimensions = dimensions
        self.sampling_strategy = sampling_strategy
        self.local_random = np.random.RandomState(random_seed)
        
        self.response_metrics = defaultdict(dict)
        
        self.dimension_stats = {dim["name"]: {} for dim in dimensions}


    def generate_features(
        self,
        start_gid: int = 0,
        rank: Optional[int] = None,
        size: Optional[int] = None,
        local_random: Optional[np.random.RandomState] = None,
    ) -> List[InputFeature]:
        """
        Generate features according to the specified dimensions and sampling strategy.

        Parameters:
        -----------
        start_gid : int
            Starting GID for feature numbering
        rank : Optional[int]
            MPI rank for distributed generation
        size : Optional[int]
            MPI size for distributed generation
        local_random : Optional[np.random.RandomState]
            Random generator to use

        Returns:
        --------
        List[InputFeature]
            Generated features
        """
        if local_random is None:
            local_random = self.local_random

        dim_names = [dim["name"] for dim in self.dimensions]
        dim_ranges = [dim["range"] for dim in self.dimensions]
        dim_scales = [dim.get("scale", "linear") for dim in self.dimensions]
        dim_priorities = [dim.get("priority", 1.0) for dim in self.dimensions]

        samples_per_dim = self._calculate_balanced_samples_per_dim(dim_priorities, self.n_features)
        
        features = []

        if self.sampling_strategy == "grid":
            # Create a grid sampling with appropriate scales
            positions = self._generate_grid_positions(dim_ranges, dim_scales, samples_per_dim)
        elif self.sampling_strategy == "random":
            # Random sampling with density function
            positions = self._generate_random_positions(dim_ranges, dim_scales, self.n_features)
        elif self.sampling_strategy == "stratified":
            # Stratified sampling to ensure coverage
            positions = self._generate_stratified_positions(dim_ranges, dim_scales, samples_per_dim)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        positions = positions[:self.n_features]

        # Determine which features to generate based on MPI rank/size
        if (rank is not None) and (size is not None):
            feature_indices = list(range(rank, min(self.n_features, len(positions)), size))
        else:
            feature_indices = list(range(min(self.n_features, len(positions))))

        for i in feature_indices:
            gid = start_gid + i

            position = positions[i]

            # Create position dictionary with named dimensions
            position_dict = {dim_names[j]: position[j] for j in range(len(dim_names))}

            # Generate encoding based on position
            encoding = self._generate_encoding_from_position(position, position_dict)

            # Create input filter based on position
            input_filter = self._create_input_filter_from_position(position, position_dict)

            metadata = {
                **position_dict,
                "dimension_values": position.tolist(),
            }

            feature = InputFeature(
                gid=gid, 
                position=position, 
                encoding=encoding,
                input_filter=input_filter,
                kwargs={"metadata": metadata}
            )

            features.append(feature)
            self.features[gid] = feature

            for j, dim_name in enumerate(dim_names):
                if dim_name not in self.dimension_stats:
                    self.dimension_stats[dim_name] = {"values": []}
                if "values" not in self.dimension_stats[dim_name]:
                    self.dimension_stats[dim_name]["values"] = []
                self.dimension_stats[dim_name]["values"].append(position[j])

        # Compute statistics for each dimension
        for dim_name in self.dimension_stats:
            if "values" in self.dimension_stats[dim_name]:
                values = np.array(self.dimension_stats[dim_name]["values"])
                self.dimension_stats[dim_name]["min"] = values.min() if len(values) > 0 else None
                self.dimension_stats[dim_name]["max"] = values.max() if len(values) > 0 else None
                self.dimension_stats[dim_name]["mean"] = values.mean() if len(values) > 0 else None
                self.dimension_stats[dim_name]["std"] = values.std() if len(values) > 0 else None

        return features
    

    def _calculate_balanced_samples_per_dim(self, dim_priorities, target_n_features):
        """
        Calculate number of samples per dimension to get close to target_n_features.

        Parameters:
        -----------
        dim_priorities : List[float]
            Priorities for each dimension
        target_n_features : int
            Target number of features to generate

        Returns:
        --------
        List[int]
            Number of samples for each dimension
        """
        # Initial calculation based on priorities
        n_dims = len(dim_priorities)
        total_priority = sum(dim_priorities)
        avg_priority = total_priority / n_dims

        # Initial distribution - root of n_features adjusted by priority
        initial_samples = [
            max(2, np.floor(target_n_features ** (1 / n_dims) * p / avg_priority))
            for p in dim_priorities
        ]

        samples_per_dim = [int(s) for s in initial_samples]
        current_total = np.prod(samples_per_dim)

        # If significantly short of the target, adjust the samples per dimension
        if current_total < 0.8 * target_n_features:
            # Calculate the scaling factor needed to reach the target
            scale_factor = (target_n_features / current_total) ** (1 / n_dims)

            # Apply the scaling factor to each dimension
            for i in range(n_dims):
                # Scale proportionally to priority and round to integer
                samples_per_dim[i] = max(2, int(samples_per_dim[i] * scale_factor))

            # Recalculate the total
            current_total = np.prod(samples_per_dim)

        # Sort dimensions by priority (descending)
        dim_indices = list(range(n_dims))
        dim_indices.sort(key=lambda i: dim_priorities[i], reverse=True)

        # Final adjustment to get closer to target
        max_iterations = 100
        iteration = 0

        # If still under the target, incrementally add samples to high-priority dimensions
        while current_total < target_n_features and iteration < max_iterations:
            progress_made = False

            for idx in dim_indices:
                old_total = current_total
                samples_per_dim[idx] += 1
                new_total = np.prod(samples_per_dim)

                # Keep the increment if it gets closer to the target
                if abs(new_total - target_n_features) < abs(old_total - target_n_features):
                    current_total = new_total
                    progress_made = True
                    if current_total >= target_n_features:
                        break
                else:
                    # Undo if it makes things worse
                    samples_per_dim[idx] -= 1

            # If no progress was made in this iteration, break to avoid infinite loop
            if not progress_made:
                break

            iteration += 1

        # If over the target, try to reduce samples from low-priority dimensions
        if current_total > 1.2 * target_n_features:
            # Reverse the dimension indices (low priority first)
            dim_indices.reverse()

            iteration = 0
            while current_total > target_n_features and iteration < max_iterations:
                progress_made = False

                for idx in dim_indices:
                    # Lower limit is 2 samples per dimension
                    if samples_per_dim[idx] <= 2:
                        continue

                    old_total = current_total
                    samples_per_dim[idx] -= 1
                    new_total = np.prod(samples_per_dim)

                    # Keep the reduction if it gets closer to the target
                    if abs(new_total - target_n_features) < abs(old_total - target_n_features):
                        current_total = new_total
                        progress_made = True
                    else:
                        # Undo if it makes things worse
                        samples_per_dim[idx] += 1

                if not progress_made:
                    break

                iteration += 1

        logger.info(f"Target features: {target_n_features}, Actual grid points: {np.prod(samples_per_dim)}")
        logger.info(f"Samples per dimension: {samples_per_dim}")

        return samples_per_dim

    def _generate_grid_positions(self, dim_ranges, dim_scales, samples_per_dim):
        """Generate positions on a grid with the specified scales."""

        grid_points = []
        for i, (d_range, d_scale, n_samples) in enumerate(zip(dim_ranges, dim_scales, samples_per_dim)):
            if d_scale == "linear":
                points = np.linspace(d_range[0], d_range[1], n_samples)
            elif d_scale == "log":
                points = np.logspace(np.log10(max(d_range[0], 1e-10)), np.log10(d_range[1]), n_samples)
            elif d_scale == "logistic":
                # Logistic scale concentrates points in the middle of the range
                u = np.linspace(0.1, 0.9, n_samples)
                points = d_range[0] + (d_range[1] - d_range[0]) * u
            else:
                raise ValueError(f"Unknown scale: {d_scale}")
            grid_points.append(points)
        
        mesh_grid = np.meshgrid(*grid_points, indexing='ij')
        
        positions = np.column_stack([grid.flatten() for grid in mesh_grid])
        
        # Shuffle to avoid systematic bias
        self.local_random.shuffle(positions)
        
        return positions

    def _generate_random_positions(self, dim_ranges, dim_scales, n_samples):
        """Generate random positions with the specified scales."""
        positions = np.zeros((n_samples, len(dim_ranges)))
        
        for i, (d_range, d_scale) in enumerate(zip(dim_ranges, dim_scales)):
            if d_scale == "linear":
                positions[:, i] = self.local_random.uniform(d_range[0], d_range[1], n_samples)
            elif d_scale == "log":
                positions[:, i] = np.power(
                    10, 
                    self.local_random.uniform(
                        np.log10(max(d_range[0], 1e-10)), 
                        np.log10(d_range[1]), 
                        n_samples
                    )
                )
            elif d_scale == "logistic":
                # Sample more densely in the middle
                u = self.local_random.uniform(0, 1, n_samples)
                # Apply sigmoid transformation
                v = 1 / (1 + np.exp(-6 * (u - 0.5)))
                positions[:, i] = d_range[0] + (d_range[1] - d_range[0]) * v
            else:
                raise ValueError(f"Unknown scale: {d_scale}")
        
        if self.density_function:
            weights = np.array([self.density_function(pos) for pos in positions])
            weights = weights / np.sum(weights)
            indices = self.local_random.choice(
                n_samples, size=n_samples, replace=True, p=weights
            )
            positions = positions[indices]
        
        return positions

    def _generate_stratified_positions(self, dim_ranges, dim_scales, samples_per_dim):
        """
        Generate stratified samples to ensure coverage of all regions.
        Uses dimension properties and priorities to guide sampling.
        """
        # generate a grid
        positions_grid = self._generate_grid_positions(dim_ranges, dim_scales, samples_per_dim)

        # find high-priority dimensions
        dim_names = [dim["name"] for dim in self.dimensions]
        dim_priorities = [dim.get("priority", 1.0) for dim in self.dimensions]
        high_priority_dims = [i for i, priority in enumerate(dim_priorities) 
                             if priority > 1.0]

        jitter = np.zeros_like(positions_grid)

        for i, (d_range, d_scale, n_samples) in enumerate(zip(dim_ranges, dim_scales, samples_per_dim)):
            cell_size = (d_range[1] - d_range[0]) / n_samples

            # Apply different jitter strategies based on scale type and priority
            if d_scale == "linear":
                # Higher priority dimensions get smaller jitter for more uniform coverage
                jitter_scale = 0.45
                if i in high_priority_dims:
                    jitter_scale = 0.35

                jitter[:, i] = self.local_random.uniform(-jitter_scale, jitter_scale, 
                                                        len(positions_grid)) * cell_size
            elif d_scale == "log":
                # For log scale, jitter is proportional to value
                jitter_scale = 0.15
                if i in high_priority_dims:
                    jitter_scale = 0.1

                rel_jitter = self.local_random.uniform(-jitter_scale, jitter_scale, 
                                                      len(positions_grid))
                jitter[:, i] = positions_grid[:, i] * rel_jitter
            elif d_scale == "logistic":
                # Similar to linear but scaled by logistic derivative
                x_norm = (positions_grid[:, i] - d_range[0]) / (d_range[1] - d_range[0])
                logistic_derivative = x_norm * (1 - x_norm)

                jitter_scale = 0.45
                if i in high_priority_dims:
                    jitter_scale = 0.35

                jitter[:, i] = self.local_random.uniform(-jitter_scale, jitter_scale, 
                                                       len(positions_grid)) * cell_size * logistic_derivative

        positions = positions_grid + jitter

        # Force additional coverage in poorly sampled regions for all dimensions
        if len(positions) > 10:
            for i, (d_range, priority) in enumerate(zip(dim_ranges, dim_priorities)):
                # Skip dimensions with low priority
                if priority < 1.0:
                    continue

                # Define regions to check (divide dimension into segments)
                n_regions = 5
                region_size = (d_range[1] - d_range[0]) / n_regions

                values = positions[:, i]
                samples_per_region = []

                for r in range(n_regions):
                    region_start = d_range[0] + r * region_size
                    region_end = region_start + region_size
                    region_samples = np.sum((values >= region_start) & (values < region_end))
                    samples_per_region.append(region_samples)

                # Find undersampled regions
                avg_samples = np.mean(samples_per_region)
                min_expected = max(2, avg_samples / 2)

                undersampled_regions = []
                for r in range(n_regions):
                    if samples_per_region[r] < min_expected:
                        undersampled_regions.append(r)

                # Add points to undersampled regions if needed
                if undersampled_regions and priority >= 1.0:
                    n_extra = max(len(undersampled_regions), 
                                  int(len(positions) * 0.05 * priority))

                    if n_extra > 0:
                        extra_positions = positions[:n_extra].copy()

                        # Assign new positions in undersampled regions
                        for idx, extra_pos in enumerate(extra_positions):

                            region = undersampled_regions[idx % len(undersampled_regions)]
                            region_start = d_range[0] + region * region_size

                            new_pos = region_start + self.local_random.random() * region_size
                            extra_pos[i] = new_pos

                        positions = np.vstack([positions, extra_positions])

        # Clip to range
        for i, d_range in enumerate(dim_ranges):
            positions[:, i] = np.clip(positions[:, i], d_range[0], d_range[1])

        return positions

    
    def _generate_encoding_from_position(self, position, position_dict):
        """Generate an encoding based on the feature's position in feature space."""
        # Default encoding parameters
        encoding_params = {
            "peak_rate": self.encoding_distribution.get("peak_rate", 100.0),
        }
        
        # Modify peak rate based on coordinates if requested
        # e.g. scale firing rate based on temporal frequency
        if "rate_scaling_factor" in self.encoding_distribution:
            if "temporal_frequency" in position_dict:
                freq = position_dict["temporal_frequency"]
                scale_factor = self.encoding_distribution["rate_scaling_factor"]
                encoding_params["peak_rate"] *= (1.0 + scale_factor * freq / 100.0)
        
        # Tuning parameters
        tuning_params = {}
        if self.encoding_distribution.get("feature_type") == "receptive_field":
            tuning_params["tuning_width"] = self.encoding_distribution.get("tuning_width", 0.2)
            
            # Modify tuning width based on position if requested
            # e.g. narrower tuning for higher spatial frequencies
            if "width_scaling_factor" in self.encoding_distribution:
                if "spatial_frequency" in position_dict:
                    sf = position_dict["spatial_frequency"]
                    scale_factor = self.encoding_distribution["width_scaling_factor"]
                    tuning_params["tuning_width"] *= (1.0 - scale_factor * sf / 10.0)
                    tuning_params["tuning_width"] = max(0.05, tuning_params["tuning_width"])
        
        return FeatureEncoding(
            feature_type=self.encoding_distribution.get("feature_type", "linear_rate"),
            encoder_params=encoding_params,
            tuning_params=tuning_params,
        )

    def _create_input_filter_from_position(self, position, position_dict):
        """Create an appropriate input filter based on the feature's position."""

        # Delegate to modality's filter if available
        if hasattr(self.modality, "create_input_filter"):
            return self.modality.create_input_filter(position)
        
        # Otherwise create a generic filter based on dimensions
        def generic_filter(signal):
            """Filter the input signal based on feature position."""
            # TODO: review generic filtering logic
            filtered_signal = signal.copy()
            
            # Simple bandpass filter around the preferred frequency
            if "temporal_frequency" in position_dict:
                freq = position_dict["temporal_frequency"]
                if hasattr(self.modality, "sample_rate"):
                    sample_rate = self.modality.sample_rate
                else:
                    sample_rate = 1000.0  # Default assumption 1kHz
                
                b, a = sp_signal.butter(
                    2,
                    [(0.5 * freq) / (0.5 * sample_rate), (1.5 * freq) / (0.5 * sample_rate)],
                    btype='band'
                )
                
                if len(filtered_signal.shape) == 1:
                    filtered_signal = sp_signal.filtfilt(b, a, filtered_signal)
                else:
                    for c in range(filtered_signal.shape[1]):
                        filtered_signal[:, c] = sp_signal.filtfilt(b, a, filtered_signal[:, c])
            
            if "spatial_location" in position_dict:
                # TODO: target specific spatial structure of the signal
                # e.g.: weight by distance from preferred location
                pass
            
            return filtered_signal
        
        return generic_filter

    def initialize_encoders(self, time_config: "EncoderTimeConfig"):
        """Initialize all feature encoders with the given time configuration."""
        for feature in self.features.values():
            feature.initialize_encoder(time_config)

    def process_stimulus(self, stimulus: np.ndarray, time_config: "EncoderTimeConfig", batch_size: int = 10):
        """
        Process a stimulus and get responses from all features.
        
        Parameters:
        -----------
        stimulus : np.ndarray
            Input stimulus
        time_config : EncoderTimeConfig
            Time configuration for encoding
        batch_size : int
            Number of features to process in each batch
            
        Returns:
        --------
        Dict[int, Any]
            Dictionary mapping GIDs to responses
        """
        self.initialize_encoders(time_config)
        
        processed_stimulus = self.modality.preprocess_signal(stimulus)
        
        responses = {}
        feature_batches = np.array_split(
            list(self.features.items()), 
            max(1, len(self.features) // batch_size)
        )
        
        for batch in feature_batches:
            for gid, feature in batch:
                activation = feature.input_filter(processed_stimulus)
                response = feature.get_response(processed_stimulus)
                responses[gid] = {
                    'activation': activation,
                    'spikes': response
                }
        
        return responses

    def analyze_responses(self,
                          responses: Dict[int, Dict[str, Any]],
                          metrics: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None):
        """
        Analyze feature responses and compute metrics.
        
        Parameters:
        -----------
        responses : Dict[int, Dict]
            Dictionary of responses from process_stimulus
        metrics : List[str], optional
            List of metrics to compute, defaults to ['mean', 'max', 'std', 'fano']
            
        Returns:
        --------
        Dict
            Dictionary of computed metrics
        """
        if metrics is None:
            metrics = ['mean', 'max', 'std', 'fano']
        
        results = defaultdict(dict)
        
        for gid, response in responses.items():

            feature = self.features[gid]
            activation = response['activation']
            spikes = response['spikes']
            
            # Compute activation metrics
            if 'mean' in metrics:
                results[gid]['mean_activation'] = np.mean(activation)
            
            if 'max' in metrics:
                results[gid]['max_activation'] = np.max(activation)
                
            if 'std' in metrics:
                results[gid]['std_activation'] = np.std(activation)
            
            if hasattr(spikes, 'shape') and len(spikes.shape) > 0:
                if isinstance(spikes, list):
                    spike_counts = [len(s) for s in spikes]
                else:
                    spike_counts = np.sum(spikes, axis=1)
                
                if 'spike_rate' in metrics:
                    results[gid]['spike_rate'] = np.mean(spike_counts)
                
                if 'fano' in metrics and len(spike_counts) > 1:
                    # Fano factor (variance/mean ratio)
                    mean_count = np.mean(spike_counts)
                    if mean_count > 0:
                        results[gid]['fano_factor'] = np.var(spike_counts) / mean_count
                    else:
                        results[gid]['fano_factor'] = 0
            
            for key, value in feature.kwargs.get('metadata', {}).items():
                results[gid][key] = value
        
        # Compute population-level metrics
        population_metrics = {}
        
        # Correlations between metrics and dimensions
        for dim in self.dimensions:
            dim_name = dim["name"]
            dim_values = [results[gid][dim_name] for gid in results]
            
            for metric in ['mean_activation', 'spike_rate', 'fano_factor']:
                if all(metric in results[gid] for gid in results):
                    metric_values = [results[gid][metric] for gid in results]
                    corr = np.corrcoef(dim_values, metric_values)[0, 1]
                    population_metrics[f'{dim_name}_{metric}_correlation'] = corr
        
        self.response_metrics = results
        self.population_metrics = population_metrics
        
        return {
            'feature_metrics': results,
            'population_metrics': population_metrics
        }

    def plot_responses(self, metrics_dict=None, plot_dimensions=None, figsize=(15, 10), cmap='viridis'):
        """
        Visualize feature responses across dimensions.
        
        Parameters:
        -----------
        metrics_dict : Dict, optional
            Dictionary of metrics from analyze_responses, uses stored metrics if None
        plot_dimensions : List[str], optional
            Dimensions to plot, defaults to first two dimensions
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Generated figure
        """
        if metrics_dict is None:
            metrics_dict = self.response_metrics
            
        if not metrics_dict:
            raise ValueError("No metrics available. Run analyze_responses first.")
            
        # Default to first two dimensions if not specified
        if plot_dimensions is None:
            plot_dimensions = [dim["name"] for dim in self.dimensions[:2]]
            
        if len(plot_dimensions) < 2:
            raise ValueError("Need at least two dimensions to plot")
            
        gids = list(metrics_dict.keys())
        x_values = [metrics_dict[gid][plot_dimensions[0]] for gid in gids]
        y_values = [metrics_dict[gid][plot_dimensions[1]] for gid in gids]
        
        available_metrics = set()
        for gid in gids:
            available_metrics.update(
                key for key in metrics_dict[gid].keys() 
                if key not in [dim["name"] for dim in self.dimensions]
            )
        
        n_metrics = min(len(available_metrics), 6)  # Limit to 6 metrics
        plot_metrics = list(available_metrics)[:n_metrics]
        
        fig = plt.figure(figsize=figsize)
        
        grid_size = int(np.ceil(np.sqrt(n_metrics)))
        
        for i, metric in enumerate(plot_metrics):
            ax = fig.add_subplot(grid_size, grid_size, i+1)

            metric_array = np.asarray([metrics_dict[gid].get(metric, 0) for gid in gids])
            norm_metric_array = metric_array / np.max(metric_array)
            color_values = norm_metric_array

            scatter = ax.scatter(
                x_values, y_values, 
                c=color_values, 
                s=50, alpha=0.7,
                cmap=cmap
            )
            
            ax.set_xlabel(plot_dimensions[0])
            ax.set_ylabel(plot_dimensions[1])
            ax.set_title(f"{metric}")
            
            plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        return fig

    def get_feature_tuning_curves(self, dimension, responses=None):
        """
        Generate tuning curves for all features along a specific dimension.
        
        Parameters:
        -----------
        dimension : str
            Dimension name
        responses : Dict, optional
            Responses dict, uses stored metrics if None
            
        Returns:
        --------
        Dict
            Tuning curves for each feature
        """
        if responses is None:
            responses = self.response_metrics
            
        if not responses:
            raise ValueError("No responses available")
            
        # Find index of the dimension
        dim_idx = next(
            (i for i, d in enumerate(self.dimensions) if d["name"] == dimension), 
            None
        )
        
        if dim_idx is None:
            raise ValueError(f"Dimension {dimension} not found")
            
        dim_values = [self.features[gid].position[dim_idx] for gid in self.features]
        
        unique_values = sorted(set(dim_values))
        
        tuning_curves = {}
        
        for gid, feature in self.features.items():
            if gid in responses:
                tuning_curves[gid] = {
                    'dimension_value': feature.position[dim_idx],
                    'metrics': responses[gid]
                }
        
        return tuning_curves, unique_values

    def plot_population_tuning(self, dimension, metric='mean_activation', responses=None):
        """
        Plot population tuning curves for a specific dimension and metric.
        
        Parameters:
        -----------
        dimension : str
            Dimension name to show tuning for
        metric : str
            Metric to use for tuning curve
        responses : Dict, optional
            Responses dict, uses stored metrics if None
            
        Returns:
        --------
        matplotlib.figure.Figure
            Generated figure
        """
        if responses is None:
            responses = self.response_metrics
            
        tuning_curves, unique_values = self.get_feature_tuning_curves(dimension, responses)
        
        grouped_metrics = defaultdict(list)
        
        for gid, data in tuning_curves.items():
            dim_value = data['dimension_value']
            if metric in data['metrics']:
                grouped_metrics[dim_value].append(data['metrics'][metric])
        
        x_values = sorted(grouped_metrics.keys())
        y_mean = [np.mean(grouped_metrics[x]) for x in x_values]
        y_std = [np.std(grouped_metrics[x]) for x in x_values]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(x_values, y_mean, 'b-', linewidth=2)
        ax.fill_between(
            x_values, 
            [y - s for y, s in zip(y_mean, y_std)],
            [y + s for y, s in zip(y_mean, y_std)],
            alpha=0.3, color='b'
        )
        
        ax.set_xlabel(dimension)
        ax.set_ylabel(metric)
        ax.set_title(f"Population tuning curve for {dimension}")
        
        dim_details = next((d for d in self.dimensions if d["name"] == dimension), None)
        if dim_details and dim_details.get("scale") == "log":
            ax.set_xscale('log')
        
        return fig

    def plot_tuning_heatmap(
        self, 
        x_dimension: str, 
        y_dimension: str, 
        metric: str = 'mean_activation',
        responses=None,
        figsize=(12, 10),
        cmap: str = 'viridis',
        log_scale: List[str] = None,
        interpolation: str = 'bicubic'
    ):
        """
        Plot a 2D heatmap showing population tuning across two dimensions.

        Parameters:
        -----------
        x_dimension : str
            First dimension name for the x-axis
        y_dimension : str
            Second dimension name for the y-axis
        metric : str
            Metric to visualize (e.g., 'mean_activation', 'spike_rate')
        responses : Dict, optional
            Responses dict, uses stored metrics if None
        figsize : Tuple[int, int]
            Figure size
        cmap : str
            Colormap for the heatmap
        log_scale : List[str], optional
            List of dimensions to display in log scale (e.g., ['temporal_frequency'])
        interpolation : str
            Interpolation method for heatmap

        Returns:
        --------
        matplotlib.figure.Figure
            Generated figure
        """
        from scipy.interpolate import griddata

        if responses is None:
            responses = self.response_metrics

        if not responses:
            raise ValueError("No responses available. Run analyze_responses first.")

        x_dim_idx = next(
            (i for i, d in enumerate(self.dimensions) if d["name"] == x_dimension), 
            None
        )
        y_dim_idx = next(
            (i for i, d in enumerate(self.dimensions) if d["name"] == y_dimension), 
            None
        )

        if x_dim_idx is None or y_dim_idx is None:
            raise ValueError(f"Dimension {x_dimension} or {y_dimension} not found")

        # Default to linear scale if not specified
        if log_scale is None:
            log_scale = []

        # Extract data points
        x_values = []
        y_values = []
        z_values = []

        for gid, feature in self.features.items():
            if gid in responses and metric in responses[gid]:
                x_val = feature.position[x_dim_idx]
                y_val = feature.position[y_dim_idx]
                z_val = responses[gid][metric]

                x_values.append(x_val)
                y_values.append(y_val)
                z_values.append(z_val)

        # Handle empty data
        if not x_values:
            raise ValueError(f"No data points available for {metric}")

        n_grid = 100

        if x_dimension in log_scale and np.min(x_values) > 0:
            x_grid = np.logspace(
                np.log10(np.min(x_values)),
                np.log10(np.max(x_values)),
                num=n_grid
            )
        else:
            x_grid = np.linspace(min(x_values), max(x_values), num=n_grid)

        if y_dimension in log_scale and np.min(y_values) > 0:
            y_grid = np.logspace(
                np.log10(np.min(y_values)),
                np.log10(np.max(y_values)),
                num=n_grid
            )
        else:
            y_grid = np.linspace(min(y_values), max(y_values), num=n_grid)

        X, Y = np.meshgrid(x_grid, y_grid)

        Z = griddata(
            (x_values, y_values), 
            z_values, 
            (X, Y), 
            method='cubic',
            fill_value=np.nan
        )

        # Replace any remaining NaNs with nearest interpolation
        mask = np.isnan(Z)
        if np.any(mask):
            Z_nearest = griddata(
                (x_values, y_values), 
                z_values, 
                (X, Y), 
                method='nearest'
            )
            Z[mask] = Z_nearest[mask]

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            Z,
            origin='lower',
            extent=[min(x_grid), max(x_grid), min(y_grid), max(y_grid)],
            aspect='auto',
            cmap=cmap,
            interpolation=interpolation
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(metric)

        if x_dimension in log_scale:
            ax.set_xscale('log')

        if y_dimension in log_scale:
            ax.set_yscale('log')

        ax.set_xlabel(x_dimension)
        ax.set_ylabel(y_dimension)

        ax.set_title(f"Tuning Heatmap: {metric} across {x_dimension} and {y_dimension}")

        scatter = ax.scatter(
            x_values, 
            y_values, 
            c='white',
            s=10,
            alpha=0.3,
            edgecolors='black',
            linewidths=0.5
        )

        plt.tight_layout()
        return fig

    def plot_3d_tuning_surface(
        self, 
        x_dimension: str, 
        y_dimension: str, 
        metric: str = 'mean_activation',
        responses=None,
        figsize=(12, 10),
        cmap: str = 'viridis',
        elev: float = 30,
        azim: float = -60,
        log_scale: List[str] = None
    ):
        """
        Plot a 3D surface showing population tuning across two dimensions.

        Parameters:
        -----------
        x_dimension : str
            First dimension name for the x-axis
        y_dimension : str
            Second dimension name for the y-axis
        metric : str
            Metric to visualize (e.g., 'mean_activation', 'spike_rate')
        responses : Dict, optional
            Responses dict, uses stored metrics if None
        figsize : Tuple[int, int]
            Figure size
        cmap : str
            Colormap for the surface
        elev : float
            Elevation angle for 3D view
        azim : float
            Azimuth angle for 3D view
        log_scale : List[str], optional
            List of dimensions to display in log scale (e.g., ['temporal_frequency'])

        Returns:
        --------
        matplotlib.figure.Figure
            Generated figure
        """
        from scipy.interpolate import griddata
        from mpl_toolkits.mplot3d import Axes3D

        if responses is None:
            responses = self.response_metrics

        if not responses:
            raise ValueError("No responses available. Run analyze_responses first.")

        x_dim_idx = next(
            (i for i, d in enumerate(self.dimensions) if d["name"] == x_dimension), 
            None
        )
        y_dim_idx = next(
            (i for i, d in enumerate(self.dimensions) if d["name"] == y_dimension), 
            None
        )

        if x_dim_idx is None or y_dim_idx is None:
            raise ValueError(f"Dimension {x_dimension} or {y_dimension} not found")

        if log_scale is None:
            log_scale = []

        # Extract data points
        x_values = []
        y_values = []
        z_values = []

        for gid, feature in self.features.items():
            if gid in responses and metric in responses[gid]:
                x_val = feature.position[x_dim_idx]
                y_val = feature.position[y_dim_idx]
                z_val = responses[gid][metric]

                x_values.append(x_val)
                y_values.append(y_val)
                z_values.append(z_val)

        if not x_values:
            raise ValueError(f"No data points available for {metric}")

        x_unique = np.sort(np.unique(x_values))
        y_unique = np.sort(np.unique(y_values))

        if x_dimension in log_scale and np.min(x_values) > 0:
            x_grid = np.logspace(
                np.log10(np.min(x_values)),
                np.log10(np.max(x_values)),
                num=50
            )
        else:
            x_grid = np.linspace(min(x_values), max(x_values), num=50)

        if y_dimension in log_scale and np.min(y_values) > 0:
            y_grid = np.logspace(
                np.log10(np.min(y_values)),
                np.log10(np.max(y_values)),
                num=50
            )
        else:
            y_grid = np.linspace(min(y_values), max(y_values), num=50)

        X, Y = np.meshgrid(x_grid, y_grid)

        Z = griddata(
            (x_values, y_values), 
            z_values, 
            (X, Y), 
            method='cubic',
            fill_value=np.nan
        )

        # Replace any remaining NaNs with nearest interpolation
        mask = np.isnan(Z)
        if np.any(mask):
            Z_nearest = griddata(
                (x_values, y_values), 
                z_values, 
                (X, Y), 
                method='nearest'
            )
            Z[mask] = Z_nearest[mask]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(
            X, Y, Z, 
            cmap=cmap,
            linewidth=0, 
            antialiased=True,
            alpha=0.8
        )

        scatter = ax.scatter(
            x_values, 
            y_values, 
            z_values,
            c=z_values,
            cmap=cmap,
            s=30,
            alpha=0.8
        )

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=metric)

        x_label = x_dimension
        y_label = y_dimension

        if x_dimension in log_scale:
            x_label += " (log scale)"
            ax.set_xscale('log')

        if y_dimension in log_scale:
            y_label += " (log scale)"
            ax.set_yscale('log')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(metric)

        ax.view_init(elev=elev, azim=azim)

        ax.set_title(f"3D Tuning Surface: {metric} across {x_dimension} and {y_dimension}")

        plt.tight_layout()
        return fig

    def export_metadata(self, signal_id, stimulus, output_path=None):
        """
        Export feature population data.

        Parameters:
        -----------
        signal_id : str
            Identifier for the signal
        stimulus : np.ndarray
            The stimulus data
        output_path : str, optional
            Pathname to save data to (HDF5 format)

        Returns:
        --------
        Dict
            Exported data dictionary
        """
        export_data = {
            'name': self.name,
            'n_features': self.n_features,
            'dimensions': self.dimensions,
            'dimension_stats': self.dimension_stats,
            'population_metrics': getattr(self, 'population_metrics', {}),
        }

        # Create flat dictionaries for features
        feature_data = {
            'gids': [],
            'positions': []
        }

        metadata_keys = set()
        metric_keys = set()

        # Collect all metadata and metric keys
        for gid, feature in self.features.items():
            metadata_dict = feature.kwargs.get('metadata', {})
            for key in metadata_dict:
                metadata_keys.add(key)

            if hasattr(self, 'response_metrics') and gid in self.response_metrics:
                for key in self.response_metrics[gid]:
                    metric_keys.add(key)

        for key in metadata_keys:
            feature_data[f'metadata_{key}'] = []

        for key in metric_keys:
            feature_data[f'metric_{key}'] = []

        # Collect data in column format
        for gid, feature in self.features.items():
            feature_data['gids'].append(gid)
            feature_data['positions'].append(feature.position)

            metadata_dict = feature.kwargs.get('metadata', {})
            for key in metadata_keys:
                if key in metadata_dict:
                    feature_data[f'metadata_{key}'].append(metadata_dict[key])
                else:
                    feature_data[f'metadata_{key}'].append(None)  # Use None for missing values

            if hasattr(self, 'response_metrics') and gid in self.response_metrics:
                metrics_dict = self.response_metrics[gid]
                for key in metric_keys:
                    if key in metrics_dict:
                        feature_data[f'metric_{key}'].append(metrics_dict[key])
                    else:
                        feature_data[f'metric_{key}'].append(None)
            else:
                for key in metric_keys:
                    feature_data[f'metric_{key}'].append(None)

        # Save to h5 file if requested
        if output_path:
            write_signal(output_path, self.name, self.dimensions,
                         signal_id, stimulus)

        return export_data

    
def create_dynamical_response_system(
        env,
        population_name = "dynamical_response_features",
        n_features=100,
        random_seed=42,
        sample_dt_ms = 1.0,
        stimulus_duration = 10.0,  # Overall signal duration [s]
        sampling_strategy = 'stratified', # "grid", "random", "stratified"
        register_population = True,
):
    """
    Create a an input feature system for dynamical response characterization.
    
    Parameters:
    -----------
    n_features : int
        Number of features to generate
    stimulus_duration : int
        Duration of stimulus in seconds
    sample_rate_hz : int
        Sample rate in Hz
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple
        (feature_space, modality, population, time_config)
    """
    local_random = np.random.RandomState(random_seed)
    
    feature_space = FeatureSpace(name="dynamical_response_space")

    comm = env.comm
    

    sample_rate_hz = 1000.0 / sample_dt_ms  # Sample rate [Hz]
    stimulus_duration_ms=stimulus_duration * 1000.0

    
    # Create spatio-temporal modality
    spatio_temporal_modality = SpatioTemporalModality(
        name="spatio_temporal",
        input_shape=(int(stimulus_duration * sample_rate_hz), ),
        temporal_bounds=(0, 1),
        frequency_bounds=(1, 100),
        spatial_bounds=(0, 1),
        sample_rate=sample_rate_hz,
    )
    
    # Register the modality
    feature_space.register_modality(spatio_temporal_modality)
    
    # Define dimensions for the feature population
    dimensions = [
        {
            "name": "temporal_position",
            "range": (0, 1),
            "scale": "linear",
            "priority": 1.0,
        },
        {
            "name": "temporal_frequency",
            "range": (1, 100),
            "scale": "log",
            "priority": 2.0,  # Higher priority = more samples along this dimension
        },
        {
            "name": "spatial_position",
            "range": (0, 1),
            "scale": "linear",
            "priority": 1.5,
        },
        {
            "name": "spatial_width",
            "range": (0.05, 0.5),
            "scale": "linear",
            "priority": 0.5,  # Lower priority = fewer samples along this dimension
        },
    ]

    start_gid = 0
    if population_name in env.celltypes:
        start_gid = env.celltypes[population_name]["start"]
        n_features = env.celltypes[population_name]["num"]
    
    # Create dynamical response population
    population = DynamicalResponsePopulation(
        name=population_name,
        feature_space=feature_space,
        modality=spatio_temporal_modality,
        n_features=n_features,
        dimensions=dimensions,
        sampling_strategy=sampling_strategy,  # "grid", "random", "stratified"
        encoding_distribution={
            "feature_type": "linear_rate",
            "peak_rate": 100.0,
            "rate_scaling_factor": 0.5,  # Higher frequencies = higher rates
        },
        random_seed=random_seed,
    )
    
    # Generate features
    population.generate_features(start_gid=start_gid,
                                 rank=comm.rank, size=comm.size)
    
    # Create time config for encoding
    time_config = EncoderTimeConfig(
        duration_ms=1,  # Process 1 ms at a time
        dt_ms=1.0,
    )

    # Determine population id
    if register_population:
        population_info = env.register_population(population.name,
                                                  {"All": n_features})
    if population_info is not None:
        start_gid = population_info['population_start_gid']

    population.start_gid = start_gid
    
    return feature_space, spatio_temporal_modality, population, time_config


def run_dynamical_response_characterization(signal_id = None,
                                            dataset_prefix = "./datasets",
                                            output_prefix = ".",
                                            config_prefix = "./config",
                                            config = "Dynamical_Response_Features.yaml",
                                            population_name = "dynamical_response_features",
                                            register_population = True,
                                            input_signal_file = None,  # Set to path of HDF5 file to read signal from
                                            stimulus_duration = 1,
                                            n_features = 150,
                                            sample_dt_ms=1.0,
                                            random_seed = 42,
                                            n_dimensions = 64,
                                            dry_run = False,
                                            plot = True,
                                            output_path = None,
                                            comm = None,
                                            io_kwargs = {
                                                'io_size': 1,
                                                'write_size': 1,
                                                'chunk_size': 10000,
                                                'value_chunk_size': 20000,
                                                },
                                            ):
    """Run the dynamical response characterization system."""

    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if (input_signal_file is None) and (signal_id is None):
        signal_id = "drc_signal"
    
    # np.seterr(all="raise")
    params = dict(locals())
    # params["config"] = params.pop("config_file")
    params["Model Name"] = "dynamical_response_features"
    params["config"] = config
    params["config_prefix"] = config_prefix
    env = Env(**params)
    
    # Create the system
    feature_space, modality, population, time_config = create_dynamical_response_system(
        env,
        population_name=population_name,
        n_features=n_features,
        stimulus_duration=stimulus_duration,
        sample_dt_ms=sample_dt_ms,
        random_seed=random_seed,
        register_population=register_population,
    )

    n_features = population.n_features

    comm = env.comm
    rank = env.comm.rank
    
    # Create a test stimulus with different spatio-temporal patterns
    duration_ms = stimulus_duration * 1000.0
    sample_rate_hz = 1000.0 / sample_dt_ms # Sample rate [Hz]

    stimulus = None
    t = None
    signal_metadata = {}
    input_signal_id = signal_id

    # Signal input configuration
    use_generated_signal = input_signal_file is None

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
                    
                if input_signal_id is not None:
                    signal_id = input_signal_id
                    
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
    signal_data = comm.bcast((stimulus, t, signal_metadata, stimulus_duration, n_dimensions), root=0)
    stimulus, t, signal_metadata, stimulus_duration, n_dimensions = signal_data

    if stimulus is not None:
        signal_id = input_signal_id
            
    # Fall back to generated signal if reading failed or not requested
    if stimulus is None:
        if rank == 0:
            logging.info("Generating new test signal")

            t, stimulus = create_multidimensional_signal(
                duration=stimulus_duration,
                sample_rate=sample_rate_hz, 
                n_dimensions=n_dimensions,
                signal_type="mixed"
            )

            signal_metadata = {
                'source': 'generated',
                'signal_type': 'mixed',
                'duration': stimulus_duration,
                'sample_rate': sample_rate_hz,
                'n_dimensions': n_dimensions
            }
    comm.barrier()
    
    # Broadcast signal reading results to all ranks
    if comm.size > 1:
        signal_data = comm.bcast((stimulus, t, signal_metadata, stimulus_duration, n_dimensions), root=0)
        stimulus, t, signal_metadata, stimulus_duration, n_dimensions = signal_data
    
    if output_prefix is not None:
        output_path = os.path.join(output_prefix, output_path)

    if not dry_run:
        output_spikes_namespace = f"Spatiotemporal Feature Spikes"
        generate_input_spike_trains(
            env,
            population,
            signal=stimulus,
            signal_id=signal_id,
            coords_path=None,
            output_path=output_path,
            output_spikes_namespace=output_spikes_namespace,
            output_spike_train_attr_name="Spike Train",
            **io_kwargs,
        )

    export_data = None
    if rank == 0:
        export_data = population.export_metadata(signal_id,
                                                 stimulus,
                                                 output_path=output_path if not dry_run else None)
    comm.barrier()
    

    analysis_results = None
    if plot:
        # Process the stimulus and obtain spike responses
        local_spike_responses = population.process_stimulus(stimulus, time_config)

        # Analyze responses
        local_analysis_results = population.analyze_responses(local_spike_responses)
        
        #positions = np.array(comm.reduce(local_positions, op=list_concat_op, root=0))
        response_metrics = comm.reduce(local_analysis_results['feature_metrics'], op=dict_merge_op, root=0)
        spike_responses = comm.reduce(local_spike_responses, op=dict_merge_op, root=0)
        
    if plot and (rank == 0):

        fig1 = population.plot_responses(response_metrics)
        fig1a = population.plot_responses(response_metrics, plot_dimensions=['temporal_frequency',
                                                                            'spatial_position'])
    
        # Show population tuning curves
        fig2 = population.plot_population_tuning('temporal_frequency', metric='mean_activation',
                                                 responses=response_metrics)
        fig3 = population.plot_population_tuning('spatial_position', metric='mean_activation',
                                                 responses=response_metrics)

        # 3D surface plot for temporal frequency and spatial position together
        fig4 = population.plot_3d_tuning_surface(
            responses=response_metrics,
            x_dimension='temporal_frequency',
            y_dimension='spatial_position',
            metric='mean_activation',
            cmap='inferno',
            #log_scale=['temporal_frequency'],  # Apply log scale to frequency axis
            elev=30,  # Elevation viewing angle
            azim=-45   # Azimuth viewing angle
        )
        
        # 2D heatmap for the same dimensions as above
        fig5 = population.plot_tuning_heatmap(
            responses=response_metrics,
            x_dimension='temporal_frequency',
            y_dimension='spatial_position',
            metric='mean_activation',
            #log_scale=['temporal_frequency'],
            cmap='inferno'
        )
        
        # temporal position vs. temporal frequency plot:
        fig6 = population.plot_tuning_heatmap(
            responses=response_metrics,
            x_dimension='temporal_position',
            y_dimension='temporal_frequency',
            metric='mean_activation',
            #log_scale=['temporal_frequency']
        )
        
        plt.show()
        
    comm.barrier()
    return population, analysis_results, export_data

if __name__ == "__main__":
    
    run_dynamical_response_characterization(signal_id = "drc_features_20240912",
                                            config = "Network_Clamp_PYR_gid_48041.yaml",
                                            stimulus_duration = 10,
                                            dataset_prefix = "datasets",
                                            output_path = "dynamical_response_spike_trains_n150_10s_3.h5",
                                            output_prefix = "datasets",
                                            population_name = "DRC",
                                            n_features = 150,
                                            io_kwargs={'io_size': 1,
                                                       'write_size': 10,
                                                       },
                                            dry_run = True)



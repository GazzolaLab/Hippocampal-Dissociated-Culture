import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings

# Set matplotlib to use LaTeX rendering as specified in user preferences
plt.rcParams.update({
    'text.usetex': False,  # Set to True if LaTeX is available
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

def plot_individual_neuron_mfdfa(
    mfdfa_result: Dict,
    neuron_gid: int,
    analysis_type: str = 'rate',
    figsize: Tuple[float, float] = (15, 10),
    save_path: Optional[str] = None,
    show_profile: bool = True
) -> plt.Figure:
    """
    Create MFDFA visualization for a single neuron.
    
    Parameters:
    -----------
    mfdfa_result : dict
        MFDFA analysis result for single neuron
    neuron_gid : int
        Neuron identifier
    analysis_type : str
        Type of analysis ('rate', 'isi', 'count')
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
    show_profile : bool
        Whether to show time series and profile plots
        
    Returns:
    --------
    matplotlib.Figure
        Generated figure object
    """
    if not mfdfa_result.get('valid_analysis', False):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'No valid MFDFA analysis for neuron {neuron_gid}\n({analysis_type})',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    # Determine subplot layout
    n_rows = 3 if show_profile else 2
    n_cols = 2
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.3)
    
    # Extract data
    scales = mfdfa_result['scales']
    q_values = mfdfa_result['q_values']
    fluctuation_matrix = mfdfa_result['fluctuation_function']
    scaling_exponents = mfdfa_result['scaling_exponents']
    spectrum = mfdfa_result['multifractal_spectrum']
    hurst = mfdfa_result['hurst_exponent']
    multifractality = mfdfa_result['multifractality_strength']
    
    # 1. Fluctuation Function F(s,q)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot subset of q values for clarity
    q_indices = np.linspace(0, len(q_values)-1, min(8, len(q_values)), dtype=int)
    colors_q = plt.cm.viridis(np.linspace(0, 1, len(q_indices)))
    
    for i, q_idx in enumerate(q_indices):
        q = q_values[q_idx]
        f_values = fluctuation_matrix[:, q_idx]
        valid_mask = f_values > 0
        
        if np.any(valid_mask):
            ax1.loglog(scales[valid_mask], f_values[valid_mask], 
                      'o-', color=colors_q[i], label=f'q = {q:.1f}', 
                      markersize=4, linewidth=1.5)
    
    ax1.set_xlabel('Scale s')
    ax1.set_ylabel('F(s,q)')
    ax1.set_title(f'Fluctuation Function - Neuron {neuron_gid}')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Scaling Exponents h(q)
    ax2 = fig.add_subplot(gs[0, 1])
    
    valid_h_mask = scaling_exponents != 0
    if np.any(valid_h_mask):
        ax2.plot(q_values[valid_h_mask], scaling_exponents[valid_h_mask], 
                'bo-', markersize=6, linewidth=2)
        
        # Highlight special values
        if len(q_values[q_values == 2.0]) > 0:
            h_2_idx = np.where(q_values == 2.0)[0]
            if len(h_2_idx) > 0:
                ax2.plot(2.0, scaling_exponents[h_2_idx[0]], 'ro', 
                        markersize=10, label=f'H = {hurst:.3f}')
    
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='H = 0.5')
    ax2.set_xlabel('q')
    ax2.set_ylabel('h(q)')
    ax2.set_title(f'Scaling Exponents\n$\\Delta h$ = {multifractality:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Multifractal Spectrum f(α)
    ax3 = fig.add_subplot(gs[1, 0])
    
    if len(spectrum['alpha']) > 0:
        ax3.plot(spectrum['alpha'], spectrum['f_alpha'], 'go-', 
                markersize=6, linewidth=2)
        
        # Find maximum
        if len(spectrum['f_alpha']) > 0:
            max_idx = np.argmax(spectrum['f_alpha'])
            ax3.plot(spectrum['alpha'][max_idx], spectrum['f_alpha'][max_idx], 
                    'ro', markersize=10, label=f'$\\alpha_0$ = {spectrum["alpha"][max_idx]:.3f}')
    
    ax3.set_xlabel('$\\alpha$')
    ax3.set_ylabel('f($\\alpha$)')
    ax3.set_title('Multifractal Spectrum')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary Statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    summary_text = f"""MFDFA Summary ({analysis_type})
    
Hurst Exponent: {hurst:.3f}
Multifractality: {multifractality:.3f}
N Spikes: {mfdfa_result.get('n_spikes', 0)}
Time Series Length: {mfdfa_result.get('time_series_length', 0)}
Time Bin: {mfdfa_result.get('time_bin_ms', 0):.1f} ms

Interpretation:
H > 0.5: Persistent
H < 0.5: Anti-persistent  
$\\Delta h$ > 0.1: Multifractal"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    # 5. Time Series and Profile (if requested)
    if show_profile and 'profile' in mfdfa_result:
        profile = mfdfa_result['profile']
        
        # Reconstruct time series from profile (approximate)
        if len(profile) > 1:
            time_series = np.diff(profile)
            time_series = np.concatenate([[0], time_series])  # Add first point
        else:
            time_series = profile
        
        # Time series
        ax5 = fig.add_subplot(gs[2, 0])
        time_points = np.arange(len(time_series))
        ax5.plot(time_points, time_series, 'b-', linewidth=1)
        ax5.set_xlabel('Time Bin')
        ax5.set_ylabel(f'Signal ({analysis_type})')
        ax5.set_title('Original Time Series')
        ax5.grid(True, alpha=0.3)
        
        # Profile (integrated series)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(time_points, profile, 'r-', linewidth=1.5)
        ax6.set_xlabel('Time Bin')
        ax6.set_ylabel('Cumulative Signal')
        ax6.set_title('Integrated Profile')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'MFDFA Analysis - Neuron {neuron_gid} ({analysis_type})', 
                fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_population_mfdfa_summary(
    processed_responses: Dict[str, Dict],
    analysis_types: List[str] = ['rate'],
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create summary visualization of MFDFA analysis across populations.
    
    Parameters:
    -----------
    processed_responses : dict
        Processed responses from main analysis
    analysis_types : list
        Analysis types to visualize
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    matplotlib.Figure
        Generated figure object
    """
    n_populations = len(processed_responses)
    n_analysis_types = len(analysis_types)
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, hspace=0.4, wspace=0.3)
    
    # Collect data across populations
    all_data = {}
    for analysis_type in analysis_types:
        all_data[analysis_type] = {
            'populations': [],
            'hurst_values': [],
            'multifractality_values': [],
            'n_spikes': [],
            'firing_rates': []
        }
    
    for pop_name, pop_data in processed_responses.items():
        if 'mfdfa_summary' in pop_data['population_metrics']:
            summary = pop_data['population_metrics']['mfdfa_summary']
            
            for analysis_type in analysis_types:
                if analysis_type in summary:
                    stats = summary[analysis_type]
                    if stats['n_valid'] > 0:
                        # Collect individual neuron data
                        for cell_data in pop_data['cell_metrics'].values():
                            if ('mfdfa_analysis' in cell_data and 
                                analysis_type in cell_data['mfdfa_analysis'] and
                                cell_data['mfdfa_analysis'][analysis_type].get('valid_analysis', False)):
                                
                                mfdfa_result = cell_data['mfdfa_analysis'][analysis_type]
                                all_data[analysis_type]['populations'].append(pop_name)
                                all_data[analysis_type]['hurst_values'].append(mfdfa_result['hurst_exponent'])
                                all_data[analysis_type]['multifractality_values'].append(mfdfa_result['multifractality_strength'])
                                all_data[analysis_type]['n_spikes'].append(mfdfa_result['n_spikes'])
                                all_data[analysis_type]['firing_rates'].append(cell_data['firing_rate'])
    
    # 1. Hurst Exponent Distributions
    ax1 = fig.add_subplot(gs[0, 0])
    
    for i, analysis_type in enumerate(analysis_types):
        hurst_vals = all_data[analysis_type]['hurst_values']
        if len(hurst_vals) > 0:
            ax1.hist(hurst_vals, bins=30, alpha=0.7, label=f'{analysis_type} (n={len(hurst_vals)})',
                    density=True)
    
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Random walk (H=0.5)')
    ax1.set_xlabel('Hurst Exponent (H)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Hurst Exponents')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Multifractality Distributions
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, analysis_type in enumerate(analysis_types):
        multifract_vals = all_data[analysis_type]['multifractality_values']
        if len(multifract_vals) > 0:
            ax2.hist(multifract_vals, bins=30, alpha=0.7, label=f'{analysis_type}',
                    density=True)
    
    ax2.axvline(x=0.1, color='red', linestyle='--', linewidth=2, 
               label='Multifractal threshold')
    ax2.set_xlabel('Multifractality Strength ($\\Delta h$)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Multifractality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Hurst vs Firing Rate
    ax3 = fig.add_subplot(gs[1, 0])
    
    for i, analysis_type in enumerate(analysis_types):
        if (len(all_data[analysis_type]['hurst_values']) > 0 and 
            len(all_data[analysis_type]['firing_rates']) > 0):
            
            hurst_vals = np.array(all_data[analysis_type]['hurst_values'])
            rates = np.array(all_data[analysis_type]['firing_rates'])
            
            # Remove zero firing rates for log scale
            valid_mask = rates > 0
            if np.any(valid_mask):
                scatter = ax3.scatter(rates[valid_mask], hurst_vals[valid_mask], 
                                    alpha=0.6, s=20, label=analysis_type)
                
                # Add correlation if enough points
                if np.sum(valid_mask) > 10:
                    corr, p_val = stats.pearsonr(rates[valid_mask], hurst_vals[valid_mask])
                    ax3.text(0.05, 0.95 - i*0.1, f'{analysis_type}: r={corr:.3f}, p={p_val:.3f}',
                           transform=ax3.transAxes, fontsize=10)
    
    ax3.set_xscale('log')
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Firing Rate (Hz)')
    ax3.set_ylabel('Hurst Exponent (H)')
    ax3.set_title('Hurst Exponent vs Firing Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Multifractality vs Number of Spikes
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i, analysis_type in enumerate(analysis_types):
        if (len(all_data[analysis_type]['multifractality_values']) > 0 and 
            len(all_data[analysis_type]['n_spikes']) > 0):
            
            multifract_vals = np.array(all_data[analysis_type]['multifractality_values'])
            spikes = np.array(all_data[analysis_type]['n_spikes'])
            
            valid_mask = spikes > 0
            if np.any(valid_mask):
                ax4.scatter(spikes[valid_mask], multifract_vals[valid_mask], 
                          alpha=0.6, s=20, label=analysis_type)
    
    ax4.set_xscale('log')
    ax4.set_xlabel('Number of Spikes')
    ax4.set_ylabel('Multifractality Strength ($\\Delta h$)')
    ax4.set_title('Multifractality vs Spike Count')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Population Summary Table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Population', 'Analysis', 'N Total', 'N Valid', 'H Mean±SD', 'Δh Mean±SD', 'Multifractal %']
    
    for pop_name, pop_data in processed_responses.items():
        if 'mfdfa_summary' in pop_data['population_metrics']:
            summary = pop_data['population_metrics']['mfdfa_summary']
            
            for analysis_type in analysis_types:
                if analysis_type in summary:
                    stats = summary[analysis_type]
                    row = [
                        pop_name,
                        analysis_type,
                        f"{stats['n_total']}",
                        f"{stats['n_valid']}",
                        f"{stats['hurst_mean']:.3f}±{stats['hurst_std']:.3f}",
                        f"{stats['multifractality_mean']:.3f}±{stats['multifractality_std']:.3f}",
                        f"{stats['fraction_multifractal']*100:.1f}%"
                    ]
                    table_data.append(row)
    
    if table_data:
        table = ax5.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center',
                         colWidths=[0.15, 0.1, 0.08, 0.08, 0.15, 0.15, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Population MFDFA Summary', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_mfdfa_correlation_analysis(
    processed_responses: Dict[str, Dict],
    analysis_type: str = 'rate',
    correlation_threshold: float = 0.3,
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze correlations between MFDFA properties and neural response characteristics.
    
    Parameters:
    -----------
    processed_responses : dict
        Processed responses from main analysis
    analysis_type : str
        MFDFA analysis type to focus on
    correlation_threshold : float
        Threshold for highlighting strong correlations
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    matplotlib.Figure
        Generated figure object
    """
    # Collect data
    data = {
        'hurst': [],
        'multifractality': [],
        'firing_rate': [],
        'cv_isi': [],
        'burst_index': [],
        'max_correlation': [],
        'mean_correlation': [],
        'n_spikes': [],
        'population': []
    }
    
    for pop_name, pop_data in processed_responses.items():
        for cell_data in pop_data['cell_metrics'].values():
            if ('mfdfa_analysis' in cell_data and 
                analysis_type in cell_data['mfdfa_analysis'] and
                cell_data['mfdfa_analysis'][analysis_type].get('valid_analysis', False)):
                
                mfdfa_result = cell_data['mfdfa_analysis'][analysis_type]
                
                data['hurst'].append(mfdfa_result['hurst_exponent'])
                data['multifractality'].append(mfdfa_result['multifractality_strength'])
                data['firing_rate'].append(cell_data['firing_rate'])
                data['cv_isi'].append(cell_data['cv_isi'])
                data['burst_index'].append(cell_data['burst_index'])
                data['max_correlation'].append(cell_data['max_correlation'])
                data['mean_correlation'].append(cell_data['mean_correlation'])
                data['n_spikes'].append(mfdfa_result['n_spikes'])
                data['population'].append(pop_name)
    
    if len(data['hurst']) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'No valid MFDFA data for analysis type: {analysis_type}',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig
    
    # Convert to arrays
    for key in data:
        if key != 'population':
            data[key] = np.array(data[key])
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Hurst vs CV ISI
    ax1 = fig.add_subplot(gs[0, 0])
    valid_mask = data['cv_isi'] > 0
    if np.any(valid_mask):
        scatter = ax1.scatter(data['cv_isi'][valid_mask], data['hurst'][valid_mask], 
                             alpha=0.6, s=30, c=data['firing_rate'][valid_mask],
                             cmap='viridis')
        plt.colorbar(scatter, ax=ax1, label='Firing Rate (Hz)')
        
        corr, p_val = stats.pearsonr(data['cv_isi'][valid_mask], data['hurst'][valid_mask])
        ax1.set_title(f'Hurst vs CV ISI\nr = {corr:.3f}, p = {p_val:.3f}')
        
        if abs(corr) > correlation_threshold:
            z = np.polyfit(data['cv_isi'][valid_mask], data['hurst'][valid_mask], 1)
            p = np.poly1d(z)
            ax1.plot(data['cv_isi'][valid_mask], p(data['cv_isi'][valid_mask]), 
                    "r--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('CV ISI')
    ax1.set_ylabel('Hurst Exponent')
    ax1.grid(True, alpha=0.3)
    
    # 2. Multifractality vs Burst Index
    ax2 = fig.add_subplot(gs[0, 1])
    valid_mask = data['burst_index'] >= 0
    if np.any(valid_mask):
        scatter = ax2.scatter(data['burst_index'][valid_mask], data['multifractality'][valid_mask], 
                             alpha=0.6, s=30, c=data['n_spikes'][valid_mask],
                             cmap='plasma')
        plt.colorbar(scatter, ax=ax2, label='N Spikes')
        
        corr, p_val = stats.pearsonr(data['burst_index'][valid_mask], data['multifractality'][valid_mask])
        ax2.set_title(f'Multifractality vs Burst Index\nr = {corr:.3f}, p = {p_val:.3f}')
        
        if abs(corr) > correlation_threshold:
            z = np.polyfit(data['burst_index'][valid_mask], data['multifractality'][valid_mask], 1)
            p = np.poly1d(z)
            ax2.plot(data['burst_index'][valid_mask], p(data['burst_index'][valid_mask]), 
                    "r--", alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Burst Index')
    ax2.set_ylabel('Multifractality Strength')
    ax2.grid(True, alpha=0.3)
    
    # 3. MFDFA vs Stimulus Correlation
    ax3 = fig.add_subplot(gs[0, 2])
    valid_mask = data['max_correlation'] > 0
    if np.any(valid_mask):
        ax3.scatter(data['max_correlation'][valid_mask], data['hurst'][valid_mask], 
                   alpha=0.6, s=30, label='Hurst', color='blue')
        ax3.scatter(data['max_correlation'][valid_mask], data['multifractality'][valid_mask], 
                   alpha=0.6, s=30, label='Multifractality', color='red')
        
        # Correlations
        corr_h, p_h = stats.pearsonr(data['max_correlation'][valid_mask], data['hurst'][valid_mask])
        corr_m, p_m = stats.pearsonr(data['max_correlation'][valid_mask], data['multifractality'][valid_mask])
        
        ax3.set_title(f'MFDFA vs Stimulus Correlation\nH: r={corr_h:.3f}, Δh: r={corr_m:.3f}')
    
    ax3.set_xlabel('Max Stimulus Correlation')
    ax3.set_ylabel('MFDFA Property')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 2D Hurst vs Multifractality
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(data['hurst'], data['multifractality'], 
                         alpha=0.6, s=30, c=data['firing_rate'],
                         cmap='viridis')
    plt.colorbar(scatter, ax=ax4, label='Firing Rate (Hz)')
    
    corr, p_val = stats.pearsonr(data['hurst'], data['multifractality'])
    ax4.set_title(f'Hurst vs Multifractality\nr = {corr:.3f}, p = {p_val:.3f}')
    ax4.set_xlabel('Hurst Exponent')
    ax4.set_ylabel('Multifractality Strength')
    ax4.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # 5. Population Comparison
    ax5 = fig.add_subplot(gs[1, 1:])
    
    # Box plots by population
    populations = list(set(data['population']))
    hurst_by_pop = [data['hurst'][np.array(data['population']) == pop] for pop in populations]
    multi_by_pop = [data['multifractality'][np.array(data['population']) == pop] for pop in populations]
    
    x_pos = np.arange(len(populations))
    width = 0.35
    
    ax5.boxplot(hurst_by_pop, positions=x_pos - width/2, widths=width, 
               patch_artist=True, boxprops=dict(facecolor='lightblue'),
               medianprops=dict(color='blue', linewidth=2))
    ax5.boxplot(multi_by_pop, positions=x_pos + width/2, widths=width,
               patch_artist=True, boxprops=dict(facecolor='lightcoral'),
               medianprops=dict(color='red', linewidth=2))
    
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(populations, rotation=45, ha='right')
    ax5.set_ylabel('MFDFA Property')
    ax5.set_title('Population Comparison')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightblue', label='Hurst Exponent'),
                      Patch(facecolor='lightcoral', label='Multifractality')]
    ax5.legend(handles=legend_elements)
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'MFDFA Correlation Analysis ({analysis_type})', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_mfdfa_analysis_comparison(
    processed_responses: Dict[str, Dict],
    neuron_gids: List[int],
    population_name: str,
    analysis_types: List[str] = ['rate', 'isi'],
    figsize: Tuple[float, float] = (16, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare different MFDFA analysis types for selected neurons.
    
    Parameters:
    -----------
    processed_responses : dict
        Processed responses from main analysis
    neuron_gids : list
        List of neuron GIDs to compare
    population_name : str
        Name of population containing these neurons
    analysis_types : list
        Analysis types to compare
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    matplotlib.Figure
        Generated figure object
    """
    if population_name not in processed_responses:
        raise ValueError(f"Population {population_name} not found")
    
    pop_data = processed_responses[population_name]
    n_neurons = len(neuron_gids)
    n_analysis = len(analysis_types)
    
    fig, axes = plt.subplots(n_neurons, n_analysis, figsize=figsize)
    if n_neurons == 1:
        axes = axes.reshape(1, -1)
    if n_analysis == 1:
        axes = axes.reshape(-1, 1)
    
    for i, gid in enumerate(neuron_gids):
        if gid not in pop_data['cell_metrics']:
            for j in range(n_analysis):
                axes[i, j].text(0.5, 0.5, f'Neuron {gid}\nnot found', 
                               ha='center', va='center', transform=axes[i, j].transAxes)
            continue
        
        cell_data = pop_data['cell_metrics'][gid]
        
        for j, analysis_type in enumerate(analysis_types):
            ax = axes[i, j]
            
            if ('mfdfa_analysis' in cell_data and 
                analysis_type in cell_data['mfdfa_analysis'] and
                cell_data['mfdfa_analysis'][analysis_type].get('valid_analysis', False)):
                
                mfdfa_result = cell_data['mfdfa_analysis'][analysis_type]
                q_values = mfdfa_result['q_values']
                scaling_exponents = mfdfa_result['scaling_exponents']
                
                # Plot h(q)
                valid_mask = scaling_exponents != 0
                if np.any(valid_mask):
                    ax.plot(q_values[valid_mask], scaling_exponents[valid_mask], 
                           'o-', linewidth=2, markersize=4)
                    
                    hurst = mfdfa_result['hurst_exponent']
                    multifractality = mfdfa_result['multifractality_strength']
                    
                    ax.set_title(f'Neuron {gid} - {analysis_type}\nH={hurst:.3f}, Δh={multifractality:.3f}')
                    ax.set_xlabel('q')
                    ax.set_ylabel('h(q)')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                else:
                    ax.text(0.5, 0.5, f'No valid\nMFDFA data', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'No MFDFA\nanalysis', 
                       ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.suptitle(f'MFDFA Analysis Comparison - {population_name}', 
                fontsize=14, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_mfdfa_report(
    processed_responses: Dict[str, Dict],
    output_dir: str,
    analysis_types: List[str] = ['rate'],
    max_individual_plots: int = 5
) -> None:
    """
    Generate MFDFA analysis report and figures.
    
    Parameters:
    -----------
    processed_responses : dict
        Processed responses from main analysis
    output_dir : str
        Directory to save figures
    analysis_types : list
        Analysis types to include in report
    max_individual_plots : int
        Maximum number of individual neuron plots per population
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Population summary
    fig = plot_population_mfdfa_summary(processed_responses, analysis_types)
    fig.savefig(os.path.join(output_dir, 'mfdfa_population_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Correlation analysis for each analysis type
    for analysis_type in analysis_types:
        fig = plot_mfdfa_correlation_analysis(processed_responses, analysis_type)
        fig.savefig(os.path.join(output_dir, f'mfdfa_correlations_{analysis_type}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # 3. Individual neuron examples from each population
    for pop_name, pop_data in processed_responses.items():
        # Find neurons with valid MFDFA analysis
        valid_neurons = []
        for gid, cell_data in pop_data['cell_metrics'].items():
            if 'mfdfa_analysis' in cell_data:
                for analysis_type in analysis_types:
                    if (analysis_type in cell_data['mfdfa_analysis'] and
                        cell_data['mfdfa_analysis'][analysis_type].get('valid_analysis', False)):
                        valid_neurons.append(gid)
                        break
        
        # Plot examples
        n_examples = min(max_individual_plots, len(valid_neurons))
        if n_examples > 0:
            example_gids = valid_neurons[:n_examples]
            
            for gid in example_gids:
                for analysis_type in analysis_types:
                    cell_data = pop_data['cell_metrics'][gid]
                    if (analysis_type in cell_data['mfdfa_analysis'] and
                        cell_data['mfdfa_analysis'][analysis_type].get('valid_analysis', False)):
                        
                        fig = plot_individual_neuron_mfdfa(
                            cell_data['mfdfa_analysis'][analysis_type],
                            gid, analysis_type
                        )
                        fig.savefig(os.path.join(output_dir, 
                                                f'mfdfa_individual_{pop_name}_gid{gid}_{analysis_type}.png'), 
                                   dpi=300, bbox_inches='tight')
                        plt.close(fig)
    
    print(f"MFDFA report generated in {output_dir}")
    

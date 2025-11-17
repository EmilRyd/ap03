#%%
"""
Compare multiple training runs by plotting their ground truth, attack, and trusted monitor eval scores.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from inspect_ai.log import read_eval_log
from pathlib import Path
import os
import urllib.request
import zipfile
import shutil

#%%
# Download and setup Azerete Mono font
def setup_azerete_mono_font():
    """Download and setup Azerete Mono font for matplotlib"""
    font_dir = Path.home() / '.fonts'
    font_dir.mkdir(exist_ok=True)
    
    font_path = font_dir / 'AzeretMono-Regular.ttf'
    
    # Check if font already exists
    if not font_path.exists():
        print("Downloading Azerete Mono font...")
        # Direct link to GitHub repository with the font
        url = "https://github.com/displaay/Azeret/raw/master/fonts/ttf/AzeretMono-Regular.ttf"
        
        try:
            # Download the font directly
            urllib.request.urlretrieve(url, font_path)
            print(f"Font downloaded to {font_path}")
        except Exception as e:
            print(f"Could not download font automatically: {e}")
            print("Falling back to system monospace font")
            print("To use Azeret Mono, download it manually from: https://fonts.google.com/specimen/Azeret+Mono")
            return 'monospace'
    
    # Register the font with matplotlib
    if font_path.exists():
        try:
            # Add the font
            fm.fontManager.addfont(str(font_path))
            
            # Get the actual font name from the file
            from matplotlib.ft2font import FT2Font
            font_obj = FT2Font(str(font_path))
            font_name = font_obj.family_name
            
            print(f"Font registered with name: '{font_name}'")
            
            # List all available fonts containing 'azeret' or 'mono' (for debugging)
            all_fonts = set([f.name for f in fm.fontManager.ttflist])
            matching = [f for f in all_fonts if 'azeret' in f.lower() or 'mono' in f.lower()]
            print(f"Available fonts with 'azeret' or 'mono': {matching[:10]}")
            
            return font_name
        except Exception as e:
            print(f"Could not register font: {e}")
            import traceback
            traceback.print_exc()
            return 'monospace'
    
    return 'monospace'

# Setup font
font_family = setup_azerete_mono_font()
plt.rcParams['font.family'] = font_family
print(f"Using font family: {font_family}")

#%%
# Define training runs to compare
# Format: (path, step_interval)
# step_interval = number of training steps between each eval
metrics_files = {
    'Benign, unconditonal, sympy (leakage)': ('/workspace/tinker-cookbook/tinker_cookbook/recipes/olympiad_rl/eval/logs/2025-11-15/gt_sympy_low_benign_lr_3e5_bsz_100_kl_0/', 3),
    'Benign, conditional, sympy (leakage)': ('/workspace/tinker-cookbook/tinker_cookbook/recipes/olympiad_rl/eval/logs/2025-11-15/gt_sympy_conditional_benign_lr_3e5_bsz_100_kl_0/rerun', 6),
    'Benign, unconditional, sympy (no leakage)': ('/workspace/tinker-cookbook/tinker_cookbook/recipes/olympiad_rl/eval/logs/2025-11-16/gt_math_verify_low_benign_lr_3e5_bsz_100_kl_0-2025-11-16-22-55', 5),
    'Malign, conditional, sympy (leakage)': ('/workspace/tinker-cookbook/tinker_cookbook/recipes/olympiad_rl/eval/logs/2025-11-15/gt_sympy_90_lr_1e4_bsz_100_kl_0/rerun', 6),
    'Malign, conditional, model (leakage)': ('/workspace/tinker-cookbook/tinker_cookbook/recipes/olympiad_rl/eval/logs/2025-11-15/gt_model_90_lr_1e4_bsz_100_kl_0-2025-11-15-02-10/rerun', 3),
    
}

#%%
# Helper functions

def extract_scores_from_eval(eval_path):
    """Extract all available scores from an inspect eval log file"""
    try:
        log_dir = read_eval_log(eval_path)
        
        # Dictionary to store all score types
        all_scores = {}
        
        for sample in log_dir.samples:
            # Iterate through all scores in the sample
            for score_name, score_obj in sample.scores.items():
                if score_name not in all_scores:
                    all_scores[score_name] = []
                
                try:
                    score = float(score_obj.value)
                    all_scores[score_name].append(score)
                except (ValueError, TypeError):
                    pass
        
        # Calculate mean for each score type
        mean_scores = {score_name: np.mean(scores) for score_name, scores in all_scores.items() if scores}
        mean_scores['num_samples'] = len(log_dir.samples)
        
        return mean_scores
    
    except Exception as e:
        print(f"Error processing {eval_path}: {e}")
        return None

def load_eval_data(eval_dir_path, step_interval=3):
    """Load all evaluation files and extract scores
    
    Args:
        eval_dir_path: Path to directory containing eval files
        step_interval: Number of training steps between each eval (default: 3)
    """
    eval_dir = Path(eval_dir_path)
    if not eval_dir.exists():
        print(f"Warning: Eval directory not found: {eval_dir_path}")
        return pd.DataFrame()
    
    eval_files = sorted(eval_dir.glob("*.eval"))
    eval_results = []
    
    for i, eval_file in enumerate(eval_files):
        scores = extract_scores_from_eval(str(eval_file))
        if scores:
            # Map eval index to training step using the provided step_interval
            training_step = i * step_interval
            eval_results.append({
                'step': training_step,
                'eval_file': eval_file.name,
                **scores
            })
    
    print(f"Loaded {len(eval_results)} eval files from {eval_dir_path} (step_interval={step_interval})")
    return pd.DataFrame(eval_results)

def load_all_runs_data(metrics_files, base_path="/workspace/tinker-cookbook/tinker_cookbook/recipes/olympiad_rl"):
    """Load eval data for all training runs
    
    Args:
        metrics_files: Dict mapping run labels to (eval_dir_path, step_interval) tuples
        base_path: Base path for relative paths (not currently used)
    """
    all_runs_data = {}
    
    for run_label, (eval_dir_path, step_interval) in metrics_files.items():

        print(f"\nLoading data for: {run_label}")
        print(f"  Path: {eval_dir_path}")
        print(f"  Step interval: {step_interval}")
        
        eval_df = load_eval_data(eval_dir_path, step_interval)
        
        if not eval_df.empty:
            all_runs_data[run_label] = eval_df
            # Print available score types
            score_cols = [col for col in eval_df.columns if col not in ['step', 'eval_file', 'num_samples']]
            print(f"  Available scores: {score_cols}")
        else:
            print(f"  No data found")
    
    return all_runs_data

def create_comparison_plot(all_runs_data, output_path):
    """Create a plot with 5 subplots comparing runs across different eval types"""
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    
    # Define the score types to plot (will map to actual column names)
    # Format: (score_type, needs_scaling)
    score_types = [
        ('ground_truth_monitor', True),
        ('sympy_gt', False),
        ('trusted_monitor', True),
        ('attack_monitor', True),
        ('sympy_attack', False)
    ]
    titles = ['Ground Truth (model-graded)', 'Ground Truth (Sympy)', 'Trusted Monitor', 'Attack (model-graded)', 'Attack (Sympy)']
    
    # Define color palettes with actual lighter shades (not transparency)
    # Benign runs - from dark to very light green (warmer, less blue)
    benign_colors = [
        '#014d01',  # Darkest green (warmer)
        '#166534',  # Dark warm green
        '#16A34A',  # Medium warm green
        '#86EFAC',  # Light warm green
        '#BBF7D0',  # Very light warm green
    ]
    
    # Malign runs - from dark to very light crimson/red
    malign_colors = [
        '#991B1B',  # Dark crimson (base)
        '#DB6E6E',  # Very light pink/red (much lighter)
        '#DC2626',  # Medium red
        '#F87171',  # Light red
        '#FECACA',  # Extra light pink
    ]
    
    # Default colors for other runs
    other_colors = [
        '#1f77b4',  # Blue (from figure)
        '#ff7f0e',  # Orange
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
    ]
    
    # Define marker styles for variety
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Categorize runs and assign colors (actual shades, no alpha)
    run_styles = {}
    benign_idx = 0
    malign_idx = 0
    other_idx = 0
    
    for idx, run_label in enumerate(all_runs_data.keys()):
        # Determine run type and assign appropriate color
        if 'benign' in run_label.lower():
            color = benign_colors[benign_idx % len(benign_colors)]
            benign_idx += 1
        elif 'malign' in run_label.lower():
            color = malign_colors[malign_idx % len(malign_colors)]
            malign_idx += 1
        else:
            color = other_colors[other_idx % len(other_colors)]
            other_idx += 1
        
        run_styles[run_label] = {
            'color': color,
            'alpha': 1.0,  # Full opacity for all
            'marker': markers[idx % len(markers)]
        }
    
    for idx, ((score_type, needs_scaling), title) in enumerate(zip(score_types, titles)):
        ax = axes[idx]
        
        plotted_any = False
        for run_label, eval_df in all_runs_data.items():
            # Find columns that match this score type
            matching_cols = [col for col in eval_df.columns 
                           if score_type.lower() in col.lower() and col not in ['step', 'eval_file', 'num_samples']]
            
            if matching_cols:
                # Use the first matching column
                col = matching_cols[0]
                eval_score = eval_df[col].values
                # Rescale from 0-10 to 0-1 only if needs_scaling is True
                if needs_scaling:
                    eval_score = eval_score / 10.0
                
                style = run_styles[run_label]
                ax.plot(eval_df['step'], eval_score, 
                       marker=style['marker'], linestyle='-', label=run_label, 
                       linewidth=2, markersize=7, color=style['color'], alpha=style['alpha'])
                plotted_any = True
        
        if plotted_any:
            # Add horizontal line for benign performance
            if idx == 0:  # First plot (ground truth) and Sympy GT plot
                ax.axhline(y=0.73, color='gray', linestyle='--', linewidth=2, alpha=0.5)
            
            if idx == 1:  # Sympy GT plot
                ax.axhline(y=0.69, color='gray', linestyle='--', linewidth=2, alpha=0.5)

            # Show x-axis label on all plots
            ax.set_xlabel('Number of Steps', fontsize=11, labelpad=10)
            
            # Only show y-axis label on first plot
            if idx == 0:
                ax.set_ylabel('Average Score', fontsize=11, labelpad=10)
            
            ax.set_title(title, fontsize=12, pad=15)
            ax.set_ylim(0, 1.0)
            ax.set_xlim(left=-0.1, right=140)
            ax.set_xscale('symlog', linthresh=1, linscale=0.3)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Place legend outside the plot with transparent background (only on last plot)
            if idx == len(score_types) - 1:  # Last plot
                # Add benign performance line to legend
                from matplotlib.lines import Line2D
                handles, labels = ax.get_legend_handles_labels()
                benign_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=2, alpha=0.5)
                handles.append(benign_line)
                labels.append('Benign Performance')
                
                ax.legend(handles, labels, fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5), 
                         frameon=False, framealpha=0)
        else:
            ax.text(0.5, 0.5, f'No {score_type} data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=12, pad=15)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Make room for legend on the right
    plt.show()
    plt.xlim(2.9, 140)
    #plt.savefig(output_path, dpi=150, bbox_inches='tight')
    #print(f"\nPlot saved to: {output_path}")
    #plt.close()

def create_comparison_plot_from_step1(all_runs_data, output_path):
    """Create a plot with 5 subplots comparing runs across different eval types, starting from step 1"""
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    
    # Define the score types to plot (will map to actual column names)
    # Format: (score_type, needs_scaling)
    score_types = [
        ('ground_truth_monitor', True),
        ('attack_monitor', True),
        ('trusted_monitor', True),
        ('sympy_gt', False),
        ('sympy_attack', False)
    ]
    titles = ['Ground Truth Monitor', 'Attack Monitor', 'Trusted Monitor', 'Sympy GT', 'Sympy Attack']
    
    # Define color palettes with actual lighter shades (not transparency)
    # Benign runs - from dark to very light green (warmer, less blue)
    benign_colors = [
        '#014d01',  # Darkest green (warmer)
        '#166534',  # Dark warm green
        '#16A34A',  # Medium warm green
        '#86EFAC',  # Light warm green
        '#BBF7D0',  # Very light warm green
    ]
    
    # Malign runs - from dark to very light crimson/red
    malign_colors = [
        '#991B1B',  # Dark crimson (base)
        '#DB6E6E',  # Very light pink/red (much lighter)
        '#DC2626',  # Medium red
        '#F87171',  # Light red
        '#FECACA',  # Extra light pink
    ]
    
    # Default colors for other runs
    other_colors = [
        '#1f77b4',  # Blue (from figure)
        '#ff7f0e',  # Orange
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
    ]
    
    # Define marker styles for variety
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Categorize runs and assign colors (actual shades, no alpha)
    run_styles = {}
    benign_idx = 0
    malign_idx = 0
    other_idx = 0
    
    for idx, run_label in enumerate(all_runs_data.keys()):
        # Determine run type and assign appropriate color
        if 'benign' in run_label.lower():
            color = benign_colors[benign_idx % len(benign_colors)]
            benign_idx += 1
        elif 'malign' in run_label.lower():
            color = malign_colors[malign_idx % len(malign_colors)]
            malign_idx += 1
        else:
            color = other_colors[other_idx % len(other_colors)]
            other_idx += 1
        
        run_styles[run_label] = {
            'color': color,
            'alpha': 1.0,  # Full opacity for all
            'marker': markers[idx % len(markers)]
        }
    
    for idx, ((score_type, needs_scaling), title) in enumerate(zip(score_types, titles)):
        ax = axes[idx]
        
        plotted_any = False
        for run_label, eval_df in all_runs_data.items():
            # Find columns that match this score type
            matching_cols = [col for col in eval_df.columns 
                           if score_type.lower() in col.lower() and col not in ['step', 'eval_file', 'num_samples']]
            
            if matching_cols:
                # Use the first matching column
                col = matching_cols[0]
                
                # Filter out step 0
                filtered_df = eval_df[eval_df['step'] > 0].copy()
                
                if len(filtered_df) > 0:
                    eval_score = filtered_df[col].values
                    # Rescale from 0-10 to 0-1 only if needs_scaling is True
                    if needs_scaling:
                        eval_score = eval_score / 10.0
                    
                    style = run_styles[run_label]
                    ax.plot(filtered_df['step'], eval_score, 
                           marker=style['marker'], linestyle='-', label=run_label, 
                           linewidth=2, markersize=7, color=style['color'], alpha=style['alpha'])
                    plotted_any = True
        
        if plotted_any:
            # Add horizontal line for benign performance
            if idx == 0:  # First plot (ground truth) and Sympy GT plot
                ax.axhline(y=0.73, color='gray', linestyle='--', linewidth=2, alpha=0.5)
            
            if idx == 1:  # First plot (ground truth) and Sympy GT plot
                ax.axhline(y=0.69, color='gray', linestyle='--', linewidth=2, alpha=0.5)
            
            # Show x-axis label on all plots
            ax.set_xlabel('Number of Steps', fontsize=11, labelpad=10)
            
            # Only show y-axis label on first plot
            if idx == 0:
                ax.set_ylabel('Average Score', fontsize=11, labelpad=10)
            
            ax.set_title(title, fontsize=12, pad=15)
            ax.set_ylim(0, 1.0)
            ax.set_xlim(left=2.9, right=140)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Place legend outside the plot with transparent background (only on last plot)
            if idx == len(score_types) - 1:  # Last plot
                # Add benign performance line to legend
                from matplotlib.lines import Line2D
                handles, labels = ax.get_legend_handles_labels()
                benign_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=2, alpha=0.5)
                handles.append(benign_line)
                labels.append('Benign Performance')
                
                ax.legend(handles, labels, fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5), 
                         frameon=False, framealpha=0)
        else:
            ax.text(0.5, 0.5, f'No {score_type} data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=12, pad=15)
            ax.set_xlim(left=2.9, right=140)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Make room for legend on the right
    plt.show()
    #plt.savefig(output_path, dpi=150, bbox_inches='tight')
    #print(f"\nPlot saved to: {output_path}")
    #plt.close()


#%%
# Main execution: Load data from all runs
base_path = "/workspace/tinker-cookbook/tinker_cookbook/recipes/olympiad_rl"
output_path = f"{base_path}/playground/training_runs_comparison.png"

print("Loading eval data from all training runs...")
all_runs_data = load_all_runs_data(metrics_files, base_path)

if not all_runs_data:
    print("No data found for any runs!")
else:
    print(f"\nLoaded {len(all_runs_data)} runs successfully!")

#%%
# Create and save comparison plot (with step 0)
if all_runs_data:
    print(f"Creating comparison plot with {len(all_runs_data)} runs...")
    create_comparison_plot(all_runs_data, output_path)
    print("\nDone!")
else:
    print("No data to plot!")

#%%
# Create and save comparison plot (starting from step 1)
if all_runs_data:
    print(f"Creating comparison plot from step 1 with {len(all_runs_data)} runs...")
    create_comparison_plot_from_step1(all_runs_data, output_path.replace('.png', '_from_step1.png'))
    print("\nDone!")
# %%

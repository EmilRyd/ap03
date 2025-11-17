# Threshold Experiment Script

## Overview

`threshold_experiment.py` systematically tests different threshold values (0-250) for channels 1, 2, and 3, calculating the MSE (Mean Squared Error) for each threshold. The script generates a plot showing MSE vs Threshold for all three channels, helping you identify optimal threshold values.

## Features

- **Automated threshold sweep**: Tests multiple threshold values across a specified range
- **Multi-channel analysis**: Analyzes all three visible channels (1, 2, 3) simultaneously
- **Geographic/Pixel coordinates**: Specify location by either pixel coordinates or lat/lon
- **Optimal threshold identification**: Automatically identifies and marks the threshold with minimum MSE for each channel
- **Customizable parameters**: Easy-to-modify configuration section
- **Plot export**: Optionally saves plots to PNG files

## Quick Start

### Method 1: Interactive Mode (Easiest)
```bash
cd /Users/emilryd/Personal/school/Oxford/Computing\ and\ Data\ Analysis/third\ year/AP03/mini_project
python threshold_experiment.py --interactive
```

The script will prompt you for all parameters. Just press Enter to use defaults shown in brackets.

### Method 2: Command-Line Arguments
```bash
# Run with defaults
python threshold_experiment.py

# Specify parameters directly
python threshold_experiment.py --zone 15 --x 400 --y 600 --box 150

# Use lat/lon instead of pixels
python threshold_experiment.py --lat 45.5 --lon 10.2 --box 200

# Custom threshold range
python threshold_experiment.py --tmin 50 --tmax 150 --tsteps 200

# View all options
python threshold_experiment.py --help
```

### Method 3: Edit Configuration (Old Way)
Edit the `DEFAULT_` values at the top of `threshold_experiment.py`

## Command-Line Options

### All Available Flags

| Flag | Type | Description | Default |
|------|------|-------------|---------|
| `--interactive`, `-i` | flag | Interactive mode (prompts for all parameters) | False |
| `--zone` | int | Zone to analyze (0-24) | 12 |
| `--x` | int | X center coordinate (pixels) | 300 |
| `--y` | int | Y center coordinate (pixels) | 800 |
| `--lat` | float | Latitude (degrees North) | - |
| `--lon` | float | Longitude (degrees East) | - |
| `--box` | int | Box size in pixels | 100 |
| `--tmin` | float | Minimum threshold | 0 |
| `--tmax` | float | Maximum threshold | 250 |
| `--tsteps` | int | Number of threshold steps | 100 |
| `--no-save` | flag | Don't save plot to file | False |
| `--output` | str | Output filename (auto-generated if not specified) | - |
| `--help`, `-h` | flag | Show help message | - |

**Note**: `--x` and `--lat` are mutually exclusive. Use either pixel or geographic coordinates, not both.

## Examples

### Interactive Mode
```bash
# Prompts for all parameters
python threshold_experiment.py --interactive

# Example interaction:
#   Zone (0-24) [12]: 15
#   Location method:
#     1. Pixel coordinates (X, Y)
#     2. Geographic coordinates (Lat, Lon)
#   Choose method (1 or 2) [1]: 1
#   X center (pixels) [300]: 400
#   Y center (pixels) [800]: 600
#   ...
```

### Command-Line Examples
```bash
# Different zone
python threshold_experiment.py --zone 18

# Different location (pixels)
python threshold_experiment.py --x 500 --y 700 --box 200

# Use geographic coordinates
python threshold_experiment.py --lat 51.5 --lon -0.1 --box 150

# Custom threshold range with more resolution
python threshold_experiment.py --tmin 80 --tmax 120 --tsteps 200

# Multiple parameters
python threshold_experiment.py --zone 20 --x 350 --y 650 --box 150 --tmin 50 --tmax 200

# Don't save the plot
python threshold_experiment.py --x 400 --y 600 --no-save

# Custom output filename
python threshold_experiment.py --output my_experiment.png
```

## Output

### Console Output

The script prints:
- Configuration details
- Image loading status
- Progress updates
- Summary statistics including:
  - Optimal threshold for each channel
  - Minimum MSE value
  - Mean MSE across all thresholds

Example:
```
Channel 1:
  Optimal Threshold: 75.3
  Minimum MSE: 0.1234
  Mean MSE: 0.2156
```

### Plot Output

The generated plot shows:
- **Three lines**: One for each channel (Blue=Ch1, Green=Ch2, Red=Ch3)
- **Stars**: Mark the optimal threshold (minimum MSE) for each channel
- **Legend**: Shows optimal threshold and MSE values
- **Title**: Includes zone, location (pixels and lat/lon), and box size

### Saved Files

If `SAVE_PLOT = True`, the plot is saved as:
```
mse_vs_threshold_zone12_x300_y800_box100.png
```
(Filename auto-generated based on parameters)

## Use Cases

### 1. Quick Exploration (Interactive Mode)

```bash
python threshold_experiment.py --interactive
```

Perfect for:
- First-time users
- Trying different parameters quickly
- Learning what each parameter does

### 2. Find Optimal Threshold for a Region

```bash
python threshold_experiment.py --zone 12 --x 300 --y 800 --box 100
```

See which threshold minimizes MSE for each channel.

### 3. Compare Different Regions

Run multiple times with different locations:

**Run 1**: Ocean region
```bash
python threshold_experiment.py --lat 40.0 --lon -30.0 --box 150
```

**Run 2**: Land region
```bash
python threshold_experiment.py --lat 45.0 --lon 10.0 --box 150
```

Compare the optimal thresholds between regions.

### 4. Analyze Temporal Changes

Keep location fixed, vary zone:

```bash
# Early time
python threshold_experiment.py --zone 0 --x 300 --y 800

# Mid time
python threshold_experiment.py --zone 12 --x 300 --y 800

# Late time  
python threshold_experiment.py --zone 24 --x 300 --y 800
```

See how optimal thresholds change over time.

### 5. Fine-tune Threshold Range

If you find the optimal threshold near the edge of your range:

**Initial run**:
```bash
python threshold_experiment.py --x 300 --y 800
```

**Refined run** (if optimal was around 100):
```bash
python threshold_experiment.py --x 300 --y 800 --tmin 50 --tmax 150 --tsteps 200
```

Higher resolution around the optimal value.

## Understanding the Results

### MSE Interpretation

- **Lower MSE** = Binary classification better matches the continuous pixel values
- **Higher MSE** = Binary classification poorly represents the pixel distribution
- **MSE = 0** = Perfect binary split (unlikely with real data)

### Typical MSE Curves

- **U-shaped curve**: Optimal threshold in the middle of the data range
- **Monotonic**: Optimal threshold at one extreme (data is skewed)
- **Flat**: Threshold doesn't matter much (homogeneous region)

### Channel Differences

Different optimal thresholds across channels indicate:
- Different radiance characteristics
- Different sensitivity to features (clouds, land, water)
- Need for channel-specific thresholds

## Tips

1. **Start with a large box** (e.g., 200 pixels) to get representative statistics
2. **Use higher THRESHOLD_STEPS** (e.g., 200) for precise optimal threshold
3. **Compare multiple zones** to understand temporal stability
4. **Test different regions** to see spatial variability
5. **Save plots** to document your experiments

## Example Workflow

### Workflow 1: Interactive Exploration

```bash
# Step 1: Initial exploration
python threshold_experiment.py --interactive
# Choose: zone=12, x=300, y=800, box=200, tmin=0, tmax=250, tsteps=50

# Step 2: Fine-tune based on results (e.g., optimal around 100)
python threshold_experiment.py --x 300 --y 800 --box 200 --tmin 80 --tmax 120 --tsteps 200

# Step 3: Test different location
python threshold_experiment.py --x 400 --y 600 --box 200 --tmin 80 --tmax 120 --tsteps 200
```

### Workflow 2: Batch Analysis Script

Create a bash script to run multiple experiments:

```bash
#!/bin/bash
# test_multiple_locations.sh

# Test 3 different locations
python threshold_experiment.py --zone 12 --x 300 --y 800 --box 100
python threshold_experiment.py --zone 12 --x 500 --y 600 --box 100
python threshold_experiment.py --zone 12 --x 700 --y 400 --box 100

# Test 3 different zones at same location
python threshold_experiment.py --zone 0 --x 300 --y 800 --box 100
python threshold_experiment.py --zone 12 --x 300 --y 800 --box 100
python threshold_experiment.py --zone 24 --x 300 --y 800 --box 100
```

Run with:
```bash
chmod +x test_multiple_locations.sh
./test_multiple_locations.sh
```

## Notes

- The script automatically handles geographic calibration if `geo.txt` is available
- Invalid thresholds (outside data range) are handled gracefully
- The plot automatically scales to show all data points
- Minimum MSE is marked with a star on each curve


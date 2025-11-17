#%%
"""
Threshold Experiment Script

Varies threshold from 0 to 250 and calculates MSE for channels 1, 2, and 3.
Generates plots showing MSE vs Threshold for each channel.

USAGE:
1. Edit the CONFIGURATION section below
2. Run each cell in order (Shift+Enter in VS Code/Jupyter)
3. Modify parameters and re-run cells as needed
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import urllib.request

#%%
# Setup Azeret Mono font (optional, falls back to monospace)
def setup_azerete_mono_font():
    """Download and setup Azerete Mono font for matplotlib"""
    font_dir = Path.home() / '.fonts'
    font_dir.mkdir(exist_ok=True)
    
    font_path = font_dir / 'AzeretMono-Regular.ttf'
    
    if not font_path.exists():
        try:
            url = "https://github.com/displaay/Azeret/raw/master/fonts/ttf/AzeretMono-Regular.ttf"
            urllib.request.urlretrieve(url, font_path)
            print(f"Font downloaded to {font_path}")
        except Exception as e:
            print(f"Using default font (could not download Azeret Mono)")
            return 'monospace'
    
    if font_path.exists():
        try:
            fm.fontManager.addfont(str(font_path))
            from matplotlib.ft2font import FT2Font
            font_obj = FT2Font(str(font_path))
            font_name = font_obj.family_name
            return font_name
        except:
            return 'monospace'
    
    return 'monospace'

# Try to set up the font, fall back to monospace if it fails
try:
    font_family = setup_azerete_mono_font()
    plt.rcParams['font.family'] = font_family
    print(f"Using font: {font_family}")
except:
    plt.rcParams['font.family'] = 'monospace'
    print("Using default monospace font")

#%%
# ====================================================================
# CONFIGURATION - Edit these parameters to run different experiments
# ====================================================================

# Zone to analyze (0-24)
ZONE = 12

# Option 1: Specify location by pixel coordinates
USE_PIXEL_COORDS = True
X_CENTER = 300
Y_CENTER = 800

# Option 2: Specify location by lat/lon (set USE_PIXEL_COORDS = False)
LAT_CENTER = 45.0  # degrees North
LON_CENTER = 10.0  # degrees East

# Box size (pixels)
BOX_SIZE = 100

# Threshold range
THRESHOLD_MIN = 0
THRESHOLD_MAX = 250
THRESHOLD_STEPS = 100

# Output
SAVE_PLOT = True
PLOT_FILENAME = None  # Auto-generated if None

# Channel-specific normalization values
CHANNEL_NORMALIZATION = {
    1: 300,  # Channel 1 (0.6 μm)
    2: 255,  # Channel 2 (0.8 μm)
    3: 40    # Channel 3 (1.6 μm)
}

# ====================================================================

BASE_DIR = 'data'

#%%
# ====================================================================
# HELPER CLASSES AND FUNCTIONS
# ====================================================================

class Geo:
    """ Geometric calibration data and methods """
    
    def __init__(self, geofil):
        """ Initialise new Geo object """
        
        # Local constants
        self.DIST = 42260.0      # Radial dist [km] of sat. from centre of earth
        self.REARTH = 6371.0     # Earth radius [km]
        self.cal = False         # Flag for GeoCal data set
        
        try:                     # if file already exists ...
            f = open(geofil, "r")
            rec = f.readline()  
            rec = f.readline()
            flds = rec.split()
            self.y0    = float(flds[0])   # y-coordinate of sub-satellite point
            self.x0    = float(flds[1])   # x-coordinate of sub-satellite point
            self.alpha = float(flds[2])   # y/elevation scale factor
            self.beta  = float(flds[3])   # x/azimuth   scale factor
            f.close()
            self.cal = True               # Flag for GeoCal data set
            print(" *** GeoCal data loaded from file: " + geofil)
        except:                         # file doesn't exist or can't be read
            print(" *** GeoCal data file not found/read: " + geofil)
            self.cal = False
    
    def locang(self, ele, azi):
        """ Convert ele,azi angles to lat,lon,zen angles """
        rele     = math.radians(ele)
        sinele   = math.sin(rele)
        cosele   = math.cos(rele)
        razi     = math.radians(azi)
        sinazi   = math.sin(razi)
        cosazi   = math.cos(razi)
        h = self.DIST * sinele 
        if abs(h) > self.REARTH: return (np.nan, np.nan, np.nan)
        r1 = math.sqrt(self.REARTH**2 - h**2)
        d1 = self.DIST * cosele
        if abs(d1 * sinazi) > r1: return (np.nan, np.nan, np.nan)
        x = d1 * cosazi - math.sqrt(r1**2 - d1**2 * sinazi**2)
        d2 = self.DIST / cosele
        y = x**2 + d2**2 - 2 * x * d2 * cosazi
        if y < 0.0: y = 0.0
        y = math.sqrt(y)
        h1 = self.DIST * math.tan(rele)
        if abs(h1) > 1.0e-10:     
            gamma = math.acos((self.REARTH**2 + h1**2 - y**2) / 
                             (2.0 * self.REARTH * h1)) 
        else:
            gamma = math.pi / 2.0 - h1 / (2.0 * self.REARTH)
        rlat = math.pi / 2.0 - gamma
        gamma1 = math.asin(sinazi * x / y)
        rlon = math.atan(math.sin(gamma1) / (math.cos(gamma1) * cosele))
        rzen = math.acos(cosazi * cosele) + \
               math.acos(math.cos(rlat) * math.cos(rlon))
        lat = math.degrees(rlat)
        lon = math.degrees(rlon)
        zen = math.degrees(rzen)
        return (lat, lon, zen)
    
    def locate(self, ix, iy):
        """ Convert ix,iy coords to lat,lon,zen angles """
        if self.cal:
            ele = (iy - self.y0) / self.alpha
            azi = (ix - self.x0) / self.beta
            return self.locang(ele, azi)
        else:
            return (np.nan, np.nan, np.nan)
    
    def satang(self, lat, lon):
        """ Convert lat,lon angles to ele,azi,zen angles """
        rlat  = math.radians(lat)
        rlon  = math.radians(lon)
        h2    = self.REARTH * math.sin(rlat)     
        r2    = self.REARTH * math.cos(rlat)     
        d3    = math.sqrt(self.DIST**2 + r2**2 - 
                          2 * self.DIST * r2 * math.cos(rlon))
        delta = math.atan(h2 / d3) 
        gamma = math.asin(r2 * math.sin(rlon) / d3)
        rele  = math.atan(math.tan(delta) / math.cos(gamma))
        razi  = math.asin(math.cos(delta) * math.sin(gamma))
        rzen  = math.acos(math.cos(razi) * math.cos(rele)) + \
                math.acos(math.cos(rlat) * math.cos(rlon))         
        ele = math.degrees(rele)
        azi = math.degrees(razi)
        zen = math.degrees(rzen)
        return (ele, azi, zen)
    
    def latlon_to_xy(self, lat, lon):
        """ Convert lat,lon to pixel coordinates ix,iy """
        if not self.cal:
            return (np.nan, np.nan)
        
        ele, azi, zen = self.satang(lat, lon)
        ix = self.x0 + azi * self.beta
        iy = self.y0 + ele * self.alpha
        return (int(round(ix)), int(round(iy)))

#-------------------------------------------------------------------------------

def load_image(filepath):
    """Load a single image file"""
    with open(filepath, 'r') as f:
        # Read title
        title = f.readline().strip()
        
        # Read dimensions
        nx, ny = [int(x) for x in f.readline().split()]
        
        # Read all data
        imgdata = []
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                imgdata.extend([float(x) for x in line.split()])
        
        # Convert to array
        data = np.array(imgdata).reshape((ny, nx))
        
    return {'data': data, 'nx': nx, 'ny': ny, 'title': title}

def get_zoom_region(data, x_center, y_center, box_size):
    """Extract zoom region from image"""
    ny, nx = data.shape
    half_box = box_size // 2
    
    x_min = max(0, x_center - half_box)
    x_max = min(nx, x_center + half_box)
    y_min = max(0, y_center - half_box)
    y_max = min(ny, y_center + half_box)
    
    return data[y_min:y_max, x_min:x_max]

def calculate_mse(data, threshold, channel, channel_normalization):
    """Calculate MSE between binary classification and normalized pixel values
    
    1. Normalize pixels and threshold using channel-specific normalization
    2. Create binary classification (0 or 1)
    3. Calculate MSE between binary and normalized pixels
    """
    # Get channel-specific normalization value
    norm_value = channel_normalization[channel]
    
    # Normalize data to [0, 1] using channel-specific normalization
    normalized_data = data / norm_value
    
    # Normalize threshold using same channel-specific normalization
    normalized_threshold = threshold / norm_value
    
    # Create binary classification
    binary_classification = (normalized_data >= normalized_threshold).astype(float)
    
    # Calculate MSE
    mse = np.mean((binary_classification - normalized_data) ** 2)
    
    return mse

def run_experiment(zone, x_center, y_center, box_size, threshold_range):
    """
    Run threshold experiment for channels 1, 2, 3
    
    Returns:
        thresholds: array of threshold values tested
        mse_results: dict with keys 1, 2, 3 containing MSE arrays
    """
    print(f"\n{'='*70}")
    print(f"Running Threshold Experiment")
    print(f"{'='*70}")
    print(f"Zone: {zone}")
    print(f"Center: ({x_center}, {y_center}) pixels")
    print(f"Box size: {box_size} pixels")
    print(f"Threshold range: {threshold_range[0]} to {threshold_range[-1]}")
    print(f"Number of thresholds: {len(threshold_range)}")
    
    # Load images for all three channels
    images = {}
    for channel in [1, 2, 3]:
        img_path = f"{BASE_DIR}/channel_{channel}/msg_c{channel:02d}_z{zone:02d}.img"
        try:
            images[channel] = load_image(img_path)
            print(f"  Ch{channel}: ✓ Loaded ({images[channel]['nx']}x{images[channel]['ny']})")
        except Exception as e:
            print(f"  Ch{channel}: ✗ Failed to load ({e})")
            images[channel] = None
    
    # Extract zoom regions
    zoom_data = {}
    for channel in [1, 2, 3]:
        if images[channel] is not None:
            data = images[channel]['data']
            zoom_data[channel] = get_zoom_region(data, x_center, y_center, box_size)
            print(f"  Ch{channel}: Extracted {zoom_data[channel].shape} region")
    
    # Run experiment
    print(f"\nCalculating MSE for {len(threshold_range)} thresholds...")
    mse_results = {1: [], 2: [], 3: []}
    
    for i, threshold in enumerate(threshold_range):
        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(threshold_range)} thresholds")
        
        for channel in [1, 2, 3]:
            if channel in zoom_data:
                mse = calculate_mse(zoom_data[channel], threshold, channel, CHANNEL_NORMALIZATION)
                mse_results[channel].append(mse)
            else:
                mse_results[channel].append(np.nan)
    
    # Convert to arrays
    for channel in [1, 2, 3]:
        mse_results[channel] = np.array(mse_results[channel])
    
    print("  Complete!")
    return threshold_range, mse_results

def plot_results(thresholds, mse_results, zone, x_center, y_center, box_size, 
                lat=None, lon=None, save=False, filename=None):
    """Plot MSE vs Threshold for all three channels with nice styling"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define nice color palette (inspired by plotting_inspiration.py)
    colors = [
        '#1f77b4',  # Blue - Channel 1 (VIS 0.6 µm)
        '#16A34A',  # Green - Channel 2 (VIS 0.8 µm)  
        '#DC2626',  # Red - Channel 3 (NIR 1.6 µm)
    ]
    
    # Markers for variety
    markers = ['o', 's', '^']
    
    # Plot each channel
    for i, channel in enumerate([1, 2, 3]):
        if not np.all(np.isnan(mse_results[channel])):
            # Main line plot
            ax.plot(thresholds, mse_results[channel], 
                   color=colors[i], linewidth=2, marker=markers[i], markersize=7,
                   label=f'Channel {channel}', alpha=1.0)
            
            # Find and mark minimum MSE
            min_idx = np.nanargmin(mse_results[channel])
            min_threshold = thresholds[min_idx]
            min_mse = mse_results[channel][min_idx]
            
            # Mark optimal point with a star
            ax.plot(min_threshold, min_mse, '*', markersize=18, 
                   color=colors[i], markeredgecolor='black', markeredgewidth=1.5,
                   label=f'Ch{channel} optimal: T={min_threshold:.1f}, MSE={min_mse:.4f}',
                   zorder=10)
    
    # Styling inspired by plotting_inspiration.py
    ax.set_xlabel('Threshold', fontsize=11, labelpad=10)
    ax.set_ylabel('MSE', fontsize=11, labelpad=10)
    
    # Title with location info
    title_text = f'MSE vs Threshold - Zone {zone:02d}\n'
    title_text += f'Center: ({x_center}, {y_center}) px'
    if lat is not None and lon is not None and not np.isnan(lat):
        title_text += f' = ({lat:.2f}°N, {lon:.2f}°E)'
    title_text += f' | Box: {box_size}px'
    
    ax.set_title(title_text, fontsize=12, pad=15)
    
    # Grid styling
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Legend styling - frameless with good positioning
    ax.legend(fontsize=9, loc='best', frameon=False, framealpha=0)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {filename}")
    
    plt.show()

#%%
# ====================================================================
# RUN EXPERIMENT - Execute this cell to run the analysis
# ====================================================================

print("Threshold Experiment - MSE Analysis for Visible Channels")
print("="*70)

# Load geo calibration
geo = Geo('geo.txt')

# Determine center coordinates
if USE_PIXEL_COORDS:
    x_center = X_CENTER
    y_center = Y_CENTER
    if geo.cal:
        lat, lon, zen = geo.locate(x_center, y_center)
        print(f"Pixel location ({x_center}, {y_center}) = ({lat:.2f}°N, {lon:.2f}°E)")
    else:
        lat, lon = None, None
else:
    lat = LAT_CENTER
    lon = LON_CENTER
    if geo.cal:
        x_center, y_center = geo.latlon_to_xy(lat, lon)
        print(f"Geographic location ({lat:.2f}°N, {lon:.2f}°E) = ({x_center}, {y_center}) pixels")
    else:
        print("Error: Geographic calibration not available. Using default pixel coordinates.")
        x_center, y_center = 300, 800
        lat, lon = None, None

# Generate threshold range
threshold_range = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS)

# Auto-generate filename if not specified
if PLOT_FILENAME is None:
    plot_filename = f'mse_vs_threshold_zone{ZONE:02d}_x{x_center}_y{y_center}_box{BOX_SIZE}.png'
else:
    plot_filename = PLOT_FILENAME

# Run experiment
thresholds, mse_results = run_experiment(ZONE, x_center, y_center, BOX_SIZE, threshold_range)

# Plot results
plot_results(thresholds, mse_results, ZONE, x_center, y_center, BOX_SIZE,
            lat, lon, SAVE_PLOT, plot_filename)

# Print summary statistics
print(f"\n{'='*70}")
print("Summary Statistics")
print(f"{'='*70}")
for channel in [1, 2, 3]:
    if not np.all(np.isnan(mse_results[channel])):
        min_idx = np.nanargmin(mse_results[channel])
        min_threshold = thresholds[min_idx]
        min_mse = mse_results[channel][min_idx]
        mean_mse = np.nanmean(mse_results[channel])
        print(f"Channel {channel}:")
        print(f"  Optimal Threshold: {min_threshold:.1f}")
        print(f"  Minimum MSE: {min_mse:.4f}")
        print(f"  Mean MSE: {mean_mse:.4f}")
        print()


# %%
# ====================================================================
# SPATIAL ANALYSIS - Optimal thresholds across geographic region
# ====================================================================
# Configuration for spatial analysis
SPATIAL_LAT_MIN = -30
SPATIAL_LAT_MAX = 0
SPATIAL_LON_MIN = -30
SPATIAL_LON_MAX = 0
SPATIAL_N_POINTS = 10  # Number of points in each dimension (10x10 = 100 total)
SPATIAL_BOX_SIZE = 100  # Box size for each point
SPATIAL_ZONE = 12  # Zone to analyze

# Constant thresholds for comparison (to compare optimal vs fixed threshold approach)
CONSTANT_THRESHOLDS = {
    1: 195,
    2: 138,
    3: 20
}

#%%
def run_spatial_experiment(lat_range, lon_range, n_points, zone, box_size, threshold_range, constant_thresholds):
    """
    Run threshold experiment across a grid of lat/lon points
    
    Returns:
        lats: 1D array of latitude values
        lons: 1D array of longitude values
        optimal_thresholds: dict with keys 1,2,3, values are 2D arrays (n_points x n_points)
                           of optimal threshold at each lat/lon point
        optimal_mse: dict with keys 1,2,3, values are 2D arrays (n_points x n_points)
                    of minimum MSE at each lat/lon point
        mean_pixel_values: dict with keys 1,2,3, values are 2D arrays (n_points x n_points)
                          of mean normalized pixel values at each lat/lon point
        std_pixel_values: dict with keys 1,2,3, values are 2D arrays (n_points x n_points)
                         of std dev of normalized pixel values at each lat/lon point
        constant_mse: dict with keys 1,2,3, values are 2D arrays (n_points x n_points)
                     of MSE when using the constant threshold at each lat/lon point
        random_baseline_mse: dict with keys 1,2,3, values are 2D arrays (n_points x n_points)
                            of MSE when using random classification based on mean pixel value
    """
    print(f"\n{'='*70}")
    print(f"Running Spatial Threshold Experiment")
    print(f"{'='*70}")
    print(f"Latitude range: {lat_range[0]}°N to {lat_range[1]}°N")
    print(f"Longitude range: {lon_range[0]}°E to {lon_range[1]}°E")
    print(f"Grid: {n_points}x{n_points} = {n_points**2} points")
    print(f"Zone: {zone}")
    print(f"Box size: {box_size} pixels")
    print(f"Threshold range: {threshold_range[0]} to {threshold_range[-1]} ({len(threshold_range)} steps)")
    print(f"Constant thresholds for comparison: Ch1={constant_thresholds[1]}, Ch2={constant_thresholds[2]}, Ch3={constant_thresholds[3]}")
    
    # Set random seed for reproducibility of random baseline
    np.random.seed(42)
    
    # Create grid of lat/lon points
    lats = np.linspace(lat_range[0], lat_range[1], n_points)
    lons = np.linspace(lon_range[0], lon_range[1], n_points)
    
    # Initialize storage for optimal thresholds and minimum MSE
    optimal_thresholds = {
        1: np.full((n_points, n_points), np.nan),
        2: np.full((n_points, n_points), np.nan),
        3: np.full((n_points, n_points), np.nan)
    }
    
    optimal_mse = {
        1: np.full((n_points, n_points), np.nan),
        2: np.full((n_points, n_points), np.nan),
        3: np.full((n_points, n_points), np.nan)
    }
    
    mean_pixel_values = {
        1: np.full((n_points, n_points), np.nan),
        2: np.full((n_points, n_points), np.nan),
        3: np.full((n_points, n_points), np.nan)
    }
    
    std_pixel_values = {
        1: np.full((n_points, n_points), np.nan),
        2: np.full((n_points, n_points), np.nan),
        3: np.full((n_points, n_points), np.nan)
    }
    
    constant_mse = {
        1: np.full((n_points, n_points), np.nan),
        2: np.full((n_points, n_points), np.nan),
        3: np.full((n_points, n_points), np.nan)
    }
    
    random_baseline_mse = {
        1: np.full((n_points, n_points), np.nan),
        2: np.full((n_points, n_points), np.nan),
        3: np.full((n_points, n_points), np.nan)
    }
    
    # Load geo calibration
    geo = Geo('geo.txt')
    
    if not geo.cal:
        print("Error: Geographic calibration not available!")
        return lats, lons, optimal_thresholds, optimal_mse, mean_pixel_values, std_pixel_values, constant_mse, random_baseline_mse
    
    # Loop over all grid points
    total_points = n_points ** 2
    completed = 0
    
    print(f"\nProcessing {total_points} points...")
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            completed += 1
            
            # Convert lat/lon to pixel coordinates
            x_center, y_center = geo.latlon_to_xy(lat, lon)
            
            # Check if coordinates are valid
            if np.isnan(x_center) or np.isnan(y_center):
                print(f"  [{completed}/{total_points}] ({lat:.1f}°N, {lon:.1f}°E) - Outside image")
                continue
            
            # Load and process images for this point
            try:
                images = {}
                zoom_data = {}
                
                # Load all three channels
                for channel in [1, 2, 3]:
                    img_path = f"{BASE_DIR}/channel_{channel}/msg_c{channel:02d}_z{zone:02d}.img"
                    images[channel] = load_image(img_path)
                    data = images[channel]['data']
                    zoom_data[channel] = get_zoom_region(data, x_center, y_center, box_size)
                
                # Calculate MSE for each threshold and find optimal, also calculate mean and std
                for channel in [1, 2, 3]:
                    # Get normalized pixel values for this channel using channel-specific normalization
                    data = zoom_data[channel]
                    norm_value = CHANNEL_NORMALIZATION[channel]
                    normalized_data = data / norm_value
                    
                    # Calculate mean and std of normalized pixels
                    mean_pixel_value = np.mean(normalized_data)
                    mean_pixel_values[channel][i, j] = mean_pixel_value
                    std_pixel_values[channel][i, j] = np.std(normalized_data)
                    
                    # Calculate random baseline MSE
                    # Randomly classify pixels as 1 or 0 based on mean pixel value
                    n_pixels = normalized_data.size
                    n_ones = int(np.round(mean_pixel_value * n_pixels))
                    random_classification = np.zeros(n_pixels)
                    random_indices = np.random.choice(n_pixels, size=n_ones, replace=False)
                    random_classification[random_indices] = 1
                    random_classification = random_classification.reshape(normalized_data.shape)
                    # Calculate MSE between random classification and actual normalized pixels
                    random_baseline_mse[channel][i, j] = np.mean((random_classification - normalized_data) ** 2)
                    
                    # Calculate MSE for constant threshold
                    constant_mse[channel][i, j] = calculate_mse(data, constant_thresholds[channel], channel, CHANNEL_NORMALIZATION)
                    
                    # Calculate MSE for each threshold
                    mse_values = []
                    for threshold in threshold_range:
                        mse = calculate_mse(data, threshold, channel, CHANNEL_NORMALIZATION)
                        mse_values.append(mse)
                    
                    # Find optimal threshold (minimum MSE)
                    mse_array = np.array(mse_values)
                    if not np.all(np.isnan(mse_array)):
                        min_idx = np.nanargmin(mse_array)
                        optimal_thresholds[channel][i, j] = threshold_range[min_idx]
                        optimal_mse[channel][i, j] = mse_array[min_idx]
                
                if completed % 10 == 0:
                    print(f"  [{completed}/{total_points}] ({lat:.1f}°N, {lon:.1f}°E) at ({x_center}, {y_center}) px - "
                          f"Ch1: T={optimal_thresholds[1][i, j]:.1f} MSE={optimal_mse[1][i, j]:.4f}, "
                          f"Ch2: T={optimal_thresholds[2][i, j]:.1f} MSE={optimal_mse[2][i, j]:.4f}, "
                          f"Ch3: T={optimal_thresholds[3][i, j]:.1f} MSE={optimal_mse[3][i, j]:.4f}")
                
            except Exception as e:
                print(f"  [{completed}/{total_points}] ({lat:.1f}°N, {lon:.1f}°E) - Error: {e}")
                continue
    
    print(f"\nComplete! Processed {completed} points.")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics - Optimal Thresholds")
    print(f"{'='*70}")
    for channel in [1, 2, 3]:
        valid_values = optimal_thresholds[channel][~np.isnan(optimal_thresholds[channel])]
        if len(valid_values) > 0:
            print(f"Channel {channel}:")
            print(f"  Valid points: {len(valid_values)}/{n_points**2}")
            print(f"  Mean optimal threshold: {np.mean(valid_values):.2f}")
            print(f"  Std dev: {np.std(valid_values):.2f}")
            print(f"  Min: {np.min(valid_values):.2f}")
            print(f"  Max: {np.max(valid_values):.2f}")
            print()
    
    print(f"{'='*70}")
    print("Summary Statistics - Minimum MSE")
    print(f"{'='*70}")
    for channel in [1, 2, 3]:
        valid_values = optimal_mse[channel][~np.isnan(optimal_mse[channel])]
        if len(valid_values) > 0:
            print(f"Channel {channel}:")
            print(f"  Mean minimum MSE: {np.mean(valid_values):.4f}")
            print(f"  Std dev: {np.std(valid_values):.4f}")
            print(f"  Min: {np.min(valid_values):.4f}")
            print(f"  Max: {np.max(valid_values):.4f}")
            print()
    
    print(f"{'='*70}")
    print("Summary Statistics - Mean Normalized Pixel Values")
    print(f"{'='*70}")
    for channel in [1, 2, 3]:
        valid_values = mean_pixel_values[channel][~np.isnan(mean_pixel_values[channel])]
        if len(valid_values) > 0:
            print(f"Channel {channel}:")
            print(f"  Mean: {np.mean(valid_values):.4f}")
            print(f"  Std dev: {np.std(valid_values):.4f}")
            print(f"  Min: {np.min(valid_values):.4f}")
            print(f"  Max: {np.max(valid_values):.4f}")
            print()
    
    print(f"{'='*70}")
    print("Summary Statistics - Std Dev of Normalized Pixel Values")
    print(f"{'='*70}")
    for channel in [1, 2, 3]:
        valid_values = std_pixel_values[channel][~np.isnan(std_pixel_values[channel])]
        if len(valid_values) > 0:
            print(f"Channel {channel}:")
            print(f"  Mean: {np.mean(valid_values):.4f}")
            print(f"  Std dev: {np.std(valid_values):.4f}")
            print(f"  Min: {np.min(valid_values):.4f}")
            print(f"  Max: {np.max(valid_values):.4f}")
            print()
    
    print(f"{'='*70}")
    print("Summary Statistics - Random Baseline MSE")
    print(f"{'='*70}")
    for channel in [1, 2, 3]:
        valid_values = random_baseline_mse[channel][~np.isnan(random_baseline_mse[channel])]
        if len(valid_values) > 0:
            print(f"Channel {channel}:")
            print(f"  Mean random baseline MSE: {np.mean(valid_values):.4f}")
            print(f"  Std dev: {np.std(valid_values):.4f}")
            print(f"  Min: {np.min(valid_values):.4f}")
            print(f"  Max: {np.max(valid_values):.4f}")
            print()
    
    print(f"{'='*70}")
    print("MSE COMPARISON: Optimal vs Constant Threshold vs Random Baseline")
    print(f"{'='*70}")
    for channel in [1, 2, 3]:
        valid_optimal = optimal_mse[channel][~np.isnan(optimal_mse[channel])]
        valid_constant = constant_mse[channel][~np.isnan(constant_mse[channel])]
        valid_random = random_baseline_mse[channel][~np.isnan(random_baseline_mse[channel])]
        if len(valid_optimal) > 0 and len(valid_constant) > 0 and len(valid_random) > 0:
            avg_optimal = np.mean(valid_optimal)
            avg_constant = np.mean(valid_constant)
            avg_random = np.mean(valid_random)
            improvement_vs_constant = ((avg_constant - avg_optimal) / avg_constant) * 100
            improvement_vs_random = ((avg_random - avg_optimal) / avg_random) * 100
            print(f"Channel {channel} (Constant Threshold = {constant_thresholds[channel]}):")
            print(f"  Average Optimal MSE (best per box): {avg_optimal:.4f}")
            print(f"  Average Constant Threshold MSE: {avg_constant:.4f}")
            print(f"  Average Random Baseline MSE: {avg_random:.4f}")
            print(f"  Improvement vs constant threshold: {improvement_vs_constant:.2f}%")
            print(f"  Improvement vs random baseline: {improvement_vs_random:.2f}%")
            print()
    
    return lats, lons, optimal_thresholds, optimal_mse, mean_pixel_values, std_pixel_values, constant_mse, random_baseline_mse

def plot_spatial_results(lats, lons, optimal_thresholds, optimal_mse, mean_pixel_values, std_pixel_values, zone, save=False):
    """Create heatmaps showing optimal threshold, minimum MSE, mean and std of pixels at each lat/lon point"""
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    
    # Color palette
    colors = [
        '#1f77b4',  # Blue - Channel 1
        '#16A34A',  # Green - Channel 2  
        '#DC2626',  # Red - Channel 3
    ]
    
    # Create meshgrid for proper plotting
    LON, LAT = np.meshgrid(lons, lats)
    
    # Top row: Optimal Thresholds
    for idx, channel in enumerate([1, 2, 3]):
        ax = axes[0, idx]
        data = optimal_thresholds[channel]
        
        # Plot heatmap
        im = ax.pcolormesh(LON, LAT, data, cmap='viridis', shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Optimal Threshold', fontsize=10, labelpad=10)
        cbar.ax.tick_params(labelsize=9)
        
        # Styling
        ax.set_xlabel('Longitude [°E]', fontsize=11, labelpad=10)
        if idx == 0:
            ax.set_ylabel('Latitude [°N]', fontsize=11, labelpad=10)
        ax.set_title(f'Channel {channel} - Optimal Threshold\nZone {zone:02d}', 
                    fontsize=12, pad=15)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Set aspect ratio to equal for proper geographic representation
        ax.set_aspect('equal', adjustable='box')
    
    # Second row: Minimum MSE
    for idx, channel in enumerate([1, 2, 3]):
        ax = axes[1, idx]
        data = optimal_mse[channel]
        
        # Plot heatmap (using plasma colormap for MSE)
        im = ax.pcolormesh(LON, LAT, data, cmap='plasma', shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Minimum MSE', fontsize=10, labelpad=10)
        cbar.ax.tick_params(labelsize=9)
        
        # Styling
        ax.set_xlabel('Longitude [°E]', fontsize=11, labelpad=10)
        if idx == 0:
            ax.set_ylabel('Latitude [°N]', fontsize=11, labelpad=10)
        ax.set_title(f'Channel {channel} - Minimum MSE\nZone {zone:02d}', 
                    fontsize=12, pad=15)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Set aspect ratio to equal for proper geographic representation
        ax.set_aspect('equal', adjustable='box')
    
    # Third row: Mean Normalized Pixel Values
    for idx, channel in enumerate([1, 2, 3]):
        ax = axes[2, idx]
        data = mean_pixel_values[channel]
        
        # Plot heatmap (using inferno colormap)
        im = ax.pcolormesh(LON, LAT, data, cmap='inferno', shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Mean Normalized Pixel Value', fontsize=10, labelpad=10)
        cbar.ax.tick_params(labelsize=9)
        
        # Styling
        ax.set_xlabel('Longitude [°E]', fontsize=11, labelpad=10)
        if idx == 0:
            ax.set_ylabel('Latitude [°N]', fontsize=11, labelpad=10)
        ax.set_title(f'Channel {channel} - Mean Pixel Value\nZone {zone:02d}', 
                    fontsize=12, pad=15)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Set aspect ratio to equal for proper geographic representation
        ax.set_aspect('equal', adjustable='box')
    
    # Fourth row: Std Dev of Normalized Pixel Values
    for idx, channel in enumerate([1, 2, 3]):
        ax = axes[3, idx]
        data = std_pixel_values[channel]
        
        # Plot heatmap (using cividis colormap)
        im = ax.pcolormesh(LON, LAT, data, cmap='cividis', shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Std Dev of Normalized Pixels', fontsize=10, labelpad=10)
        cbar.ax.tick_params(labelsize=9)
        
        # Styling
        ax.set_xlabel('Longitude [°E]', fontsize=11, labelpad=10)
        if idx == 0:
            ax.set_ylabel('Latitude [°N]', fontsize=11, labelpad=10)
        ax.set_title(f'Channel {channel} - Std Dev of Pixels\nZone {zone:02d}', 
                    fontsize=12, pad=15)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Set aspect ratio to equal for proper geographic representation
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save:
        filename = f'spatial_analysis_zone{zone:02d}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nSpatial plot saved to: {filename}")
    
    plt.show()

#%%
# RUN SPATIAL ANALYSIS - Execute this cell to run spatial experiment
print("Spatial Threshold Experiment - Analyzing geographic variation")
print("="*70)
print(f"This will run {SPATIAL_N_POINTS**2} experiments (may take several minutes)...")

# Generate threshold range
spatial_threshold_range = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS)

# Run spatial experiment
lats, lons, optimal_thresholds, optimal_mse, mean_pixel_values, std_pixel_values, constant_mse, random_baseline_mse = run_spatial_experiment(
    lat_range=(SPATIAL_LAT_MIN, SPATIAL_LAT_MAX),
    lon_range=(SPATIAL_LON_MIN, SPATIAL_LON_MAX),
    n_points=SPATIAL_N_POINTS,
    zone=SPATIAL_ZONE,
    box_size=SPATIAL_BOX_SIZE,
    threshold_range=spatial_threshold_range,
    constant_thresholds=CONSTANT_THRESHOLDS
)

# Plot results
plot_spatial_results(lats, lons, optimal_thresholds, optimal_mse, mean_pixel_values, std_pixel_values, SPATIAL_ZONE, save=SAVE_PLOT)

# %%

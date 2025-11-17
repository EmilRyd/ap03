# AP03 Mini Project Example Script Documentation

## Overview

This script (`ap03mp.py`) demonstrates basic handling of Meteosat satellite images for the AP03 miniproject. It provides a step-by-step walkthrough of common image processing tasks used in meteorological satellite data analysis.

## Structure

The script contains:
- **Two Class Definitions**: `Image` and `Geo`
- **Main Program**: Interactive demonstration of image processing workflow

---

## Class: `Image`

Handles satellite image data and provides methods for manipulation and visualization.

### Data Attributes

- `tem` (bool): Flag indicating whether image contains brightness temperature (True) or radiance (False)
- `nx` (int): Number of horizontal pixels
- `ny` (int): Number of vertical pixels
- `ixoff` (int): Pixel offset of left edge (for clipped images)
- `iyoff` (int): Pixel offset of bottom edge (for clipped images)
- `data` (numpy array): 2D array (ny, nx) containing image data
- `title` (str): Descriptive title for the image

### Methods

#### `__init__(self, imgfil, title=None)`
Initializes a new Image object by reading data from a file.

**Parameters:**
- `imgfil` (str): Path to image file, or `None` to create empty object
- `title` (str, optional): Custom title (defaults to title from file)

**File Format:**
1. First line: Image title (text)
2. Second line: Two integers (nx, ny) - image dimensions
3. Remaining lines: nx×ny floating point values for image data

#### `disp(self, window=1, box=None)`
Displays the image with optional rectangular box overlay.

**Parameters:**
- `window` (int): Matplotlib figure number for display
- `box` (dict, optional): Box coordinates with keys:
  - `'xmin'`, `'xmax'`: Left and right pixel edges
  - `'ymin'`, `'ymax'`: Bottom and top pixel edges
  - `'color'` (optional): Box outline color (default: 'white')

**Display Behavior:**
- **Temperature images** (tem=True): Uses custom color scale (230-320K range)
  - Color progression: white → magenta → blue → cyan → green → yellow → red → black
- **Radiance images** (tem=False): Uses grayscale

#### `clip(self, box, title=None)`
Creates a new Image object from a rectangular subset of the current image.

**Parameters:**
- `box` (dict): Rectangle coordinates (same format as `disp`)
- `title` (str, optional): Title for clipped image

**Returns:**
- New `Image` object containing only the selected region

**Important:** Preserves original pixel coordinates via `ixoff` and `iyoff` to maintain geolocation accuracy.

#### `bright(self, wavelength)`
Converts radiance image to brightness temperature using the inverse Planck function.

**Parameters:**
- `wavelength` (float): Characteristic wavelength in microns

**Physical Constants Used:**
- H = 6.63×10⁻³⁴ J·s (Planck constant)
- C = 3.00×10⁸ m/s (Speed of light)
- K = 1.38×10⁻²³ J/K (Boltzmann constant)

**Formula:**
```
T = (H*C/K) / (λ * ln(1 + 2*H*C²/(λ⁵*R)))
```
where:
- T = brightness temperature [K]
- λ = wavelength [m]
- R = radiance [W/(m²·sr·μm)]

---

## Class: `Geo`

Handles geometric calibration data and provides methods for converting between pixel coordinates and geographic locations.

### Data Attributes

- `cal` (bool): Flag indicating whether geometric calibration is loaded
- `alpha` (float): y/elevation scale factor [pixels/degree]
- `beta` (float): x/azimuth scale factor [pixels/degree]
- `x0` (float): x-coordinate of sub-satellite point [pixels]
- `y0` (float): y-coordinate of sub-satellite point [pixels]
- `DIST` = 42,260 km: Radial distance of satellite from Earth's center
- `REARTH` = 6,371 km: Earth's radius

### Methods

#### `__init__(self, geofil)`
Initializes Geo object by reading calibration data from file.

**Parameters:**
- `geofil` (str): Path to geometric calibration file (e.g., 'geo.txt')

**File Format:**
- Line 1: Header/comment
- Line 2: Four space-separated floats: `y0 x0 alpha beta`

#### `locang(self, ele, azi)`
Converts elevation/azimuth angles to latitude/longitude/zenith.

**Parameters:**
- `ele` (float): Elevation angle [degrees]
- `azi` (float): Azimuth angle [degrees]

**Returns:**
- Tuple: `(lat, lon, zen)` in degrees, or `(nan, nan, nan)` if no Earth intersection

**Description:** Uses spherical geometry to find where a ray from the satellite at given angles intersects Earth's surface.

#### `locate(self, ix, iy)`
Converts pixel coordinates to geographic coordinates.

**Parameters:**
- `ix` (int): Pixel x-coordinate
- `iy` (int): Pixel y-coordinate

**Returns:**
- Tuple: `(lat, lon, zen)` where:
  - `lat`: Latitude [degrees N]
  - `lon`: Longitude [degrees E]
  - `zen`: Zenith angle [degrees]

**Process:**
1. Converts pixel coordinates to elevation/azimuth using calibration parameters
2. Calls `locang()` to convert to geographic coordinates

#### `satang(self, lat, lon)`
Converts geographic coordinates to satellite viewing angles (inverse of `locang`).

**Parameters:**
- `lat` (float): Latitude [degrees N]
- `lon` (float): Longitude [degrees E]

**Returns:**
- Tuple: `(ele, azi, zen)` - elevation, azimuth, and zenith angles [degrees]

---

## Main Program Workflow

The script runs an interactive demonstration with 10 steps:

### Step 1: Load Visible Channel Image
```python
c1 = Image('msg_c01_z12.img')
c1.disp()
```
Loads and displays channel 1 (visible) image from zone 12.

### Step 2: Define Region of Interest
```python
box = {'xmin':250, 'xmax':450, 'ymin':350, 'ymax':550}
c1.disp(box=box)
```
Overlays a rectangular box on the image to select a region for analysis.

### Step 3: Extract Subset
```python
c1b = c1.clip(box, title='Selected part of Vis image')
c1b.disp()
```
Creates a new image containing only the boxed region.

### Step 4: Load Corresponding IR Image
```python
c9 = Image('msg_c09_z12.img')
c9b = c9.clip(box, title='Corresponding part of Ch9 image')
c9b.disp(window=2)
```
Loads channel 9 (10.8μm infrared window) and extracts matching region.

### Step 5: Convert to Brightness Temperature
```python
c9b.bright(wavelength=10.79)
c9b.disp(window=2)
```
Converts channel 9 radiance to brightness temperature using λ=10.79μm.

### Step 6: Create Scatter Plot
```python
c1vec = c1b.data.flatten()
c9vec = c9b.data.flatten()
plt.scatter(c1vec, c9vec, s=1, color='black')
```
Creates scatter plot of Ch9 temperature vs Ch1 radiance to identify relationships between channels.

### Step 7: Define Cloud Detection Thresholds
```python
c9cld = 280  # Temperature threshold [K]
c1cld = 100  # Radiance threshold
```
Sets thresholds for cloud-free identification:
- **Cloud-free pixels**: Ch9 > 280K AND Ch1 < 100
- Warmer temperatures and lower visible reflectance indicate clear sky

### Step 8: Apply Cloud Mask
```python
c1mask = c1b.data.__lt__(c1cld)
c9mask = c9b.data.__gt__(c9cld)
cloudmask = np.logical_and(c1mask, c9mask)
c1b.data = np.where(cloudmask, c1b.data, 100)
c9b.data = np.where(cloudmask, c9b.data, 200)
```
Creates boolean mask identifying cloud-free pixels and sets cloudy pixels to fixed values (100 for Ch1, 200K for Ch9).

### Step 9: Further Subset and Histogram
```python
box2 = {'xmin':260, 'xmax':280, 'ymin':390, 'ymax':410}
c9b = c9b.clip(box2)
plt.hist(c9vec, range=[280,300], bins=51)
```
Selects smaller region and creates histogram of brightness temperatures to analyze distribution.

### Step 10: Geographic Calibration
```python
geo = Geo('geo.txt')
(latmin, lonmin, zenmin) = geo.locate(box2['xmin'], box2['ymin'])
(latmax, lonmax, zenmax) = geo.locate(box2['xmax'], box2['ymax'])
```
Loads geometric calibration and converts pixel coordinates to latitude/longitude for the analyzed region.

---

## Usage

Run the script interactively:
```bash
python3 ap03mp.py
```

### How to Interact with the Script

The script runs in **interactive mode** and will pause at multiple steps, waiting for your input.

**Two types of prompts:**

1. **Simple pauses** - Just press Enter to continue (most steps)
2. **Input prompts** - You can enter values or press Enter to use defaults (Steps 2 and 8)

**Example interaction:**
```
1. Read in and display Vis image ... [Press Enter]
   ← Just press Enter here
   
[Image displays]

2. Select a subset of the image ...
   Define a box by pixel coordinates
   (Image dimensions: nx=3712, ny=3712)
   Default values: xmin=250, xmax=450, ymin=350, ymax=550
   Press Enter to use defaults, or enter custom values:
   Enter xmin (left edge) [250]: 
   ← Press Enter for default (250), or type a number like 300 then Enter
   Enter xmax (right edge) [450]: 
   ← Press Enter for default (450), or type a number like 500 then Enter
   Enter ymin (bottom edge) [350]: 
   Enter ymax (top edge) [550]: 
   
[Box appears on image with your selected coordinates]

3. Create a new image from subset ... [Press Enter]
   ← Just press Enter here

... and so on
```

**Interactive Input Steps:**
- **Step 2**: Select box coordinates for initial region of interest
  - Enter custom values (e.g., `300`) or press Enter to use defaults
  - You'll be prompted for: xmin, xmax, ymin, ymax
  
- **Step 8**: Select smaller box coordinates for histogram analysis
  - Same format as Step 2
  - Coordinates are relative to the already-clipped image

**Benefits:**
- Examine each plot carefully before moving on
- Experiment with different regions without editing the code
- See image dimensions before selecting coordinates
- Use default values for a quick run-through

**Tip:** Keep the terminal and plot windows visible side-by-side so you can see both the prompts and the resulting images.

### Additional Customization Options

**Already Interactive:**
- ✅ Box coordinates (Step 2 and Step 8) - You can now enter custom values

**Still Hardcoded (could be made interactive):**
- Cloud detection thresholds (Step 7): `c9cld = 280`, `c1cld = 100`
- Wavelength for brightness temperature conversion (Step 5): `wavelength = 10.79`
- Histogram range and bins (Step 9): `range=[280,300], bins=51`

**Example - Making cloud thresholds interactive:**

If you want to experiment with cloud detection thresholds, you could modify Step 7:

Current code:
```python
input('7. Select threshold values for cloud detection... [Press Enter]')
c9cld = 280
c1cld = 100
```

Modified for user input:
```python
print('\n7. Select threshold values for cloud detection...')
print('   Default values: Ch9 threshold=280K, Ch1 threshold=100')
print('   Press Enter to use defaults, or enter custom values:')
c9_input = input('   Enter Ch9 temperature threshold in K [280]: ').strip()
c9cld = float(c9_input) if c9_input else 280
c1_input = input('   Enter Ch1 radiance threshold [100]: ').strip()
c1cld = float(c1_input) if c1_input else 100
```

This pattern (with default values in square brackets) can be applied to any hardcoded parameter you want to make interactive!

---

## Key Concepts Demonstrated

1. **Image I/O**: Reading structured satellite image files
2. **Subsetting**: Extracting regions of interest from full-disk images
3. **Radiometric Calibration**: Converting radiance to brightness temperature
4. **Multi-channel Analysis**: Combining visible and infrared data
5. **Cloud Detection**: Using threshold-based classification
6. **Data Masking**: Filtering pixels based on conditions
7. **Statistical Analysis**: Histograms and scatter plots
8. **Geometric Calibration**: Converting between pixel and geographic coordinates

---

## Files Required

- `msg_c01_z12.img`: Visible channel (Ch1) image for zone 12
- `msg_c09_z12.img`: Infrared channel (Ch9) image for zone 12
- `geo.txt`: Geometric calibration parameters

---

## Notes for Mini Project

This script provides a template for:
- Processing multiple satellite channels
- Identifying cloud-free regions
- Analyzing surface temperature patterns
- Computing geographic locations of features
- Creating visualizations for analysis

You can extend this by:
- Processing all 11 channels
- Analyzing all 25 zones
- Implementing more sophisticated cloud detection algorithms
- Computing derived products (e.g., temperature deficits, spectral indices)
- Creating composite images or time series analysis


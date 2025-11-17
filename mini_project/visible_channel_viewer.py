"""
Visible Channel Viewer - Displays channels 1, 2, 3 with threshold-based binary classification

For each channel:
- Left: Original image
- Right: Binary classification (red = above threshold, blue = below threshold)
- Threshold textbox below each pair
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.colors import ListedColormap

BASE_DIR = 'data'

#-------------------------------------------------------------------------------
class Geo:
    """ Geometric calibration data and methods """
    
    def __init__(self, geofil):
        """ Initialise new Geo object
        
        PARAMETERS
          geofil str : name of file containing geo.cal data, eg 'geo.txt' 
        """
        
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
    
    def locang(self, ele, azi):
        """ Convert ele,azi angles to lat,lon,zen angles """
        rele     = math.radians(ele)
        sinele   = math.sin(rele)
        cosele   = math.cos(rele)
        razi     = math.radians(azi)
        sinazi   = math.sin(razi)
        cosazi   = math.cos(razi)
        # Distance of plane of intersection from centre of earth
        h = self.DIST * sinele 
        if abs(h) > self.REARTH: return (np.nan, np.nan, np.nan)  # no sfc intersect.
        r1 = math.sqrt(self.REARTH**2 - h**2) # Radius of circle of intersection
        d1 = self.DIST * cosele
        if abs(d1 * sinazi) > r1: return (np.nan, np.nan, np.nan) # No intersection
        # Distance of line of sight
        x = d1 * cosazi - math.sqrt(r1**2 - d1**2 * sinazi**2)
        # Distance from pixel to point of intersection of earth's vertical axis with
        # plane of intersection
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
        # Convert lat,lon from degrees to radians
        rlat  = math.radians(lat)
        rlon  = math.radians(lon)
        # Height [km] of pixel above horizontal
        h2    = self.REARTH * math.sin(rlat)     
        # Distance [km] from earth's vertical axis
        r2    = self.REARTH * math.cos(rlat)     
        # Horizontal distance of pixel from satellite
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

class VisibleChannelViewer:
    def __init__(self):
        # Load geographic calibration
        self.geo = Geo('geo.txt')
        
        # Initial parameters
        self.current_zone = 12
        self.x_center = 300
        self.y_center = 800
        self.box_size = 100
        
        # Threshold values for each channel
        self.thresholds = {
            1: 50.0,
            2: 50.0,
            3: 50.0
        }
        
        # Storage
        self.images = {}  # {channel: image_data}
        self.colorbars = []  # Store all colorbars
        self.hist_buttons = []  # Store histogram button references
        self.current_display_data = {}  # Store currently displayed data for histograms
        
        # Binary colormap: blue for below, red for above
        self.binary_cmap = ListedColormap(['blue', 'red'])
        
        # Setup figure (sized for 13" MacBook, taller to accommodate extra controls)
        self.fig = plt.figure(figsize=(12, 10.5))
        
        # Create 3 rows x 2 columns layout
        # Each row has: [original image | binary classification]
        gs = self.fig.add_gridspec(3, 2, left=0.08, right=0.95, top=0.95, 
                                   bottom=0.19, hspace=0.35, wspace=0.25)
        
        # Create axes for each channel pair
        self.axes_original = []
        self.axes_binary = []
        self.threshold_boxes = []
        
        for i, channel in enumerate([1, 2, 3]):
            # Original image
            ax_orig = self.fig.add_subplot(gs[i, 0])
            self.axes_original.append(ax_orig)
            
            # Binary classification image
            ax_bin = self.fig.add_subplot(gs[i, 1])
            self.axes_binary.append(ax_bin)
        
        # Control widgets at bottom
        # Zone slider
        ax_zone = plt.axes([0.15, 0.155, 0.70, 0.02])
        self.zone_slider = Slider(ax_zone, 'Zone', 0, 24, 
                                  valinit=self.current_zone, valstep=1)
        self.zone_slider.on_changed(self.update_zone)
        
        # X, Y, Box size controls (in one row) - Pixel coordinates
        ax_x = plt.axes([0.10, 0.105, 0.10, 0.025])
        self.x_box = TextBox(ax_x, 'X(px)', initial=str(self.x_center))
        self.x_box.on_submit(self.update_x)
        
        ax_y = plt.axes([0.28, 0.105, 0.10, 0.025])
        self.y_box = TextBox(ax_y, 'Y(px)', initial=str(self.y_center))
        self.y_box.on_submit(self.update_y)
        
        ax_box = plt.axes([0.46, 0.105, 0.10, 0.025])
        self.box_box = TextBox(ax_box, 'Box', initial=str(self.box_size))
        self.box_box.on_submit(self.update_box_size)
        
        # Reset button
        ax_reset = plt.axes([0.64, 0.105, 0.08, 0.025])
        self.reset_button = Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(self.reset_view)
        
        # Lat/Lon controls (second row)
        # Calculate initial lat/lon
        lat, lon, zen = self.geo.locate(self.x_center, self.y_center)
        lat_str = f'{lat:.2f}' if not np.isnan(lat) else 'N/A'
        lon_str = f'{lon:.2f}' if not np.isnan(lon) else 'N/A'
        
        ax_lat = plt.axes([0.10, 0.055, 0.13, 0.025])
        self.lat_box = TextBox(ax_lat, 'Lat(°N)', initial=lat_str)
        self.lat_box.on_submit(self.update_from_lat)
        
        ax_lon = plt.axes([0.28, 0.055, 0.13, 0.025])
        self.lon_box = TextBox(ax_lon, 'Lon(°E)', initial=lon_str)
        self.lon_box.on_submit(self.update_from_lon)
        
        # Add a text display for current coordinates
        ax_coords = plt.axes([0.46, 0.055, 0.26, 0.025])
        ax_coords.axis('off')
        self.coords_text = ax_coords.text(0.0, 0.5, '', transform=ax_coords.transAxes,
                                          fontsize=9, verticalalignment='center')
        
        # Threshold controls (one per channel) - moved to third row
        threshold_positions = [
            [0.10, 0.01, 0.13, 0.025],  # Ch1
            [0.28, 0.01, 0.13, 0.025],  # Ch2
            [0.46, 0.01, 0.13, 0.025],  # Ch3
        ]
        
        for i, channel in enumerate([1, 2, 3]):
            ax_thresh = plt.axes(threshold_positions[i])
            thresh_box = TextBox(ax_thresh, f'Ch{channel} Thr.', 
                               initial=str(self.thresholds[channel]))
            thresh_box.on_submit(lambda text, ch=channel: self.update_threshold(ch, text))
            self.threshold_boxes.append(thresh_box)
        
        # Load and display
        self.load_zone(self.current_zone)
        self.update_display()
    
    def load_zone(self, zone):
        """Load channels 1, 2, 3 for the specified zone"""
        print(f"\nLoading zone {zone:02d}...")
        self.images = {}
        
        for channel in [1, 2, 3]:
            img_path = f"{BASE_DIR}/channel_{channel}/msg_c{channel:02d}_z{zone:02d}.img"
            try:
                self.images[channel] = self.load_image(img_path)
                print(f"  Ch{channel}: ✓ ({self.images[channel]['nx']}x{self.images[channel]['ny']})")
            except Exception as e:
                print(f"  Ch{channel}: ✗ ({e})")
                self.images[channel] = None
        
        loaded_count = len([v for v in self.images.values() if v is not None])
        print(f"Loaded {loaded_count}/3 channels")
    
    def load_image(self, filepath):
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
    
    def get_zoom_region(self, data):
        """Extract zoom region"""
        ny, nx = data.shape
        half_box = self.box_size // 2
        
        x_min = max(0, self.x_center - half_box)
        x_max = min(nx, self.x_center + half_box)
        y_min = max(0, self.y_center - half_box)
        y_max = min(ny, self.y_center + half_box)
        
        return data[y_min:y_max, x_min:x_max]
    
    def create_binary_classification(self, data, threshold):
        """Create binary classification: 0 for below threshold, 1 for above"""
        return (data >= threshold).astype(int)
    
    def update_display(self):
        """Update all channel displays"""
        # Remove old colorbars BEFORE clearing axes
        for cbar in self.colorbars:
            try:
                cbar.remove()
            except (KeyError, ValueError):
                pass
        self.colorbars = []
        
        # Clear stored display data
        self.current_display_data = {}
        
        # Update each channel
        for i, channel in enumerate([1, 2, 3]):
            ax_orig = self.axes_original[i]
            ax_bin = self.axes_binary[i]
            
            ax_orig.clear()
            ax_bin.clear()
            
            if self.images[channel] is None:
                ax_orig.text(0.5, 0.5, f'Ch{channel}\nNot Found', 
                           ha='center', va='center', transform=ax_orig.transAxes,
                           fontsize=12)
                ax_orig.axis('off')
                ax_bin.axis('off')
                continue
            
            # Get data
            data = self.images[channel]['data']
            zoom_data = self.get_zoom_region(data)
            
            # Store data for histograms
            self.current_display_data[f'{channel}_orig'] = {
                'data': zoom_data,
                'label': f'Channel {channel} - Original',
                'type': 'radiance'
            }
            
            # Display original image
            im_orig = ax_orig.imshow(zoom_data, origin='lower', cmap='gray')
            ax_orig.set_title(f'Channel {channel} - Original', fontsize=10, fontweight='bold')
            ax_orig.axis('off')
            cbar_orig = plt.colorbar(im_orig, ax=ax_orig, fraction=0.046, pad=0.04)
            cbar_orig.set_label('Radiance', fontsize=8)
            self.colorbars.append(cbar_orig)
            
            # Add histogram button for original image
            bbox_orig = ax_orig.get_position()
            button_ax_orig = plt.axes([bbox_orig.x0 + bbox_orig.width*0.15, bbox_orig.y0 - 0.040, 
                                      bbox_orig.width*0.7, 0.018])
            hist_btn_orig = Button(button_ax_orig, 'Histogram', color='#87CEEB', hovercolor='#4682B4')
            hist_btn_orig.label.set_fontsize(8)
            hist_btn_orig.on_clicked(lambda event, ch=channel: self.show_histogram(f'{ch}_orig'))
            self.hist_buttons.append(hist_btn_orig)
            
            # Create and display binary classification
            threshold = self.thresholds[channel]
            binary_data = self.create_binary_classification(zoom_data, threshold)
            
            # Store binary data for histograms
            self.current_display_data[f'{channel}_bin'] = {
                'data': binary_data,
                'label': f'Channel {channel} - Binary',
                'type': 'binary',
                'threshold': threshold
            }
            
            im_bin = ax_bin.imshow(binary_data, origin='lower', cmap=self.binary_cmap,
                                  vmin=0, vmax=1)
            ax_bin.set_title(f'Channel {channel} - Classification\nThreshold: {threshold:.1f}',
                           fontsize=10, fontweight='bold')
            ax_bin.axis('off')
            
            # Add custom colorbar labels
            cbar_bin = plt.colorbar(im_bin, ax=ax_bin, fraction=0.046, pad=0.04,
                                   ticks=[0.25, 0.75])
            cbar_bin.ax.set_yticklabels(['Below', 'Above'], fontsize=8)
            self.colorbars.append(cbar_bin)
            
            # Add histogram button for binary image
            bbox_bin = ax_bin.get_position()
            button_ax_bin = plt.axes([bbox_bin.x0 + bbox_bin.width*0.15, bbox_bin.y0 - 0.040, 
                                     bbox_bin.width*0.7, 0.018])
            hist_btn_bin = Button(button_ax_bin, 'Histogram', color='#87CEEB', hovercolor='#4682B4')
            hist_btn_bin.label.set_fontsize(8)
            hist_btn_bin.on_clicked(lambda event, ch=channel: self.show_histogram(f'{ch}_bin'))
            self.hist_buttons.append(hist_btn_bin)
            
            # Add statistics text on binary image
            below_count = np.sum(binary_data == 0)
            above_count = np.sum(binary_data == 1)
            total = binary_data.size
            below_pct = 100 * below_count / total
            above_pct = 100 * above_count / total
            
            stats_text = f'Below: {below_pct:.1f}%\nAbove: {above_pct:.1f}%'
            ax_bin.text(0.02, 0.98, stats_text, transform=ax_bin.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Update coordinate displays
        lat, lon, zen = self.geo.locate(self.x_center, self.y_center)
        if not np.isnan(lat) and not np.isnan(lon):
            lat_str = f'{lat:.2f}'
            lon_str = f'{lon:.2f}'
            self.coords_text.set_text(f'Zenith: {zen:.2f}°')
        else:
            lat_str = 'N/A'
            lon_str = 'N/A'
            self.coords_text.set_text('Outside Earth disk')
        
        # Update lat/lon textboxes
        self.lat_box.set_val(lat_str)
        self.lon_box.set_val(lon_str)
        
        # Add main title
        self.fig.suptitle(f'Visible Channel Viewer - Zone {self.current_zone:02d} | '
                         f'Center: ({self.x_center}, {self.y_center}) px = ({lat_str}°N, {lon_str}°E) | '
                         f'Box: {self.box_size}px',
                         fontsize=12, fontweight='bold')
        
        self.fig.canvas.draw_idle()
    
    def update_zone(self, val):
        new_zone = int(self.zone_slider.val)
        if new_zone != self.current_zone:
            self.current_zone = new_zone
            self.load_zone(self.current_zone)
            self.update_display()
    
    def update_x(self, text):
        try:
            self.x_center = int(text)
            self.update_display()
        except ValueError:
            print(f"Invalid X: {text}")
    
    def update_y(self, text):
        try:
            self.y_center = int(text)
            self.update_display()
        except ValueError:
            print(f"Invalid Y: {text}")
    
    def update_box_size(self, text):
        try:
            self.box_size = int(text)
            self.update_display()
        except ValueError:
            print(f"Invalid box size: {text}")
    
    def update_threshold(self, channel, text):
        try:
            self.thresholds[channel] = float(text)
            self.update_display()
            print(f"Ch{channel} threshold updated to {self.thresholds[channel]:.1f}")
        except ValueError:
            print(f"Invalid threshold for Ch{channel}: {text}")
    
    def update_from_lat(self, text):
        """Update position from latitude input"""
        try:
            # Get current lon value
            lon_text = self.lon_box.text
            if lon_text == 'N/A':
                print("Cannot update latitude: longitude is not set")
                return
            
            new_lat = float(text)
            current_lon = float(lon_text)
            
            # Convert lat/lon to x/y
            x, y = self.geo.latlon_to_xy(new_lat, current_lon)
            
            if not np.isnan(x) and not np.isnan(y):
                self.x_center = x
                self.y_center = y
                self.x_box.set_val(str(self.x_center))
                self.y_box.set_val(str(self.y_center))
                self.update_display()
                print(f"Position updated to lat={new_lat:.2f}°N, lon={current_lon:.2f}°E (x={x}, y={y})")
            else:
                print(f"Lat/Lon ({new_lat}, {current_lon}) is outside the image")
        except ValueError:
            print(f"Invalid latitude: {text}")
    
    def update_from_lon(self, text):
        """Update position from longitude input"""
        try:
            # Get current lat value
            lat_text = self.lat_box.text
            if lat_text == 'N/A':
                print("Cannot update longitude: latitude is not set")
                return
            
            new_lon = float(text)
            current_lat = float(lat_text)
            
            # Convert lat/lon to x/y
            x, y = self.geo.latlon_to_xy(current_lat, new_lon)
            
            if not np.isnan(x) and not np.isnan(y):
                self.x_center = x
                self.y_center = y
                self.x_box.set_val(str(self.x_center))
                self.y_box.set_val(str(self.y_center))
                self.update_display()
                print(f"Position updated to lat={current_lat:.2f}°N, lon={new_lon:.2f}°E (x={x}, y={y})")
            else:
                print(f"Lat/Lon ({current_lat}, {new_lon}) is outside the image")
        except ValueError:
            print(f"Invalid longitude: {text}")
    
    def reset_view(self, event):
        self.x_center = 300
        self.y_center = 800
        self.box_size = 100
        self.x_box.set_val(str(self.x_center))
        self.y_box.set_val(str(self.y_center))
        self.box_box.set_val(str(self.box_size))
        self.update_display()
    
    def show_histogram(self, data_key):
        """Show histogram for a specific channel/type in a new window"""
        if data_key not in self.current_display_data:
            print(f"No data available for {data_key}")
            return
        
        data_info = self.current_display_data[data_key]
        data = data_info['data']
        label = data_info['label']
        data_type = data_info['type']
        
        # Create new figure for histogram
        hist_fig, hist_ax = plt.subplots(figsize=(8, 6))
        hist_fig.canvas.manager.set_window_title(f'{label} Histogram')
        
        # Flatten data
        data_flat = data.flatten()
        
        if data_type == 'binary':
            # For binary data, show simple bar chart
            below_count = np.sum(data_flat == 0)
            above_count = np.sum(data_flat == 1)
            
            hist_ax.bar(['Below Threshold', 'Above Threshold'], 
                       [below_count, above_count],
                       color=['blue', 'red'], alpha=0.7, edgecolor='black')
            hist_ax.set_ylabel('Pixel Count', fontsize=12)
            hist_ax.set_title(f'{label}\nThreshold: {data_info["threshold"]:.1f} | '
                            f'Zone {self.current_zone:02d}',
                            fontsize=12, fontweight='bold')
            
            # Add percentage labels on bars
            total = len(data_flat)
            for i, (count, color) in enumerate([(below_count, 'blue'), (above_count, 'red')]):
                pct = 100 * count / total
                hist_ax.text(i, count, f'{count:,}\n({pct:.1f}%)', 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
        else:
            # For radiance data, show histogram
            n, bins, patches = hist_ax.hist(data_flat, bins=50, color='steelblue', 
                                           edgecolor='black', alpha=0.7)
            
            hist_ax.set_xlabel('Radiance', fontsize=12)
            hist_ax.set_ylabel('Frequency', fontsize=12)
            hist_ax.set_title(f'{label} - Zone {self.current_zone:02d}\n'
                            f'Center: ({self.x_center}, {self.y_center}) | Box: {self.box_size}px',
                            fontsize=12, fontweight='bold')
            
            # Add statistics text
            stats_text = f'Mean: {data_flat.mean():.2f}\n'
            stats_text += f'Std: {data_flat.std():.2f}\n'
            stats_text += f'Min: {data_flat.min():.2f}\n'
            stats_text += f'Max: {data_flat.max():.2f}\n'
            stats_text += f'Pixels: {len(data_flat)}'
            
            hist_ax.text(0.98, 0.97, stats_text, transform=hist_ax.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        hist_ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    print("Visible Channel Viewer (Channels 1, 2, 3)")
    print("=" * 60)
    print("\nFeatures:")
    print("  - Left column: Original radiance images")
    print("  - Right column: Binary classification (Blue=Below, Red=Above)")
    print("\nControls:")
    print("  - Zone Slider: Select time/zone (z00-z24)")
    print("  - X(px)/Y(px)/Box: Set zoom region center (pixels) and size")
    print("  - Lat(°N)/Lon(°E): Set zoom region center (geographic coordinates)")
    print("    * Enter latitude or longitude to navigate by coordinates")
    print("    * Displays current zenith angle")
    print("  - Reset: Return to default view (300, 800)")
    print("  - Ch1/2/3 Thr.: Set threshold for binary classification")
    print("  - Histogram buttons: Click to view histogram for each image")
    print("    * Original images: Show radiance distribution")
    print("    * Binary images: Show count of pixels above/below threshold")
    print("\nGeographic Calibration:")
    print("  - Title bar shows pixel coordinates and lat/lon for current center")
    print("  - You can input either pixels OR lat/lon to set the view center")
    print("\nLoading initial data...")
    
    viewer = VisibleChannelViewer()
    plt.show()


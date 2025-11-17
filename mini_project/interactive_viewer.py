"""
Interactive Multi-Channel Satellite Image Viewer

Features:
- View all 11 channels simultaneously in a grid
- Slider to select zone/time (z00-z24)
- Zoom controls (x center, y center, box size)
- Toggle between radiance and brightness temperature for IR channels
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button, CheckButtons
from matplotlib.colors import LinearSegmentedColormap

# Base directory for image files (relative to mini_project folder)
BASE_DIR = 'data'

# Channel wavelengths (microns) for brightness temperature conversion
WAVELENGTHS = {
    1: None,    # Visible - no conversion
    2: None,    # Near-IR - keep as radiance
    3: None,    # Near-IR
    4: 3.9,     # IR window
    5: 6.2,     # Water vapor
    6: 7.3,     # Water vapor
    7: 8.7,     # IR window
    8: 9.7,     # Ozone
    9: 10.79,   # IR window
    10: 11.94,  # IR window
    11: 13.4    # IR window
}

class ImageViewer:
    def __init__(self):
        # Initial parameters
        self.current_zone = 12  # Start at z12
        self.x_center = 1856    # Middle of 3712x3712 image
        self.y_center = 1856
        self.box_size = 1000    # Size of zoom box
        self.show_brightness_temp = True  # Toggle for IR channels
        
        # Storage for loaded images
        self.images = {}  # Will store Image objects for each channel
        self.colorbars = []  # Store colorbar references
        
        # Setup the figure and axes
        self.setup_figure()
        
        # Load initial images
        self.load_zone(self.current_zone)
        
        # Display images
        self.update_display()
        
    def setup_figure(self):
        """Create figure with grid of subplots and control widgets"""
        self.fig = plt.figure(figsize=(16, 12))
        
        # Create 3x4 grid for 11 channels (one empty subplot)
        gs = self.fig.add_gridspec(4, 4, left=0.05, right=0.95, top=0.95, 
                                   bottom=0.25, hspace=0.3, wspace=0.3)
        
        self.axes = []
        for i in range(11):
            row = i // 4
            col = i % 4
            ax = self.fig.add_subplot(gs[row, col])
            ax.axis('off')
            self.axes.append(ax)
        
        # Add control widgets at the bottom
        # Zone slider
        ax_zone = plt.axes([0.15, 0.15, 0.7, 0.03])
        self.zone_slider = Slider(
            ax_zone, 'Zone', 0, 24, valinit=self.current_zone, 
            valstep=1, color='steelblue'
        )
        self.zone_slider.on_changed(self.update_zone)
        
        # X center textbox
        ax_x = plt.axes([0.15, 0.10, 0.1, 0.03])
        self.x_box = TextBox(ax_x, 'X Center', initial=str(self.x_center))
        self.x_box.on_submit(self.update_x)
        
        # Y center textbox
        ax_y = plt.axes([0.35, 0.10, 0.1, 0.03])
        self.y_box = TextBox(ax_y, 'Y Center', initial=str(self.y_center))
        self.y_box.on_submit(self.update_y)
        
        # Box size textbox
        ax_box = plt.axes([0.55, 0.10, 0.1, 0.03])
        self.box_box = TextBox(ax_box, 'Box Size', initial=str(self.box_size))
        self.box_box.on_submit(self.update_box_size)
        
        # Reset button
        ax_reset = plt.axes([0.75, 0.10, 0.1, 0.03])
        self.reset_button = Button(ax_reset, 'Reset View')
        self.reset_button.on_clicked(self.reset_view)
        
        # Brightness temperature toggle
        ax_toggle = plt.axes([0.15, 0.05, 0.2, 0.03])
        self.temp_check = CheckButtons(ax_toggle, ['Show Brightness Temp'], [self.show_brightness_temp])
        self.temp_check.on_clicked(self.toggle_brightness_temp)
        
    def load_zone(self, zone):
        """Load all 11 channels for the specified zone"""
        print(f"Loading zone {zone:02d}...")
        self.images = {}
        
        for channel in range(1, 12):
            img_path = f"{BASE_DIR}/channel_{channel}/msg_c{channel:02d}_z{zone:02d}.img"
            try:
                self.images[channel] = self.load_image(img_path)
            except FileNotFoundError:
                print(f"Warning: Could not find {img_path}")
                self.images[channel] = None
        
        print(f"Loaded {len([v for v in self.images.values() if v is not None])} channels")
    
    def load_image(self, filepath):
        """Load a single image file"""
        with open(filepath, 'r') as f:
            title = f.readline().strip()
            nx, ny = [int(x) for x in f.readline().split()]
            imgdata = []
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    imgdata.extend([float(x) for x in line.split()])
            data = np.array(imgdata).reshape((ny, nx))
        return {'data': data, 'nx': nx, 'ny': ny, 'title': title}
    
    def brightness_temperature(self, radiance, wavelength):
        """Convert radiance to brightness temperature"""
        if wavelength is None:
            return radiance
        
        # Physical constants (from the example script)
        H = 6.63e-34       # Planck constant [m2 kg / s]
        C = 3.00e8         # Speed of light [m / s]
        K = 1.38e-23       # Boltzmann constant [m2 kg /s2 /K]
        R1 = H * C / K     # Intermediate Constant [m K]
        R2 = 2 * H * C**2  # Intermediate Constant [m4 kg / s3]
        
        w = wavelength * 1.0e-6  # convert microns to metres
        temp = R1 / w / np.log(1.0 + R2/(w**5 * radiance*1e6))
        return temp
    
    def get_zoom_region(self, data):
        """Extract zoom region from full image data"""
        ny, nx = data.shape
        half_box = self.box_size // 2
        
        # Calculate bounds with clipping
        x_min = max(0, self.x_center - half_box)
        x_max = min(nx, self.x_center + half_box)
        y_min = max(0, self.y_center - half_box)
        y_max = min(ny, self.y_center + half_box)
        
        return data[y_min:y_max, x_min:x_max]
    
    def update_display(self):
        """Update all channel displays"""
        # Remove old colorbars BEFORE clearing axes
        for cbar in self.colorbars:
            try:
                cbar.remove()
            except (KeyError, ValueError):
                pass  # Already removed
        self.colorbars = []
        
        for i, channel in enumerate(range(1, 12)):
            ax = self.axes[i]
            ax.clear()
            ax.axis('off')
            
            if self.images[channel] is None:
                ax.text(0.5, 0.5, f'Ch{channel}\nNot Found', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Get data
            data = self.images[channel]['data']
            
            # Apply brightness temperature conversion if applicable
            wavelength = WAVELENGTHS[channel]
            is_ir = wavelength is not None
            
            if is_ir and self.show_brightness_temp:
                data = self.brightness_temperature(data, wavelength)
                label = f'Ch{channel} ({wavelength}μm)\nTemp [K]'
                # Use temperature colormap
                colours = [(1,1,1),(1,0,1),(0,0,1),(0,1,1),(0,1,0),(1,1,0),
                          (1,0,0),(0,0,0)]
                cmap = LinearSegmentedColormap.from_list('tem_colours', colours)
                vmin, vmax = 230, 320
            else:
                label = f'Ch{channel}'
                if wavelength:
                    label += f' ({wavelength}μm)\nRadiance'
                else:
                    label += '\nRadiance'
                cmap = 'gray'
                vmin, vmax = None, None
            
            # Get zoom region
            zoom_data = self.get_zoom_region(data)
            
            # Display
            im = ax.imshow(zoom_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(label, fontsize=9)
            
            # Add colorbar and store reference
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self.colorbars.append(cbar)
        
        # Update title
        self.fig.suptitle(
            f'Zone {self.current_zone:02d} | Center: ({self.x_center}, {self.y_center}) | Box: {self.box_size}px',
            fontsize=14, fontweight='bold'
        )
        
        self.fig.canvas.draw_idle()
    
    def update_zone(self, val):
        """Callback for zone slider"""
        new_zone = int(self.zone_slider.val)
        if new_zone != self.current_zone:
            self.current_zone = new_zone
            self.load_zone(self.current_zone)
            self.update_display()
    
    def update_x(self, text):
        """Callback for X center textbox"""
        try:
            self.x_center = int(text)
            self.update_display()
        except ValueError:
            print(f"Invalid X value: {text}")
    
    def update_y(self, text):
        """Callback for Y center textbox"""
        try:
            self.y_center = int(text)
            self.update_display()
        except ValueError:
            print(f"Invalid Y value: {text}")
    
    def update_box_size(self, text):
        """Callback for box size textbox"""
        try:
            self.box_size = int(text)
            self.update_display()
        except ValueError:
            print(f"Invalid box size: {text}")
    
    def reset_view(self, event):
        """Reset to full image view"""
        self.x_center = 1856
        self.y_center = 1856
        self.box_size = 3712  # Full image
        self.x_box.set_val(str(self.x_center))
        self.y_box.set_val(str(self.y_center))
        self.box_box.set_val(str(self.box_size))
        self.update_display()
    
    def toggle_brightness_temp(self, label):
        """Toggle brightness temperature display"""
        self.show_brightness_temp = not self.show_brightness_temp
        self.update_display()

# Main program
if __name__ == '__main__':
    print("Starting Interactive Satellite Image Viewer...")
    print("\nControls:")
    print("  - Zone Slider: Select time/zone (z00-z24)")
    print("  - X/Y Center: Set center of zoom region")
    print("  - Box Size: Set size of zoom region (pixels)")
    print("  - Reset View: Return to full image")
    print("  - Show Brightness Temp: Toggle temperature/radiance for IR channels")
    print("\nLoading initial data...")
    
    viewer = ImageViewer()
    plt.show()


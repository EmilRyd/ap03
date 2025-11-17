"""
Multi-Channel Viewer - Displays all 11 channels simultaneously

Based on the working simple_viewer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button, CheckButtons
from matplotlib.colors import LinearSegmentedColormap

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

class MultiChannelViewer:
    def __init__(self):
        # Initial parameters
        self.current_zone = 12
        self.x_center = 1856
        self.y_center = 1856
        self.box_size = 3712  # Full image
        self.show_brightness_temp = True
        
        # Storage
        self.images = {}  # {channel: image_data}
        self.colorbars = []  # Store all colorbars
        self.hist_buttons = []  # Store histogram button references
        self.current_display_data = {}  # Store currently displayed data for histograms
        
        # Setup figure with 3x4 grid
        self.fig = plt.figure(figsize=(18, 13))
        gs = self.fig.add_gridspec(4, 4, left=0.03, right=0.97, top=0.98, 
                                   bottom=0.18, hspace=0.20, wspace=0.2)
        
        self.axes = []
        self.button_axes = []  # Store button axes
        for i in range(11):
            row = i // 4
            col = i % 4
            ax = self.fig.add_subplot(gs[row, col])
            ax.axis('off')
            self.axes.append(ax)
            
            # Create histogram button for each channel (will position in update_display)
            self.hist_buttons.append(None)
        
        # Control widgets
        # Zone slider
        ax_zone = plt.axes([0.15, 0.11, 0.7, 0.02])
        self.zone_slider = Slider(ax_zone, 'Zone', 0, 24, 
                                  valinit=self.current_zone, valstep=1)
        self.zone_slider.on_changed(self.update_zone)
        
        # X center
        ax_x = plt.axes([0.15, 0.06, 0.1, 0.025])
        self.x_box = TextBox(ax_x, 'X Center', initial=str(self.x_center))
        self.x_box.on_submit(self.update_x)
        
        # Y center
        ax_y = plt.axes([0.35, 0.06, 0.1, 0.025])
        self.y_box = TextBox(ax_y, 'Y Center', initial=str(self.y_center))
        self.y_box.on_submit(self.update_y)
        
        # Box size
        ax_box = plt.axes([0.55, 0.06, 0.1, 0.025])
        self.box_box = TextBox(ax_box, 'Box Size', initial=str(self.box_size))
        self.box_box.on_submit(self.update_box_size)
        
        # Reset button
        ax_reset = plt.axes([0.75, 0.06, 0.1, 0.025])
        self.reset_button = Button(ax_reset, 'Reset View')
        self.reset_button.on_clicked(self.reset_view)
        
        # Brightness temperature toggle
        ax_toggle = plt.axes([0.15, 0.015, 0.2, 0.025])
        self.temp_check = CheckButtons(ax_toggle, ['Show Brightness Temp'], 
                                       [self.show_brightness_temp])
        self.temp_check.on_clicked(self.toggle_brightness_temp)
        
        # Load and display
        self.load_zone(self.current_zone)
        self.update_display()
        
    def load_zone(self, zone):
        """Load all 11 channels for the specified zone"""
        print(f"\nLoading zone {zone:02d}...")
        self.images = {}
        
        for channel in range(1, 12):
            img_path = f"{BASE_DIR}/channel_{channel}/msg_c{channel:02d}_z{zone:02d}.img"
            try:
                self.images[channel] = self.load_image(img_path)
                print(f"  Ch{channel}: ✓ ({self.images[channel]['nx']}x{self.images[channel]['ny']})")
            except Exception as e:
                print(f"  Ch{channel}: ✗ ({e})")
                self.images[channel] = None
        
        loaded_count = len([v for v in self.images.values() if v is not None])
        print(f"Loaded {loaded_count}/11 channels")
    
    def load_image(self, filepath):
        """Load a single image file - using proven working method"""
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
    
    def brightness_temperature(self, radiance, wavelength):
        """Convert radiance to brightness temperature"""
        if wavelength is None:
            return radiance
        
        # Physical constants
        H = 6.63e-34       # Planck constant [m2 kg / s]
        C = 3.00e8         # Speed of light [m / s]
        K = 1.38e-23       # Boltzmann constant [m2 kg /s2 /K]
        R1 = H * C / K
        R2 = 2 * H * C**2
        
        w = wavelength * 1.0e-6  # convert microns to metres
        temp = R1 / w / np.log(1.0 + R2/(w**5 * radiance*1e6))
        return temp
    
    def get_zoom_region(self, data):
        """Extract zoom region"""
        ny, nx = data.shape
        half_box = self.box_size // 2
        
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
                pass
        self.colorbars = []
        
        # Clear stored display data
        self.current_display_data = {}
        
        # Update each channel
        for i, channel in enumerate(range(1, 12)):
            ax = self.axes[i]
            ax.clear()
            ax.axis('off')
            
            if self.images[channel] is None:
                ax.text(0.5, 0.5, f'Ch{channel}\nNot Found', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10)
                continue
            
            # Get data
            data = self.images[channel]['data']
            
            # Apply brightness temperature conversion if applicable
            wavelength = WAVELENGTHS[channel]
            is_ir = wavelength is not None
            
            if is_ir and self.show_brightness_temp:
                data = self.brightness_temperature(data, wavelength)
                label = f'Ch{channel} ({wavelength}μm)\nTemp [K]'
                # Temperature colormap
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
            
            # Store for histogram access
            self.current_display_data[channel] = {
                'data': zoom_data,
                'label': label,
                'is_temp': is_ir and self.show_brightness_temp
            }
            
            # Display
            im = ax.imshow(zoom_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(label, fontsize=8, pad=2)
            
            # Add colorbar (smaller to give more space to images)
            cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
            self.colorbars.append(cbar)
            
            # Add histogram button below the image
            bbox = ax.get_position()
            button_ax = plt.axes([bbox.x0 + bbox.width*0.3, bbox.y0 - 0.035, 
                                 bbox.width*0.4, 0.02])
            hist_btn = Button(button_ax, 'Histogram', color='lightgray', hovercolor='gray')
            hist_btn.on_clicked(lambda event, ch=channel: self.show_histogram(ch))
            self.hist_buttons[i] = hist_btn
        
        # Update main title
        self.fig.suptitle(
            f'Zone {self.current_zone:02d} | Center: ({self.x_center}, {self.y_center}) | Box: {self.box_size}px',
            fontsize=12, fontweight='bold', y=0.995
        )
        
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
    
    def reset_view(self, event):
        self.x_center = 1856
        self.y_center = 1856
        self.box_size = 3712
        self.x_box.set_val(str(self.x_center))
        self.y_box.set_val(str(self.y_center))
        self.box_box.set_val(str(self.box_size))
        self.update_display()
    
    def toggle_brightness_temp(self, label):
        self.show_brightness_temp = not self.show_brightness_temp
        self.update_display()
    
    def show_histogram(self, channel):
        """Show histogram for a specific channel in a new window"""
        if channel not in self.current_display_data:
            print(f"No data available for channel {channel}")
            return
        
        data_info = self.current_display_data[channel]
        data = data_info['data']
        label = data_info['label']
        is_temp = data_info['is_temp']
        
        # Create new figure for histogram
        hist_fig, hist_ax = plt.subplots(figsize=(8, 6))
        hist_fig.canvas.manager.set_window_title(f'Channel {channel} Histogram')
        
        # Flatten data
        data_flat = data.flatten()
        
        # Create histogram
        n, bins, patches = hist_ax.hist(data_flat, bins=50, color='steelblue', 
                                        edgecolor='black', alpha=0.7)
        
        # Labels
        if is_temp:
            hist_ax.set_xlabel('Temperature [K]', fontsize=12)
            hist_ax.set_title(f'{label} - Zone {self.current_zone:02d}\n'
                            f'Center: ({self.x_center}, {self.y_center}) | Box: {self.box_size}px',
                            fontsize=12, fontweight='bold')
        else:
            hist_ax.set_xlabel('Radiance', fontsize=12)
            hist_ax.set_title(f'{label} - Zone {self.current_zone:02d}\n'
                            f'Center: ({self.x_center}, {self.y_center}) | Box: {self.box_size}px',
                            fontsize=12, fontweight='bold')
        
        hist_ax.set_ylabel('Frequency', fontsize=12)
        hist_ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {data_flat.mean():.2f}\n'
        stats_text += f'Std: {data_flat.std():.2f}\n'
        stats_text += f'Min: {data_flat.min():.2f}\n'
        stats_text += f'Max: {data_flat.max():.2f}\n'
        stats_text += f'Pixels: {len(data_flat)}'
        
        hist_ax.text(0.98, 0.97, stats_text, transform=hist_ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    print("Multi-Channel Satellite Image Viewer")
    print("=" * 50)
    print("\nControls:")
    print("  - Zone Slider: Select time/zone (z00-z24)")
    print("  - X/Y Center: Set center of zoom region")
    print("  - Box Size: Set size of zoom region (pixels)")
    print("  - Reset View: Return to full image")
    print("  - Show Brightness Temp: Toggle temp/radiance for IR channels")
    print("  - Histogram buttons: Click to see histogram of each channel")
    print("\nLoading initial data...")
    
    viewer = MultiChannelViewer()
    plt.show()


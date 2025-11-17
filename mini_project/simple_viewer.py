"""
Simple Single-Channel Viewer - for debugging

Displays just one channel at a time with zone slider and zoom controls
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.colors import LinearSegmentedColormap

# Test with one channel first
CHANNEL = 1  # Change this to test different channels
BASE_DIR = 'data'

class SimpleViewer:
    def __init__(self):
        # Initial parameters
        self.current_zone = 12
        self.x_center = 1856
        self.y_center = 1856
        self.box_size = 3712  # Full image
        
        # Storage
        self.image_data = None
        self.colorbar = None  # Store colorbar reference
        
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Zone slider
        ax_zone = plt.axes([0.15, 0.15, 0.7, 0.03])
        self.zone_slider = Slider(ax_zone, 'Zone', 0, 24, 
                                  valinit=self.current_zone, valstep=1)
        self.zone_slider.on_changed(self.update_zone)
        
        # X center textbox
        ax_x = plt.axes([0.15, 0.10, 0.1, 0.03])
        self.x_box = TextBox(ax_x, 'X', initial=str(self.x_center))
        self.x_box.on_submit(self.update_x)
        
        # Y center textbox
        ax_y = plt.axes([0.35, 0.10, 0.1, 0.03])
        self.y_box = TextBox(ax_y, 'Y', initial=str(self.y_center))
        self.y_box.on_submit(self.update_y)
        
        # Box size textbox
        ax_box = plt.axes([0.55, 0.10, 0.1, 0.03])
        self.box_box = TextBox(ax_box, 'Box', initial=str(self.box_size))
        self.box_box.on_submit(self.update_box_size)
        
        # Reset button
        ax_reset = plt.axes([0.75, 0.10, 0.1, 0.03])
        self.reset_button = Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(self.reset_view)
        
        # Load and display
        self.load_zone(self.current_zone)
        self.update_display()
        
    def load_zone(self, zone):
        """Load one channel for the specified zone"""
        img_path = f"{BASE_DIR}/channel_{CHANNEL}/msg_c{CHANNEL:02d}_z{zone:02d}.img"
        print(f"Loading: {img_path}")
        
        try:
            self.image_data = self.load_image(img_path)
            print(f"Success! Image size: {self.image_data['nx']} x {self.image_data['ny']}")
            print(f"Data range: {self.image_data['data'].min():.2f} to {self.image_data['data'].max():.2f}")
        except Exception as e:
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()
            self.image_data = None
    
    def load_image(self, filepath):
        """Load a single image file"""
        print(f"Opening file: {filepath}")
        
        with open(filepath, 'r') as f:
            # Read title
            title = f.readline().strip()
            print(f"  Title: {title}")
            
            # Read dimensions
            dim_line = f.readline().strip()
            print(f"  Dim line: '{dim_line}'")
            nx, ny = [int(x) for x in dim_line.split()]
            print(f"  Dimensions: nx={nx}, ny={ny}")
            
            # Read all data
            imgdata = []
            line_count = 0
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    values = [float(x) for x in line.split()]
                    imgdata.extend(values)
                    line_count += 1
            
            print(f"  Read {line_count} lines with {len(imgdata)} total values")
            print(f"  Expected: {nx * ny} values")
            
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
    
    def update_display(self):
        """Update display"""
        # Remove old colorbar BEFORE clearing axes
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except (KeyError, ValueError):
                pass  # Already removed
            self.colorbar = None
        
        self.ax.clear()
        
        if self.image_data is None:
            self.ax.text(0.5, 0.5, 'Failed to load image', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
            return
        
        # Get data
        data = self.image_data['data']
        zoom_data = self.get_zoom_region(data)
        
        # Display
        im = self.ax.imshow(zoom_data, origin='lower', cmap='gray')
        self.ax.set_title(f"Channel {CHANNEL} - Zone {self.current_zone:02d}")
        self.ax.axis('off')
        self.colorbar = plt.colorbar(im, ax=self.ax)
        
        self.fig.suptitle(f'Center: ({self.x_center}, {self.y_center}) | Box: {self.box_size}px')
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

if __name__ == '__main__':
    print(f"Simple Viewer - Channel {CHANNEL}")
    print("Testing image loading...")
    viewer = SimpleViewer()
    plt.show()


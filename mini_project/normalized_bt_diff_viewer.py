"""
Normalized Brightness-Temperature Difference Viewer

Plots normalized brightness temperature differences for scientifically
relevant MSG channel pairs while retaining the interaction style of the
multi-channel viewer.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button, CheckButtons

BASE_DIR = 'data'

# Channel wavelengths (microns) for brightness temperature conversion
WAVELENGTHS = {
    1: None,    # Visible - no conversion
    2: None,    # Near-IR - keep as radiance
    3: None,    # Near-IR
    4: 3.92,    # IR window
    5: 6.31,    # Upper troposphere water vapour
    6: 7.36,    # Mid troposphere water vapour
    7: 8.71,    # IR window
    8: 9.67,    # Ozone-sensitive
    9: 10.79,   # IR window
    10: 11.94,  # IR window
    11: 13.35   # CO2 - cloud height
}

# Metadata describing physically relevant pairs
CHANNEL_PAIRS = [
    {
        'channels': (4, 10),
        'description': 'Nighttime/low cloud & forest fires vs surface temperature'
    },
    {
        'channels': (5, 6),
        'description': 'Upper vs mid-troposphere water vapour'
    },
    {
        'channels': (7, 9),
        'description': 'Thin cirrus discrimination vs surface temperature'
    },
    {
        'channels': (8, 10),
        'description': 'Ozone absorption vs IR window'
    },
    {
        'channels': (9, 11),
        'description': 'Cloud detection vs cloud height'
    }
]


def normalized_bt_diff(bt_a, bt_b):
    """Return raw and normalized brightness-temperature differences."""
    diff = bt_a - bt_b
    denom = bt_a + bt_b
    norm = np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0)
    return diff, norm


class NormalizedBTDiffViewer:
    """Viewer for normalized brightness-temperature differences between MSG pairs."""

    def __init__(self):
        # Initial parameters
        self.current_zone = 12
        self.x_center = 300
        self.y_center = 800
        self.box_size = 100
        self.show_normalized = True

        # Storage
        self.images = {}  # {channel: image_data}
        self.colorbars = []  # Store all colorbars
        self.hist_buttons = []  # Store histogram button references
        self.current_display_data = {}  # Store currently displayed data for histograms

        # 3x2 grid holds the five channel pairs comfortably
        self.fig = plt.figure(figsize=(12, 9))
        gs = self.fig.add_gridspec(3, 2, left=0.05, right=0.98, top=0.98,
                                   bottom=0.14, hspace=0.40, wspace=0.20)

        self.axes = []

        for i, _ in enumerate(CHANNEL_PAIRS):
            row = i // 2
            col = i % 2
            ax = self.fig.add_subplot(gs[row, col])
            ax.axis('off')
            self.axes.append(ax)
            self.hist_buttons.append(None)

        # Control widgets (identical layout to base viewer)
        ax_zone = plt.axes([0.12, 0.09, 0.76, 0.018])  # type: ignore
        self.zone_slider = Slider(ax_zone, 'Zone', 0, 24,
                                  valinit=self.current_zone, valstep=1)
        self.zone_slider.on_changed(self.update_zone)

        ax_x = plt.axes([0.12, 0.045, 0.1, 0.022])  # type: ignore
        self.x_box = TextBox(ax_x, 'X Center', initial=str(self.x_center))
        self.x_box.on_submit(self.update_x)

        ax_y = plt.axes([0.32, 0.045, 0.1, 0.022])  # type: ignore
        self.y_box = TextBox(ax_y, 'Y Center', initial=str(self.y_center))
        self.y_box.on_submit(self.update_y)

        ax_box = plt.axes([0.52, 0.045, 0.1, 0.022])  # type: ignore
        self.box_box = TextBox(ax_box, 'Box Size', initial=str(self.box_size))
        self.box_box.on_submit(self.update_box_size)

        ax_reset = plt.axes([0.72, 0.045, 0.16, 0.022])  # type: ignore
        self.reset_button = Button(ax_reset, 'Reset View')
        self.reset_button.on_clicked(self.reset_view)

        ax_toggle = plt.axes([0.12, 0.008, 0.25, 0.022])  # type: ignore
        self.temp_check = CheckButtons(ax_toggle, ['Show Normalized ΔT'],
                                       [self.show_normalized])
        self.temp_check.on_clicked(self.toggle_normalized_display)

        # Load and display
        self.load_zone(self.current_zone)
        self.update_display()

    def load_zone(self, zone):
        """Load all channels required for the specified zone."""
        print(f"\nLoading zone {zone:02d}...")
        self.images = {}

        for channel in range(1, 12):
            img_path = f"{BASE_DIR}/channel_{channel}/msg_c{channel:02d}_z{zone:02d}.img"
            try:
                self.images[channel] = self.load_image(img_path)
                print(f"  Ch{channel}: ✓ ({self.images[channel]['nx']}x{self.images[channel]['ny']})")
            except Exception as e:  # pragma: no cover - file issues handled at runtime
                print(f"  Ch{channel}: ✗ ({e})")
                self.images[channel] = None

        loaded_count = len([v for v in self.images.values() if v is not None])
        print(f"Loaded {loaded_count}/11 channels")

    def load_image(self, filepath):
        """Load a single image file - using proven working method"""
        with open(filepath, 'r') as f:
            title = f.readline().strip()
            nx, ny = [int(x) for x in f.readline().split()]
            imgdata = []
            for line in f:
                line = line.strip()
                if line:
                    imgdata.extend([float(x) for x in line.split()])

            data = np.array(imgdata).reshape((ny, nx))

        return {'data': data, 'nx': nx, 'ny': ny, 'title': title}

    def brightness_temperature(self, radiance, wavelength):
        """Convert radiance to brightness temperature"""
        if wavelength is None:
            return radiance
        radiance = np.clip(radiance, 1e-6, None)
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
        """Update channel-pair displays"""
        for cbar in self.colorbars:
            try:
                cbar.remove()
            except (KeyError, ValueError):
                pass
        self.colorbars = []
        self.current_display_data = {}

        for i, pair in enumerate(CHANNEL_PAIRS):
            ax = self.axes[i]
            ax.clear()
            ax.axis('off')
            ch_a, ch_b = pair['channels']
            pair_key = (ch_a, ch_b)

            img_a = self.images.get(ch_a)
            img_b = self.images.get(ch_b)
            if img_a is None or img_b is None:
                ax.text(0.5, 0.5,
                        f'Ch{ch_a} / Ch{ch_b}\nNot Found',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10)
                continue

            zoom_a = self.get_zoom_region(img_a['data'])
            zoom_b = self.get_zoom_region(img_b['data'])

            wavelength_a = WAVELENGTHS[ch_a]
            wavelength_b = WAVELENGTHS[ch_b]
            bt_a = self.brightness_temperature(zoom_a, wavelength_a)
            bt_b = self.brightness_temperature(zoom_b, wavelength_b)
            diff, norm = normalized_bt_diff(bt_a, bt_b)
            diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

            display_data = norm if self.show_normalized else diff
            cmap = 'RdBu_r'
            vmin, vmax = (-1, 1) if self.show_normalized else (np.min(diff), np.max(diff))
            if vmin == vmax:
                vmin -= 1e-6
                vmax += 1e-6

            title = (f'Ch{ch_a} ({wavelength_a}μm) vs Ch{ch_b} ({wavelength_b}μm)\n'
                     f'{pair["description"]}')
            im = ax.imshow(display_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=8, pad=4)

            stats_text = (f'ΔT mean: {diff.mean():.2f}K\n'
                          f'ΔT min/max: {diff.min():.2f}/{diff.max():.2f}K')
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                    fontsize=7, color='white', va='bottom',
                    bbox=dict(facecolor='black', alpha=0.4, boxstyle='round'))

            cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label('Normalized ΔT' if self.show_normalized else 'ΔT [K]', fontsize=8)
            self.colorbars.append(cbar)

            self.current_display_data[pair_key] = {
                'norm_data': norm,
                'diff_data': diff,
                'title': title,
                'zone': self.current_zone,
                'channels': pair_key
            }

            bbox = ax.get_position()
            button_ax = plt.axes([bbox.x0 + bbox.width*0.2, bbox.y0 - 0.045,
                                  bbox.width*0.6, 0.020])  # type: ignore
            hist_btn = Button(button_ax, 'Histogram', color='#87CEEB', hovercolor='#4682B4')
            hist_btn.label.set_fontsize(9)
            hist_btn.on_clicked(lambda event, key=pair_key: self.show_histogram(key))
            self.hist_buttons[i] = hist_btn

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
        self.x_center = 300
        self.y_center = 800
        self.box_size = 100
        self.x_box.set_val(str(self.x_center))
        self.y_box.set_val(str(self.y_center))
        self.box_box.set_val(str(self.box_size))
        self.update_display()

    def toggle_normalized_display(self, label):
        self.show_normalized = not self.show_normalized
        self.update_display()

    def show_histogram(self, pair_key):
        """Show histogram for a specific channel pair in a new window."""
        if pair_key not in self.current_display_data:
            print(f"No data available for channel pair {pair_key}")
            return

        data_info = self.current_display_data[pair_key]
        norm_data = data_info['norm_data']
        diff_data = data_info['diff_data']
        ch_a, ch_b = data_info['channels']

        hist_fig, hist_ax = plt.subplots(figsize=(8, 6))
        hist_fig.canvas.manager.set_window_title(
            f'Channel Pair {ch_a}-{ch_b} Histogram')  # type: ignore

        norm_flat = norm_data.flatten()
        hist_ax.hist(norm_flat, bins=50, color='steelblue',
                     edgecolor='black', alpha=0.7)

        hist_ax.set_xlabel('Normalized ΔT', fontsize=12)
        hist_ax.set_ylabel('Frequency', fontsize=12)
        hist_ax.set_title(
            f'{data_info["title"]}\nZone {data_info["zone"]:02d} | '
            f'Center: ({self.x_center}, {self.y_center}) | '
            f'Box: {self.box_size}px',
            fontsize=11, fontweight='bold')
        hist_ax.grid(True, alpha=0.3)

        diff_flat = diff_data.flatten()
        stats_text = (
            f'ΔT mean: {diff_flat.mean():.2f}K\n'
            f'ΔT std: {diff_flat.std():.2f}K\n'
            f'ΔT min/max: {diff_flat.min():.2f}/{diff_flat.max():.2f}K\n'
            f'Norm min/max: {norm_flat.min():.2f}/{norm_flat.max():.2f}\n'
            f'Pixels: {len(norm_flat)}'
        )

        hist_ax.text(0.98, 0.97, stats_text, transform=hist_ax.transAxes,
                     fontsize=10, va='top', ha='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    print("Normalized Brightness-Temperature Difference Viewer")
    print("=" * 50)
    print("\nControls:")
    print("  - Zone Slider: Select time/zone (z00-z24)")
    print("  - X/Y Center: Set center of zoom region")
    print("  - Box Size: Set size of zoom region (pixels)")
    print("  - Reset View: Return to full image")
    print("  - Show Normalized ΔT: Toggle normalized vs raw ΔT display")
    print("  - Histogram buttons: Click to see histogram of each channel pair")
    print("\nLoading initial data...")

    viewer = NormalizedBTDiffViewer()
    plt.show()

"""
RGB Visualisation Viewer

Builds canonical RGB composites (natural colour, cloud microphysics, etc.)
from the MSG channel set using radiance/brightness-temperature combinations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

BASE_DIR = 'data'

# Channel wavelengths (microns) used for brightness-temperature conversion
WAVELENGTHS = {
    1: None,
    2: None,
    3: None,
    4: 3.92,
    5: 6.31,
    6: 7.36,
    7: 8.71,
    8: 9.67,
    9: 10.79,
    10: 11.94,
    11: 13.35,
}

# Helper ranges used for normalising the composites
DEFAULT_REFLECTANCE_RANGE = (0.0, 1.0)
DEFAULT_BT_RANGE = (200.0, 320.0)


COMPOSITES = [
    {
        'name': 'Natural Colour RGB',
        'description': 'True-colour style daytime composite',
        'components': [
            {'type': 'reflectance', 'channel': 3, 'range': (0.0, 1.2)},
            {'type': 'reflectance', 'channel': 2, 'range': (0.0, 1.0)},
            {'type': 'reflectance', 'channel': 1, 'range': (0.0, 1.0)},
        ],
        'notes': 'Thick clouds white, snow cyan, vegetation green, land/sea dark.'
    },
    {
        'name': 'Clouds RGB',
        'description': 'Simple daytime cloud mask',
        'components': [
            {'type': 'reflectance', 'channel': 1, 'range': (0.0, 1.0)},
            {'type': 'reflectance', 'channel': 2, 'range': (0.0, 1.0)},
            {'type': 'bt', 'channel': 9, 'range': DEFAULT_BT_RANGE, 'invert': True},
        ],
        'notes': 'Cold high clouds bright cyan/white; clear land/sea dark.'
    },
    {
        'name': 'Day Microphysics RGB',
        'description': 'Cloud phase + droplet size (daytime)',
        'components': [
            {'type': 'reflectance', 'channel': 2, 'range': (0.0, 1.0)},
            # Ch4 "solar component" approximated via brightness temperature inversion
            {'type': 'bt', 'channel': 4, 'range': (220.0, 330.0), 'invert': True},
            {'type': 'bt', 'channel': 9, 'range': DEFAULT_BT_RANGE, 'invert': True},
        ],
        'notes': 'Water clouds greenish, ice clouds magenta, cold anvils bright, warm land dark blue.'
    },
    {
        'name': 'Night Microphysics RGB',
        'description': 'Low cloud / fog / phase at night',
        'components': [
            {'type': 'bt_diff', 'channels': (10, 9), 'range': (-4.0, 2.0)},
            {'type': 'bt_diff', 'channels': (9, 4), 'range': (-4.0, 6.0)},
            {'type': 'bt', 'channel': 9, 'range': DEFAULT_BT_RANGE, 'invert': True},
        ],
        'notes': 'Low water clouds aqua, mid clouds pink, high ice clouds red/orange, clear land dark.'
    },
    {
        'name': '24-hour Microphysics RGB',
        'description': 'Phase + thickness day & night',
        'components': [
            {'type': 'bt_diff', 'channels': (10, 9), 'range': (-4.0, 2.0)},
            {'type': 'bt_diff', 'channels': (9, 7), 'range': (-4.0, 5.0)},
            {'type': 'bt', 'channel': 9, 'range': DEFAULT_BT_RANGE, 'invert': True},
        ],
        'notes': 'Thin cirrus dark, thick high ice bright, low clouds green/yellow, surface dark blue.'
    },
    {
        'name': '24-hour Microphysics (Dust)',
        'description': 'Dust-sensitive variant of 24h microphysics',
        'components': [
            {'type': 'bt_diff', 'channels': (10, 9), 'range': (-4.0, 2.0)},
            {'type': 'bt_diff', 'channels': (9, 7), 'range': (-0.5, 15.0)},
            {'type': 'bt', 'channel': 9, 'range': (261.0, 289.0), 'invert': True},
        ],
        'notes': 'Dust magenta, thick ice dark red, low clouds yellow/green, clear land/sea blue.'
    }
]


class RGBCompositeViewer:
    def __init__(self):
        self.current_zone = 12
        self.x_center = 300
        self.y_center = 800
        self.box_size = 100

        self.images = {}

        self.fig = plt.figure(figsize=(13, 9))
        gs = self.fig.add_gridspec(3, 2, left=0.04, right=0.99, top=0.98,
                                   bottom=0.14, hspace=0.40, wspace=0.15)

        self.axes = []
        for i in range(len(COMPOSITES)):
            row = i // 2
            col = i % 2
            ax = self.fig.add_subplot(gs[row, col])
            ax.axis('off')
            self.axes.append(ax)

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

        self.load_zone(self.current_zone)
        self.update_display()

    def load_zone(self, zone):
        print(f"\nLoading zone {zone:02d}...")
        self.images = {}
        for channel in range(1, 12):
            img_path = f"{BASE_DIR}/channel_{channel}/msg_c{channel:02d}_z{zone:02d}.img"
            try:
                self.images[channel] = self.load_image(img_path)
                print(f"  Ch{channel}: ✓ ({self.images[channel]['nx']}x{self.images[channel]['ny']})")
            except Exception as exc:  # pragma: no cover
                print(f"  Ch{channel}: ✗ ({exc})")
                self.images[channel] = None
        loaded_count = len([v for v in self.images.values() if v is not None])
        print(f"Loaded {loaded_count}/11 channels")

    def load_image(self, filepath):
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
        if wavelength is None:
            return radiance
        radiance = np.clip(radiance, 1e-6, None)
        H = 6.63e-34
        C = 3.00e8
        K = 1.38e-23
        R1 = H * C / K
        R2 = 2 * H * C**2
        w = wavelength * 1.0e-6
        temp = R1 / w / np.log(1.0 + R2 / (w**5 * radiance * 1e6))
        return temp

    def get_zoom_region(self, data):
        ny, nx = data.shape
        half_box = self.box_size // 2
        x_min = max(0, self.x_center - half_box)
        x_max = min(nx, self.x_center + half_box)
        y_min = max(0, self.y_center - half_box)
        y_max = min(ny, self.y_center + half_box)
        return data[y_min:y_max, x_min:x_max]

    def normalize(self, data, value_range):
        arr = np.array(data, dtype=float)
        if value_range is None:
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
        else:
            vmin, vmax = value_range
        if vmax - vmin == 0:
            return np.zeros_like(arr)
        scaled = (arr - vmin) / (vmax - vmin)
        return np.clip(scaled, 0.0, 1.0)

    def component_reflectance(self, channel, comp_range, gamma=1.0):
        img = self.images.get(channel)
        if img is None:
            raise ValueError(f"Channel {channel} not loaded")
        data = self.get_zoom_region(img['data'])
        scaled = self.normalize(data, comp_range or DEFAULT_REFLECTANCE_RANGE)
        if gamma != 1.0:
            scaled = np.clip(scaled, 0, 1) ** (1.0 / gamma)
        return scaled

    def component_bt(self, channel, comp_range, invert=False):
        img = self.images.get(channel)
        if img is None:
            raise ValueError(f"Channel {channel} not loaded")
        data = self.get_zoom_region(img['data'])
        bt = self.brightness_temperature(data, WAVELENGTHS[channel])
        scaled = self.normalize(bt, comp_range or DEFAULT_BT_RANGE)
        if invert:
            scaled = 1.0 - scaled
        return scaled

    def component_bt_diff(self, channels, comp_range, invert=False):
        ch_a, ch_b = channels
        img_a = self.images.get(ch_a)
        img_b = self.images.get(ch_b)
        if img_a is None or img_b is None:
            raise ValueError(f"Channel pair {channels} not loaded")
        zoom_a = self.get_zoom_region(img_a['data'])
        zoom_b = self.get_zoom_region(img_b['data'])
        bt_a = self.brightness_temperature(zoom_a, WAVELENGTHS[ch_a])
        bt_b = self.brightness_temperature(zoom_b, WAVELENGTHS[ch_b])
        diff = bt_a - bt_b
        scaled = self.normalize(diff, comp_range)
        if invert:
            scaled = 1.0 - scaled
        return scaled

    def build_composite(self, recipe):
        channels = []
        for comp in recipe['components']:
            if comp['type'] == 'reflectance':
                channels.append(self.component_reflectance(comp['channel'], comp.get('range'),
                                                          comp.get('gamma', 1.0)))
            elif comp['type'] == 'bt':
                channels.append(self.component_bt(comp['channel'], comp.get('range'),
                                                  comp.get('invert', False)))
            elif comp['type'] == 'bt_diff':
                channels.append(self.component_bt_diff(comp['channels'], comp.get('range'),
                                                       comp.get('invert', False)))
            else:
                raise ValueError(f"Unsupported component type {comp['type']}")
        rgb = np.dstack(channels)
        return np.clip(rgb, 0.0, 1.0)

    def update_display(self):
        for ax in self.axes:
            ax.clear()
            ax.axis('off')

        for idx, recipe in enumerate(COMPOSITES):
            ax = self.axes[idx]
            try:
                rgb_image = self.build_composite(recipe)
                ax.imshow(rgb_image, origin='lower')
            except ValueError as exc:
                ax.text(0.5, 0.5, str(exc), ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='red')
                continue

            title = f"{recipe['name']}\n{recipe['description']}"
            ax.set_title(title, fontsize=9, pad=4)
            ax.text(0.02, 0.03, recipe['notes'], fontsize=7,
                    transform=ax.transAxes, wrap=True,
                    bbox=dict(facecolor='black', alpha=0.35, boxstyle='round'),
                    color='white')

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


if __name__ == '__main__':
    print("RGB Composite Visualisation")
    print("=" * 50)
    print("\nControls:")
    print("  - Zone Slider: Select time/zone (z00-z24)")
    print("  - X/Y Center: Set center of zoom region")
    print("  - Box Size: Set size of zoom region (pixels)")
    print("  - Reset View: Return to default coordinates")
    print("\nLoading initial data...")

    viewer = RGBCompositeViewer()
    plt.show()

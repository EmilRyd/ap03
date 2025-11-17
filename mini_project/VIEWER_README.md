# Interactive Multi-Channel Satellite Image Viewer

## Overview

This interactive viewer allows you to explore all 11 satellite channels simultaneously across different time zones with dynamic zoom capabilities.

## Features

✅ **All 11 channels displayed in a 3×4 grid**  
✅ **Zone/time slider** (z00 through z24)  
✅ **Interactive zoom controls** (x, y, box size)  
✅ **Brightness temperature toggle** for infrared channels  
✅ **Real-time updates** when parameters change  

## Usage

Run the viewer:
```bash
cd mini_project
python3 interactive_viewer.py
```

## Controls

### Zone Slider
- **Location**: Bottom of window
- **Function**: Select which time zone to view (0-24)
- **Usage**: Drag the slider or click on the track
- **Effect**: Loads and displays all 11 channels for the selected zone

### X Center, Y Center, Box Size
- **Location**: Text boxes below the slider
- **Function**: Control zoom region
- **Usage**: 
  - Type a value and press Enter
  - X Center: horizontal position (0-3712)
  - Y Center: vertical position (0-3712)
  - Box Size: size of zoom window in pixels
- **Effect**: All 11 channel images update to show the specified region

### Reset View Button
- **Location**: Right of the text boxes
- **Function**: Return to full image view
- **Usage**: Click the button
- **Effect**: Resets to center (1856, 1856) with full box size (3712)

### Show Brightness Temp Checkbox
- **Location**: Bottom left
- **Function**: Toggle between radiance and brightness temperature for IR channels
- **Usage**: Click to check/uncheck
- **Effect**: 
  - ✅ Checked: IR channels (4-11) show brightness temperature in Kelvin
  - ☐ Unchecked: All channels show radiance values

## Channel Information

| Channel | Wavelength (μm) | Type | Description |
|---------|----------------|------|-------------|
| 1 | Visible | VIS | Visible light (always radiance) |
| 2 | Near-IR | NIR | Near infrared (radiance only) |
| 3 | Near-IR | NIR | Near infrared (radiance only) |
| 4 | 3.9 | IR | IR window |
| 5 | 6.2 | IR | Water vapor |
| 6 | 7.3 | IR | Water vapor |
| 7 | 8.7 | IR | IR window |
| 8 | 9.7 | IR | Ozone |
| 9 | 10.79 | IR | IR window (10.8μm) |
| 10 | 11.94 | IR | IR window (12μm) |
| 11 | 13.4 | IR | IR window |

## Display Details

### Color Schemes

**Visible/Near-IR Channels (1-3):**
- Grayscale
- Shows radiance values only

**Infrared Channels (4-11):**

*Brightness Temperature Mode (default):*
- Color scale: white → magenta → blue → cyan → green → yellow → red → black
- Range: 230-320 K
- Colorbar labeled "Temp [K]"

*Radiance Mode:*
- Grayscale
- Shows raw radiance values
- Colorbar labeled "Radiance"

### Grid Layout

```
┌─────────┬─────────┬─────────┬─────────┐
│  Ch1    │  Ch2    │  Ch3    │  Ch4    │
│ (VIS)   │ (NIR)   │ (NIR)   │ (3.9μm) │
├─────────┼─────────┼─────────┼─────────┤
│  Ch5    │  Ch6    │  Ch7    │  Ch8    │
│ (6.2μm) │ (7.3μm) │ (8.7μm) │ (9.7μm) │
├─────────┼─────────┼─────────┼─────────┤
│  Ch9    │  Ch10   │  Ch11   │ (empty) │
│(10.8μm) │ (12μm)  │ (13.4μm)│         │
└─────────┴─────────┴─────────┴─────────┘
```

## Example Workflows

### 1. Browse Through Time
1. Keep zoom at default (full image)
2. Move the zone slider left/right to see how the scene changes over time
3. All 11 channels update simultaneously

### 2. Focus on a Specific Region
1. Select your zone of interest
2. Type X and Y coordinates in the text boxes (e.g., X=1000, Y=1500)
3. Set a smaller box size (e.g., 500)
4. Press Enter after each value
5. All channels zoom into that region

### 3. Compare Radiance vs Temperature
1. Select a zone and zoom region
2. Uncheck "Show Brightness Temp" to see raw radiance
3. Check it again to see temperature
4. Notice how cloud patterns and surface features appear different

### 4. Identify Features Across Channels
1. Look for clouds (bright in visible, cold in IR)
2. Look for warm surfaces (warm in IR channels 9-11)
3. Compare water vapor channels (5, 6) with window channels (9, 10, 11)

## Tips

- **Start with zone 12** (z12) - this is the default and usually has good data
- **Use Reset View** to quickly return to full image when you get lost
- **Box sizes**:
  - Full image: 3712 pixels
  - Large region: 2000 pixels
  - Medium region: 1000 pixels (default)
  - Small region: 500 pixels
  - Detailed view: 200 pixels
- **Coordinate tips**:
  - (0, 0) is bottom-left corner
  - (3712, 3712) is top-right corner
  - (1856, 1856) is center
- **Performance**: Loading a new zone takes a few seconds. Be patient!

## Troubleshooting

**"Ch X Not Found" messages:**
- Some channel/zone combinations may not exist in your data directory
- Check that all channel folders contain the expected .img files

**Images look strange:**
- Try toggling brightness temperature mode
- Check that you're within valid coordinate ranges (0-3712)
- Reset view to return to defaults

**Viewer is slow:**
- This is normal when loading all 11 channels
- Each zone change loads ~11 images
- Zoom changes are faster since data is already loaded

## Technical Details

**Memory Usage:**
- Only one zone loaded at a time (~11 images × ~13MB each ≈ 145MB)
- Zoom operations don't reload data, just crop the display

**Update Triggers:**
- Zone slider: Loads new zone, updates display
- X/Y/Box size: Updates display only (no new loading)
- Brightness temp toggle: Converts data, updates display

**File Paths:**
- Uses relative path: `data/channel_X/` (relative to mini_project folder)
- Script expects to be run from the mini_project directory
- Modify `BASE_DIR` in the script if your data structure is different


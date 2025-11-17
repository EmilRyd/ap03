# Standard modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

""" Various utility modules for ap03.py 

CONTENTS
  inputxy      : Input new x,y coordinates to move cursor box
  inputy       : Input new y coordinate to move selected image line
  movepix      : Check if selected menu option was to move multiple pixels 
  pixloc       : Text string containing pixel location
  plot_colours : Colours used for each channel's plots
  read_numbers : Read set of numbers from terminal
  tem_colours  : Rainbow colour scale for temperature images 

HISTORY
  v31Aug20 : AD Original version

"""

def inputxy(origxy):
  """ Input new x,y coordinates to move cursor box

  PARAMETERS
    origxy int : current (x,y) coordinates

  RETURNS
    ix,iy  int : new x,y coordinates

  DESCRIPTION
    Used for menu option where the user is given the choice of moving the 
    cursor box directly to a particular location. If the entry is just <CR>,
    returns current (x,y) instead. Assumes external checks for x,y within image
  """

  (ix,iy) = read_numbers(2,"Enter x,y coordinates: ")
  if np.isnan(ix):  return origxy
  else: return (ix,iy)

def inputy(origy):
  """ Input new y coordinate to move selected image line
   
  PARAMETERS
    origy int : current y-coordinate

  DESCRIPTION
    Used for image line analysis meny to move selected image line.
    If supplied y-value is outside image limits, or the entered new y values is
    otherwise invalid, this will return original vale
  """

  iyinp = input("Enter y coordinate: ")
  try: 
    return int(iyinp)
  except:
    return origy
 
def movepix(opt):
  """ Number of pixels to move

  PARAMETERS
    opt str : user-typed option, as upper case

  RETURNS
    n   int : number of pixels to move

  DESCRIPTION
    Called if user typed Un,Dn,Ln or Rn, returns n as an integer.
    If opt is just a single character (assumed U,D,L or R) returns 1.
    If n cannot be interpreted as an integer, returns 0.
  """

  if len(opt) == 1:
    return 1
  else:
    try:
      return int(opt[1:])
    except:
      return 0

def pixloc(ix,iy,geo):
  """ Text string containing pixel location

  PARAMETERS
    ix  int : pixel x-coordinate
    iy  int : pixel y-coordinate
    geo obj : Geometric calibration data and methods 

  RETURNS
    str     : Text string "(X,Y): ... (Lat,Lon): ...."

  DESCRIPTION
    Prints x,y coord and, if Geo calibration is set, lat,lon as well
    For use as part of menus showing current location
  """

  rec = "(X,Y):{:4n}{:4n}".format(ix,iy)    # basic (X,Y)... string
  if geo.cal:
    lat, lon, zen = geo.locate(ix,iy)
    if np.isfinite(lat):                    # add (Lat,Lon) ... string
      rec = rec + "  (Lat,Lon): {:6.2f}{:7.2f}".format(lat,lon)
  return rec

plot_colours = { "CV":"gray", "C9":"blue", "T9":"blue", "C10":"red", "T10":"red",
                 "TS":"green" }
"""  Colours used for each channel's plots

    Dictionary of colours for histogram/line plots according to image label
    Suggest using as col = plot_colours.get(label,'gray') so that 'gray' is 
    returned if the label isn't in the dictionary 
"""


def read_numbers ( nnum, prompt ):
  """ Read set of real numbers from terminal 

  PARAMETERS
    nnum   int : Number of numbers expected
    prompt str : Prompt printed to terminal
  
  RETURNS
    Tuple of nnum values, either all float or all np.nan

  DESCRIPTION
    The entered numbers may be separated by commas and/or spaces
    This always returns a list of nnum values
    If the user supplies more/fewer than nnum values the prompt is repeated
    If the user just types <CR> or supplies an unreadable value, this returns
    an array full of np.nan values
  """

  nflds = 0
  while nflds != nnum:
    rec = input(prompt)
    if rec == '': return np.full(nnum,np.nan)
    rec = rec.replace(',', ' ')  # replace any commas with whitespace
    flds = rec.split()           # split response into separate flds
    nflds = len(flds)
    if nflds != nnum:
      print(" *** Enter " + str(nnum) + " numbers, separated by spaces ***")
  values = []
  for f in flds:
    try:    
      values.append( float(f) )
    except: 
      return np.full(nnum,np.nan)
  return values


def tem_colours ( ):
  """ Rainbow colour scale for temperature images 

  RETURNS
      norm obj : matplotlib.colors.Normalize
      cmap obj : matplotlob.colors.LinearSegmentedColormap

  DESCRIPTION
      Returns data required to covers range 230 (white-violet...) to 320 
      (...red-black). This starts with colours as a set of (R,G,B) values
      (white ... black) which are then interpolated to range 230-320.
  """

  colours = [(1,1,1),(1,0,1),(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0),(0,0,0)]
  norm = plt.Normalize(230,320)
  cmap = LinearSegmentedColormap.from_list('tem_colours', colours)
  return ( norm, cmap )

# end of ap03_utils ---------------------------------------------------------------------------

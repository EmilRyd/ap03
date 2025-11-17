# Standard modules
import math
import time
import numpy as np
import matplotlib.pyplot as plt

# Local modules
from ap03_utils import inputy       # Input new x,y coordinates to move cursor box
from ap03_utils import movepix      # Number of pixels to move
from ap03_utils import pixloc       # Text string containing pixel location
from ap03_utils import plot_colours # Colours used for each channel's plots

# Local constants
# Assume Window#1 reserved for displaying full disk image
WINDOW_LIN = 2  # Window# for displaying line plot

class Lin:
  """ Image line analysis data and methods 

  DATA 
    No data associated with this Class

  METHODS
    __init__ : Initialise new object
      menu   : Screen menu for Image Line Analysis part of practical
      plot   : Plot image line

  USAGE
    Called once at the start of AP03 to initialise a Lin object
  
  HISTORY
    v02Sep20 : AD Original version
  """

  def __init__(self):
    """ Initialise new object  """

    pass     # nothing to initialise

  def menu ( self, iy, img, geo ):
    """ Screen menu for Image Line Analysis part of practical

    PARAMETERS
      iy  int : Image row number (y-coordinate)
      img obj : Full disk image data and methods 
      geo obj : Geometric calibration data and methods

    DESCRIPTION
      Cycles through the Line Analysis options until the user types 'X'
      The image and line plot are updated each time. 
      The line plot is displayed on a longitude axis so it is necessary for the
      geometric calibration to be applied before this section is called.
    """ 

    if not geo.cal:
      print(" *** Image Line option can only be selected after GeoCalibration ")
      time.sleep(3)
      return

    pfile = False            # True = save line plot to a file
    dfile = False            # True = save line plot data to a file
    opt = ''                 # initially re-plot current displayed image
    ix = img.nx // 2         # Set nominal x-coordinate to centre of line

    while opt != 'X':        # Loop over menu until 'X' selected
      iy = int ( np.median ( [ 1, iy, img.ny-2 ] ) )
      img.disp(opt,lin=iy)   # Update main image display
      self.plot(iy,img,geo,pfile=pfile,dfile=dfile)
      pfile = False          # Reset to False if necessary
      dfile = False

      print("Main> Lin Menu>")
      print("    Image line centre " + pixloc(ix,iy,geo) )
      print("    --- Image Selection ---------------------")
      print("      I - Select another image")
      print("    --- Cursor Positioning ------------------")
      print("      P - Select scan line using mouse")
      print("      U - Move up one line ")
      print("      D - Move down one line ")
      print("      G - Go to specific line#  ")
      print("    --- Output Selection --------------------")
      print("      F - Save graph plot as file")
      print("      O - Save graph values to file")
      print("    -----------------------------------------")
      print("      X - Return to Main Menu")
      print("    -----------------------------------------")
      opt = input ( "    Enter Option: " ).upper()

      # opt may also be a direct image selection eg 'CV'
      if   opt == 'I': opt = img.menu()
      elif opt == 'P': ix,iy = img.click()
      elif opt[0] == 'U': iy += movepix(opt)
      elif opt[0] == 'D': iy -= movepix(opt)
      elif opt == 'G': iy = inputy((iy))
      elif opt == 'F': pfile=True
      elif opt == 'O': dfile=True

  def plot(self,iy,img,geo,pfile=False,dfile=False):
    """ Plot image line 

    PARAMETERS
      iy    int : Image row number (y-coordinate)
      img   obj : Full disk image data and methods 
      geo   obj : Geometric calibration data and methods
      pfile boo : True = Save plot as file
      dfile boo : True = Write plot data to file

    DESCRIPTION
      This extract a line from the full disk and replots it on a uniform
      longitude axis, effectively expanding the edges. A second axis is 
      also constructed from sec(zenith) angle values corresponding to each
      longitude. It takes a lot of code and there are no doubt better ways.
    """

    fig = plt.figure(WINDOW_LIN,figsize=(6,3))
    plt.clf()
    label = img.label
    # use y=1.25 to move the main plot title above the label for the top x-axis
    plt.title(img.desc[label] + " - Line: " + str(iy), y=1.25)
    plt.xlabel("Longitude")
    linfull = img.images[label][iy]  # full extent of image line
    linval = []     # image line truncated between -80:80 Lon
    lonaxs = []     # longitude axis values
    secaxs = []     # sec(zen) axis values
    # construct longitude axis values and corresponding sec(zen)
    for ix in range(img.nx):
      lat, lon, zen = geo.locate(ix,iy)
      if np.isnan(zen): continue  # off the Earth's disk
      linval.append(linfull[ix])
      lonaxs.append(lon)
      rzen = math.radians(zen)  # convert zen from deg to rad
      sec = 1.0/math.cos(rzen)
      secaxs.append(sec)
    # Plot image line values  
    templt = label in ('T9','T10','TS')
    if templt:
      plt.ylabel("Temperature [K]")
      plt.ylim(220,max(linval))
    else:
      plt.ylabel("Pixel Counts")
    plt.plot( lonaxs, linval, color=plot_colours.get(label,"gray") )
    plt.grid(color="silver")
    ax = plt.gca()
    xl = ax.get_xlim()
    # Create upper axis with sec(zen) values
    # construct upper axis of sec(zen) values
    seclst = ( 4, 3, 2, 1.5, 1.2, 1.1, 0.0 ) # sec(zen) tick values
    xtickval = []   # sec(zen) tick value
    xticklon = []   # matching longitude value
    lonprev = 100.0
    secprev = 100.0
    isec = 0
    ix = 0
    # First set axis for left part of image
    # print('len=' + str(len(secaxs)))
    if len(secaxs) > 0:    # may not be set if off edge of image
      while secaxs[ix] < secprev:
        sec = secaxs[ix]
        if sec < seclst[isec]:
          xtv = seclst[isec]
          isec += 1
          dx = (xtv-secprev)/(sec-secprev)
          xtl = lonprev + dx*(lonaxs[ix]-lonprev)
          xtickval.append(xtv)
          xticklon.append(xtl)
        secprev = secaxs[ix]
        lonprev = lonaxs[ix]
        ix+=1
    # construct axis for right part of image by reflection about lon=0
      xtickval = xtickval + xtickval[::-1]
    lonplus = []
    for lon in xticklon:
      lonplus.append(abs(lon))
    xticklon = xticklon + lonplus[::-1]
    ax2 = plt.twiny()
    ax2.set_xlim(xl)
    ax2.set_xlabel("sec(zenith angle)")
    plt.xticks(xticklon,xtickval)
    plt.tight_layout()
    plt.show() 

    if pfile:     # save line plot to a file
      file = input ( "Save to file (<CR>=line.pdf): ") or "line.pdf"
      plt.savefig(file)

    if dfile:     # save line plot data to a file
      file = input ( "Save to file (<CR>=line.txt): ") or "line.txt"
      f = open(file,"w")
      f.write ( "Image=" + img.label + "  Line=" + str(iy) + "\n" )
      f.write ( "    Lon     Sec    Value \n")      
      if label in ( 'CV', 'C9', 'C10' ): 
        for ix in range(len(linval)):
          f.write ( "{:8.2f}{:8.2f}{:8n}\n".format( lonaxs[ix], secaxs[ix], 
                                                      linval[ix]) )
      else:
        for ix in range(len(linval)):
          f.write ( "{:8.2f}{:8.2f}{:8.2f}\n".format( lonaxs[ix], secaxs[ix], 
                                                      linval[ix]) )
      f.close()

# end of class Lin ----------------------------------------------------------------

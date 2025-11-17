# Standard modules
import math
import time
import numpy as np

# Local modules
from box_class  import Box           # Image cursor box data and methods
from ap03_utils import inputxy       # Input new x,y coordinates to move cursor box
from ap03_utils import movepix       # Number of pixels to move
from ap03_utils import pixloc        # Text string containing pixel location
from ap03_utils import read_numbers  # Read set of numbers from terminal

# Local constants
DIST    = 42260.0   # Radial dist [km] of geo.satellite from centre of earth
REARTH  = 6371.0    # Earth radius [km]

class Geo:
  """ Geometric calibration data and methods

  DATA 
    cal    boo : True = Geoetric calibration set
    alpha  flt : y/elevation scale factor
    beta   flt : x/azimuth   scale factor
    x0     flt : x-coordinate of sub-satellite point
    y0     flt : y-coordinate of sub-satellite point
    geofil str : Name of file containing GeoCal data (eg 'geo.txt')

 METHODS
    __init__ : Initialise new Geo object   
      menu   : Screen menu for GeoCal part of practical
      save   : Write GeoCal data to file     
      input  : Read user-input of GeoCal parameters 
      angles : Print Elevation,Azimuth angles for givem Lat,Lon
      locang : Convert ele,azi angles to lat,lon,zen angles
      locate : Convert ix,iy coords to lat,lon,zen angles
      satang : Convert lat,lon angles to ele,azi,zen angles

  USAGE
    Called once at the start to initialise a Geo object

  HISTORY
    v31Aug20 : AD Original version
  """
  
  def __init__(self,geofil):
    """ Initialise new Geo object

    PARAMETERS
      geofil str : name of file containing geo.cal data, eg 'geo.txt' 

    DESCRIPTION
      If the Geo Cal data file exists, the data are read on initialisation
      and geo.cal set True
    """

    self.cal = False
    self.alpha = 0.0
    self.beta  = 0.0
    self.x0  = 0.0
    self.y0  = 0.0
    self.geofil = geofil            # save the filename used for GeoCal data

    try:                            # if file already exists ...
      f = open(geofil,"r")
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

  def menu ( self, ix, iy, img ):
    """ Screen menu for GeoCal part of practical

    PARAMETERS
      ix  int : x-coordinate of centre of cursor box
      iy  int : y-coordinate of centre of cursor box
      img obj : Full disk image data and methods

    DESCRIPTION
      Repeat Geo menu and execute selection until the user selects 'X'
      The displayed images are updated each cycle. The usual sequence is to update
      the cursor box size/shape first, then update the main image display. 
      However, since the two are interdependent, if different main image is 
      selected, this is updated immediately after selection, before the next loop.
    """

    opt = ''
    zoomc = False
    while opt != 'X':
      box = Box(ix,iy,img,self,zoomc=zoomc)
      img.disp('',box=box)
      ix = box.ix
      iy = box.iy
      print("Main> Geo Menu>")
      print("    Cursor Box centre " + pixloc(ix,iy,self) )
      print("    --- Image Selection -------------")
      print("      I - Select another image")
      print("      Z - Toggle Zoom display contrast") 
      print("    --- Cursor Positioning ----------")
      print("      P - Position box using mouse")
      print("      U - Move up one line ")
      print("      D - Move down one line ")
      print("      L - Move left one column ")
      print("      R - Move right one column ") 
      print("      G - Go to specific (x,y)")
      print("    --- Geo Options -----------------")
      print("      A - Calc (ele,azi) from (lat,lon)")
      print("      C - Enter/view Geo.Cal parameters")
      print("    ---------------------------------")
      print("      X - Return to Main Menu")
      print("    ---------------------------------")
      opt = input ( "    Enter Option: " ).upper()

      if opt == 'I': opt = img.menu()
      if img.disp(opt,box=box): pass
      elif opt == 'Z': zoomc = not zoomc
      elif opt == 'P': ix,iy = img.click()
      elif opt[0] == 'U': iy += movepix(opt)
      elif opt[0] == 'D': iy -= movepix(opt)
      elif opt[0] == 'L': ix -= movepix(opt)
      elif opt[0] == 'R': ix += movepix(opt)
      elif opt == 'G': ix,iy = inputxy((ix,iy)) 
      elif opt == 'A': self.angles(ix,iy)
      elif opt == 'C': self.input()
      

  def save(self):
    """ Write GeoCal data to file 
 
    DESCRIPTION
      This writes data to file self.geofil established on initialisation
      which may therefore involve over-writing the original contents.
    """

    f = open(self.geofil,"w")
    f.write("     Y0        X0       ALPHA     BETA\n")
    f.write("{:10.3f}".format(self.y0))
    f.write("{:10.3f}".format(self.x0))
    f.write("{:10.3f}".format(self.alpha))
    f.write("{:10.3f}".format(self.beta))
    f.close()
    print(" *** GeoCal data written to file: " + self.geofil)
    time.sleep(3)      #  Pause for 3 seconds

  def input(self):
    """ Read user-input of GeoCal parameters

    DESCRIPTION
      Four parameters are required for the geometric calibration
        G_ALPHA scaling in elevation (vertical direction) 
        Y0      y-coordinate of sub-satellite point
        G_BETA  scaling in azimuth (horizontal direction) 
        Y0      x-coordinate of sub-satellite point
      If the calibration parameters are already established, these are printed to
      the screen and the user given the option of updating these, including just
      the ele/yoffset or the azi/xoffset pairs.
      This prints a warning if the supplied values are well outside the expected
      range and the user has to type <CR> for the program to continue
    """

    if self.cal:
      print("    Current values:")
      print("       G_alpha, Y0 = {:10.3f}, {:10.3f}".format(self.alpha,self.y0) )
      print("       G_beta,  X0 = {:10.3f}, {:10.3f}".format(self.beta, self.x0) )
      print("    Enter new values, or <CR> to keep current values")
    beta,  x0 = read_numbers(2,"      Enter G_beta,  X0: ")
    if np.isfinite(beta): 
      if beta < 40.0 or beta > 60.0:
        dummy = input(" *** Your G_beta value looks wrong! (<CR> to continue)")
      self.beta  = beta
      if x0 < 400 or x0 > 600:
        dummy = input(" *** Your X0 value looks wrong! (<CR> to continue)")
      self.x0    = x0
    alpha, y0 = read_numbers(2,"      Enter G_alpha, Y0: ")
    if np.isfinite(alpha): 
      if alpha < 40.0 or alpha > 60.0:
        dummy = input(" *** Your G_alpha value looks wrong! (<CR> to continue)")
      self.alpha = alpha
      if y0 < 400 or y0 > 600:
        dummy = input(" *** Your Y0 value looks wrong! (<CR> to continue)")
      self.y0    = y0
    # update if values for self.alpha,beta have been provided and at least one
    # pair of parameters have been modified
    if self.alpha != 0.0 and self.beta != 0.0 and \
       ( np.isfinite(alpha) or np.isfinite(beta) ): 
      self.cal   = True
      self.save()
      time.sleep(3)
    else:
      print(" *** Geometric Calibration has not been updated *** ")
      time.sleep(3)


  def angles(self,ix,iy):
    """ Print Elevation,Azimuth for given Lat,Lon 

    PARAMETERS
      ix int : Current pixel x-coordinate
      iy int : Current pixel y-coordinate

    DESCRIPTION
      Prints calculated angles to the screen, using module satang.
      User is prompted to enter Lat,Lon of a point on the surface - assumed to be
      the current pixel location ix,iy but not actually tested. ix,iy values are
      only used for part of the output text.
    """

    lat,lon = read_numbers ( 2, "      Enter Lat, Lon: " )
    ele, azi, zen = self.satang(lat,lon)
    print("        X={:4n}".format(ix) + 
             " Lon={:7.2f}".format(lon) + 
             " Azi={:6.3f}".format(azi)   ) 
    print("        Y={:4n}".format(iy) + 
             " Lat={:7.2f}".format(lat) + 
             " Ele={:6.3f}".format(ele)   ) 
    time.sleep(3)
    return

  def locang(self,ele,azi):
    """ Convert ele,azi angles to lat,lon,zen angles

    PARAMETERS
      ele flt : Elevation angle [deg]
      azi flt : Azimuth angle [deg]

    RETURNS
      lat flt : Latitude [deg N]
      lon flt : Longitude [deg E]
      zen flt : Zenith angle [deg]

    DESCRIPTION
      The inverse of SATANG.
      Uses spherical coordinate geometry to find the point of intersection of a 
      ray leaving the satellite at particular ele,azi angle with the earth surface
      If no intersection, returns (np.nan,np.nan,np.nan) instead.
    """
    rele     = math.radians(ele)
    sinele   = math.sin(rele)
    cosele   = math.cos(rele)
    razi     = math.radians(azi)
    sinazi   = math.sin(razi)
    cosazi   = math.cos(razi)
    h = DIST * sinele # Distance of plane of intersection from centre of earth 
    if abs(h) > REARTH: return (np.nan,np.nan,np.nan)  # no intersection with sfc
    r1 = math.sqrt( REARTH**2 - h**2 ) # Radius of circle of intersection
    d1 = DIST * cosele
    if abs ( d1 * sinazi ) > r1: return (np.nan,np.nan,np.nan) # No intersection
    # Distance of line of sight
    x = d1 * cosazi - math.sqrt( r1**2 - d1**2 * sinazi**2 )
    # Distance from pixel to point of intersection of earth's vertical axis with
    # plane of intersection
    d2 = DIST / cosele
    y = x**2 + d2**2 - 2 * x * d2 * cosazi
    if y < 0.0: y = 0.0
    y = math.sqrt(y)
    h1 = DIST * math.tan(rele)
    if abs(h1) > 1.0e-10:     
      gamma = math.acos( ( REARTH**2 + h1**2 - y**2 ) / ( 2.0 * REARTH * h1 ) ) 
    else:
      gamma = math.pi / 2.0 - h1 / ( 2.0 * REARTH )
    rlat = math.pi / 2.0 - gamma
    gamma1 = math.asin ( sinazi * x / y )
    rlon = math.atan ( math.sin(gamma1) / ( math.cos(gamma1) * cosele ) )
    rzen = math.acos ( cosazi * cosele ) + \
           math.acos ( math.cos(rlat) * math.cos(rlon) )
    lat = math.degrees(rlat)
    lon = math.degrees(rlon)
    zen = math.degrees(rzen)
    return ( lat, lon, zen )

  def locate(self,ix,iy):
    """ Convert ix,iy coords to lat,lon,zen angles

    PARAMETERS
      ix int : Current pixel x-coordinate
      iy int : Current pixel y-coordinate 

    RETURNS
      lat flt : Latitude [deg N]
      lon flt : Longitude [deg E]
      zen flt : Zenith angle [deg]

    DESCRIPTION 
      Uses GeoCal parameters to convert x,y to azi,ele angles 
      then uses locang to convert azi,ele to lat,lon,zen
      If GeoCal has not been set, returns (np.nan,np.nan,np.nan).
      May also return np.nan from locang if x,y beyond edge of disk
    """

    if self.cal:
      ele = ( iy - self.y0 ) / self.alpha
      azi = ( ix - self.x0 ) / self.beta
      return self.locang(ele,azi)
    else:
      return ( np.nan, np.nan, np.nan )

  def satang(self,lat,lon):
    """ Convert lat,lon agnles to ele,azi,zen angles

    PARAMETERS
      lat flt : Latitude [deg N]
      lon flt : Longitude [deg E]

    RETURNS
      ele flt : Elevation angle [deg]
      azi flt : Azimuth angle [deg]
      zen flt : Zenith angle [deg]

    DESCRIPTION
      The inverse of locang
      Alpha is elevation and beta angle of rotation about inclined axis measured
      from the central vertical. Assumes spherical earth.
      Inputs/outputs in degrees, although internally converted to radians
    """

    # Convert lat,lon from degrees to radians
    rlat  = math.radians(lat)
    rlon  = math.radians(lon)
    h2    = REARTH * math.sin(rlat)     # Height [km] of pixel above horizontal
    r2    = REARTH * math.cos(rlat)     # Distance [km] from earth's vertical axis
    # d3 = Horizontal distance of pixel from satellite
    d3    = math.sqrt ( DIST**2 + r2**2 - 2 * DIST * r2 * math.cos(rlon) )
    delta = math.atan ( h2 / d3 ) 
    gamma = math.asin ( r2 * math.sin(rlon) / d3 )
    rele  = math.atan ( math.tan(delta) / math.cos(gamma) )
    razi  = math.asin ( math.cos(delta) * math.sin(gamma) )
    rzen  = math.acos ( math.cos(razi)  * math.cos(rele) ) + \
            math.acos ( math.cos(rlat)  * math.cos(rlon) )         
    ele = math.degrees(rele)
    azi = math.degrees(razi)
    zen = math.degrees(rzen)
    return ( ele, azi, zen )

# end of class Geo -----------------------------------------------------------------

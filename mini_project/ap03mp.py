""" This is a demonstration of some basic data handling of Meteosat images
    which might be useful as a basis for the AP03 miniproject 
 
    There are two class definitions, Image and Geo, followed by the main program
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#-------------------------------------------------------------------------------
class Image:  
  """ Image data and methods 

  DATA
    tem   boo : True=brightness temperature image, False=radiance image
    nx    int : No of horizontal pixels in images
    ny    int : No of vertical pixels in images
    ixoff int : pixel offset of left edge of box
    iyoff int : pixel offset of bottom edge of box
    data  flt : np.(ny,nx) image data as 2D array
    title str : image title 

  METHODS 
    __init__ : Initialise new Img object
        disp : Display full disk image
        clip : Create new image from subset of original image
      bright : Convert radiance image to Brightness Temperature image

  HISTORY
      v16Oct20 : AD Original version
  """

  def __init__(self,imgfil,title=None):
    """ Initialise new image object 

    PARAMETERS
      imgfil str : file name for image
      title  str : description of image

    DESCRIPTION
      Read in image data as 2D array of floating values.
      If called with imgfil==None this just creates a new 'empty' object 
      If called without title parameter, the title will be taken from the file
    """

    self.tem = False    # Assume any new image is radiance
    self.ixoff = 0
    self.iyoff = 0
    if imgfil is None:  # Create an empty object
      self.nx = 0
      self.ny = 0
      self.data = 0
      self.title = ''
    else:               # Read in image data from file
      with open(imgfil) as f:
        imgtitle = f.readline()
        if title is None: self.title=imgtitle
        else: self.title=title
        nx, ny = np.fromfile(f,dtype=int,count=2,sep=" ")  
        self.nx = nx
        self.ny = ny
        imgdata = np.fromfile(f, dtype=float, count=nx*ny, sep=" ")
        self.data = imgdata.reshape((ny,nx))

  def disp(self,window=1,box=None):
    """ Display the image 

    PARAMETERS
      window int : plot window for display
      box        : box coordinates, dictionary containing keys
        'xmin'  int : Pixel# of left edge of box
        'xmax'  int : Pixel# of right edge of box
        'ymin'  int : Pixel# of bottom edge of box
        'ymax'  int : Pixel# of top edge of box
        'color' str : (optional, default='white') color of box

    DESCRIPTION
      Basic function is to display the stored image self.data
      Can also superimpose a rectangular box defined by keys 'xm

    """
    plt.figure(window)
    plt.clf()
    plt.axis('off')                  # no axes required
    plt.title(self.title)
    if self.tem:                     # colour scale for bright.tem image
      colours = [(1,1,1),(1,0,1),(0,0,1),(0,1,1),(0,1,0),(1,1,0),
                 (1,0,0),(0,0,0)]
      tnorm = plt.Normalize(230,320) # range 230-320K
      tcmap = LinearSegmentedColormap.from_list('tem_colours', colours)
      plt.imshow(self.data, origin='lower', cmap=tcmap, norm=tnorm)
      plt.colorbar(label='Temperature [K]')
    else:                            # use gray scale for radiance image
      plt.imshow(self.data, origin='lower', cmap='gray')
      plt.colorbar(label='Radiance')
    if box is not None:
      xmin = box['xmin'] - self.ixoff
      xmax = box['xmax'] - self.ixoff
      ymin = box['ymin'] - self.iyoff
      ymax = box['ymax'] - self.iyoff
      if 'color' in box: col=box['color']
      else: col='white'
      plt.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin],color=col)
           
    plt.tight_layout(pad=0.05)       # maximise size of image within window
    plt.show()

  def clip ( self, box, title=None ):
    """ Create a new image from a subset of another image

    PARAMETERS
      box   dict : Subsect coordinates, as defined in self.disp
      title str  : Title for new image (else copy original image title)

    RETURNS
      newimg : Image object

    DESCRIPTION
      Creates a new image object from a rectangular subset of an existing image
      Note that pixel numbers from the original image have to be preserved, via
      self.ixoff and self.iyoff, in order for the geolocation to work on the
      subset image
    """
    xmin = box['xmin']
    xmax = box['xmax']
    ymin = box['ymin']
    ymax = box['ymax']
    newimg = Image(None)  
    newimg.nx = xmax-xmin 
    newimg.ny = ymax-ymin
    newimg.ixoff = xmin
    newimg.iyoff = ymin
    y1 = ymin - self.iyoff
    y2 = ymax - self.iyoff
    x1 = xmin - self.ixoff
    x2 = xmax - self.ixoff
    newimg.data = self.data[y1:y2,x1:x2]
    if title is None: newimg.title = self.title
    else: newimg.title = title
    return newimg

  def bright ( self, wavelength ):
    """ Convert radiance image to Brightness Temperature image

    PARAMETERS
      wavelength flt : characteristic wavelength [microns] for channel

    DESCRIPTION
      Convert image from radiance [W/(m2 sr um) to brightness temperature [K]
      using the inverse of the Planck function at the characteristic 
      wavelength. 
    """
    # Local constants
    H = 6.63e-34       # Planck constant       [m2 kg / s]
    C = 3.00e8         # Speed of light        [m / s]
    K = 1.38e-23       # Boltzmann constant    [m2 kg /s2 /K]
    R1 = H * C / K     # Intermediate Constant [m K]
    R2 = 2 * H * C**2  # Intermediate Constant [m4 kg / s3]
    if self.tem:
      print('Apparently already a brightness temperature image')
    else: 
      w = wavelength * 1.0e-6  # convert microns to metres
      self.data = R1 / w / np.log( 1.0 + R2/(w**5 * self.data*1e6) )
      self.tem = True
# ------------------------------------------------------------------------------
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
    v16Oct20 : AD Original version
  """
 
  def __init__(self,geofil):
    """ Initialise new Geo object

    PARAMETERS
      geofil str : name of file containing geo.cal data, eg 'geo.txt' 

    DESCRIPTION
      If the Geo Cal data file exists, the data are read on initialisation
      and geo.cal set True
    """

    # Local constants
    self.DIST = 42260.0      # Radial dist [km] of sat. from centre of earth
    self.REARTH  = 6371.0    # Earth radius [km]

    try:                     # if file already exists ...
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
      ray leaving the satellite at particular ele,azi angle with the earth 
      surface
      If no intersection, returns (np.nan,np.nan,np.nan) instead.
    """
    rele     = math.radians(ele)
    sinele   = math.sin(rele)
    cosele   = math.cos(rele)
    razi     = math.radians(azi)
    sinazi   = math.sin(razi)
    cosazi   = math.cos(razi)
    # Distance of plane of intersection from centre of earth
    h = self.DIST * sinele 
    if abs(h) > self.REARTH: return (np.nan,np.nan,np.nan)  # no sfc intersect.
    r1 = math.sqrt( self.REARTH**2 - h**2 ) # Radius of circle of intersection
    d1 = self.DIST * cosele
    if abs ( d1 * sinazi ) > r1: return (np.nan,np.nan,np.nan) # No intersection
    # Distance of line of sight
    x = d1 * cosazi - math.sqrt( r1**2 - d1**2 * sinazi**2 )
    # Distance from pixel to point of intersection of earth's vertical axis with
    # plane of intersection
    d2 = self.DIST / cosele
    y = x**2 + d2**2 - 2 * x * d2 * cosazi
    if y < 0.0: y = 0.0
    y = math.sqrt(y)
    h1 = self.DIST * math.tan(rele)
    if abs(h1) > 1.0e-10:     
      gamma = math.acos( ( self.REARTH**2 + h1**2 - y**2 ) / 
                         ( 2.0 * self.REARTH * h1 ) ) 
    else:
      gamma = math.pi / 2.0 - h1 / ( 2.0 * self.REARTH )
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
    """ Convert lat,lon angles to ele,azi,zen angles

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
    # Height [km] of pixel above horizontal
    h2    = self.REARTH * math.sin(rlat)     
    # Distance [km] from earth's vertical axis
    r2    = self.REARTH * math.cos(rlat)     
    # Horizontal distance of pixel from satellite
    d3    = math.sqrt ( self.DIST**2 + r2**2 - 
                        2 * self.DIST * r2 * math.cos(rlon) )
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
#-------------------------------------------------------------------------------

# Start of main program

print('This demonstrates some basic image handling')
print('Type <CR> (press Enter) to continue through each step of demo')
print('No text input required - just press Enter at each prompt\n')

# turn on interactive mode for plots
plt.ion()  

input('1. Read in and display Vis image ... [Press Enter]')  
c1 = Image('data/channel_1/msg_c01_z12.img')  # Load visible channel image
c1.disp()                      # display image

print('\n2. Select a subset of the image ...')
print('   Define a box by pixel coordinates')
print('   (Image dimensions: nx={}, ny={})'.format(c1.nx, c1.ny))
print('   Default values: xmin=250, xmax=450, ymin=350, ymax=550')
print('   Press Enter to use defaults, or enter custom values:')

xmin_input = input('   Enter xmin (left edge) [250]: ').strip()
xmin = int(xmin_input) if xmin_input else 250

xmax_input = input('   Enter xmax (right edge) [450]: ').strip()
xmax = int(xmax_input) if xmax_input else 450

ymin_input = input('   Enter ymin (bottom edge) [350]: ').strip()
ymin = int(ymin_input) if ymin_input else 350

ymax_input = input('   Enter ymax (top edge) [550]: ').strip()
ymax = int(ymax_input) if ymax_input else 550

# define a box by pixel coordinates
box = {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}
print('   Selected box: xmin={}, xmax={}, ymin={}, ymax={}'.format(xmin, xmax, ymin, ymax))
c1.disp(box=box) # redisplay image with box outlines

input('3. Create a new image from subset ... [Press Enter]')
c1b = c1.clip(box,title='Selected part of Vis image')
c1b.disp()

input('4. Load matching Ch9 infrared window image... [Press Enter]')
c9 = Image('data/channel_9/msg_c09_z12.img')  # Load Ch9 image
c9b = c9.clip(box,title='Corresponding part of Ch9 image')
c9b.disp(window=2)   # display in a different window

input('5. Convert Ch9 radiance to brightness temperature ... [Press Enter]') 
c9b.bright(wavelength=10.79)
c9b.disp(window=2)

input('6. Create scatter plot of Ch9 v Ch1 ... [Press Enter]')
# convert 2D images to 1D arrays
#idx = cloudmask.flatten() # index in 1D array of cloud-free points
c1vec = c1b.data.flatten()
c9vec = c9b.data.flatten()
plt.figure(3)
plt.clf()
plt.title('C9 v C1  Scatter plot')
plt.xlabel('C1 Radiance')
plt.ylabel('C9 Temperature')
plt.scatter(c1vec,c9vec,s=1,color='black')
#plt.scatter(c1vec[idx],c9vec[idx],s=1)
plt.show()

input('7. Select threshold values for cloud detection... [Press Enter]')
c9cld = 280
c1cld = 100
plt.plot(plt.xlim(), [c9cld,c9cld],color='red')
plt.plot([c1cld,c1cld],plt.ylim(),color='red')
plt.text(0,290,'Cloud Free',color='red')
plt.show()

input('8. Replot images with cloud mask applied ... [Press Enter]')
# True = cloud-free, False = cloudy
c1mask = c1b.data.__lt__(c1cld) 
c9mask = c9b.data.__gt__(c9cld)
cloudmask = np.logical_and(c1mask,c9mask)
c1b.data = np.where(cloudmask,c1b.data,100) # Set cloudy data to Rad=100
c1b.title = 'Cloud mask applied to Vis image'
c1b.disp(window=1)
c9b.data = np.where(cloudmask,c9b.data,200) # Set cloudy data to T=200K
c9b.title = 'Cloud mask applied to Ch9 image'
c9b.disp(window=2)

print('\n8. Select an even smaller box of Ch9 Image ...')
print('   (Current image dimensions: nx={}, ny={})'.format(c9b.nx, c9b.ny))
print('   Default values: xmin=260, xmax=280, ymin=390, ymax=410')
print('   Press Enter to use defaults, or enter custom values:')

xmin2_input = input('   Enter xmin [260]: ').strip()
xmin2 = int(xmin2_input) if xmin2_input else 260

xmax2_input = input('   Enter xmax [280]: ').strip()
xmax2 = int(xmax2_input) if xmax2_input else 280

ymin2_input = input('   Enter ymin [390]: ').strip()
ymin2 = int(ymin2_input) if ymin2_input else 390

ymax2_input = input('   Enter ymax [410]: ').strip()
ymax2 = int(ymax2_input) if ymax2_input else 410

box2 = {'xmin':xmin2, 'xmax':xmax2, 'ymin':ymin2, 'ymax':ymax2, 'color':'black'}
print('   Selected box: xmin={}, xmax={}, ymin={}, ymax={}'.format(xmin2, xmax2, ymin2, ymax2))
c9b.disp(window=2,box=box2)
c9b = c9b.clip(box2)

input('9. Create histogram of Ch9 image ... [Press Enter]')
c9vec = c9b.data.flatten()
# Note that c9vec still contains cloudy values set to 200K, although now
# it is reduced to 1D these could also be removed completely at this point
plt.figure(3)
plt.clf()
plt.title('Histogram of cloud-cleared Ch9 box')
plt.xlabel('Radiance')
plt.ylabel('Frequency')
plt.hist(c9vec,range=[280,300],bins=51)
plt.show()

input('10. Load Geographical Calibration data ... [Press Enter]')
geo = Geo('geo.txt') 

input('10. Print Lat,Lon of lower left and upper right of box ... [Press Enter]')
(latmin,lonmin,zenmin) = geo.locate(box2['xmin'],box2['ymin'])
print(latmin,lonmin)
(latmax,lonmax,zenmax) = geo.locate(box2['xmax'],box2['ymax'])
print(latmax,lonmax)

input('Finish Demo ... [Press Enter to exit]')





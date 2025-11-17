# Standard modules
import numpy as np
import matplotlib.pyplot as plt

# Local modules
from ap03_utils import tem_colours    # Rainbow colour scale for temperature images
from ap03_utils import plot_colours   # Colours used for each channel's plots

# Local constants
# Assume window#1 in use for main image of full disk
WINDOW_ZOOM = 2      # Window# for Zoom Display
WINDOW_HIST = 3      # Window# for Histogram plots

class Box:
  """ Image cursor box data and methods

  DATA 
    ix    int : x-coordinate of box centre
    iy    int : y-coordinate of box centre
    mbox  int : Box half-size
    nbox  int : Box full size = 2*mbox + 1
    img   flt : np.(nbox,nbox) array of image values within box
    label str : Label of current image displayed (eg 'CV')
    desc  str : Image description (eg 'Visible Channel - Pixel Values')

  METHODS
    __init__ : Initialise Box object
      disp   : Create Zoom Display of current box
      hist   : Display histogram of box values 

  DESCRIPTION
    A new object is created whenever the cursor is moved or the image changed
    This is used in the Geographic and Radiometric Calibration parts of AP03.

  HISTORY
    v31Aug20 : AD Original
  """

  def __init__(self,jx,jy,img,geo,mbox=10,hist=False,zoomc=False,pfile=False):
    """ Initialise Box object 

    PARAMETERS
      jx    int : nominal x-coord of box-centre
      jy    int : nominal y-coord of box-centre
      img   obj : Full disk image data and methods
      geo   obj : Geometric calibration data and methods
      mbox  int : box half-size
      hist  boo : True = display histogram of box contents 
      zoomc boo : True = use max contrast for Zoom display, False=normal
      pfile boo : True = save histogram plot to file 

    DESCRIPTION
      A new box object is created whenever the cursor box is adjusted or 
      the underlying image changed. Since the location of the cursor box
      is also displayed on the main image this has to be called before the
      main image is updated but also after the displayed channel is changed.
    """
                        
    self.mbox = mbox       # Box half-size
    self.nbox = 2*mbox + 1 # Box size
    # Adjust central pixel location to ensure box fits within full image
    self.ix = int( np.median( [ mbox, jx, img.nx-mbox-1 ] ) )
    self.iy = int( np.median( [ mbox, jy, img.ny-mbox-1 ] ) )
    iymin = self.iy - mbox
    iymax = self.iy + mbox
    ixmin = self.ix - mbox
    ixmax = self.ix + mbox
    label = img.label      # Copy information from full disk image
    self.label = label
    self.desc = img.desc[label]
    self.img = img.images[label][iymin:iymax+1,ixmin:ixmax+1]
    self.disp(zoomc)
    if hist: self.hist(geo,pfile)

  def disp(self,zoomc):
    """ Create Zoom Display of current box 
   
    PARAMETERS
      zoomc boo : True = use max contrast for Zoom display

    DESCRIPTION
      Opens a new window (WINDOW_ZOOM) to display the contents of the cursor
      box on the main image magnified by a factor MAG = 10, to help the user
      identify landmarks and/or cloud free areas. Normally this display uses
      the same scale as the main image but, if called with zoomc=True, the
      contrast is maximised.
    """

    MAG = 10              # magnification for Zoom display
    nbox = self.nbox      # pixel size of cursor box
    nzbox = nbox * MAG    # pixel size of Zoom display
    zoom = np.ndarray((nzbox,nzbox))  # Create 2D array for Zoom image
    for ixz in range(nzbox):
      ix = ixz // MAG           # x-coord in full image
      for iyz in range(nzbox):
        iy = iyz // MAG         # y-coord in full image
        zoom[iyz,ixz] = self.img[iy,ix]
    zmin = np.min(zoom)
    zmax = np.max(zoom)
    # Within Zoom, draw line around central pixel
    i1 = self.mbox * MAG
    i2 = i1 + MAG-1
    val = 255               # value used for edge outline
    for i in range(MAG):
      zoom[i1+i,i1] = val
      zoom[i1+i,i2] = val
      zoom[i1,i1+i] = val
      zoom[i2,i1+i] = val
      val = 255 - val
    plt.figure(WINDOW_ZOOM,figsize=(3,3))
    if zoomc: plt.title("Zoom Display (enhanced)")
    else:     plt.title("Zoom Display")
    if self.label in ( 'T9', 'T10', 'TS'):   # Temperature image
      tnorm, tcmap = tem_colours()
      if zoomc: 
         plt.imshow(zoom, origin='lower', cmap = tcmap, 
                    norm=plt.Normalize(zmin,zmax) )
      else:      
         plt.imshow(zoom, origin='lower', cmap = tcmap, norm=tnorm )
    else:                                    # Pixel count image
      if zoomc: 
        plt.imshow(zoom, origin='lower', vmin=zmin, vmax=zmax, cmap='gray')
      else:
        plt.imshow(zoom, origin='lower', cmap='gray')
    plt.axis('off')
    plt.tight_layout()

  def hist(self,geo,pfile):
    """ Display histogram of box values 

    PARAMETERS
      geo   obj : Geometric calibration data and methods
      pfile boo : True = save histogram plot to file

    DESCRIPTION
      Opens a new window (WINDOW_HIST) to display a histogram of the pixel values
      within the cursor box. These are integer bins, which is natural for the
      images in pixel counts but not for the temperature images since integer
      pixel counts do not map to integer temperature values.

      If called with pfile=True the histogram plot is also saved to a file, 
      by default 'hist.pdf' but the user is prompted to change this. With python
      the file extension should also define the format of the saved image file.
    """

    # Create histogram of box data, rounding to nearest integers if temperature
    boxdata = self.img.flatten()
    imin = int(round(min(boxdata))) - 1
    imax = int(round(max(boxdata))) + 1
    ni = imax-imin+1              # number of bins to plot
    h = np.zeros(ni,dtype=int)    # initialise with zeros
    for val in boxdata:           # assign each image value to a bin
      i = int(round(val)) - imin  
      h[i] += 1
    n = sum(h)                    # total number of values binned
    h = h * 100.0/n               # convert no.in bins to %frequency
    plt.figure(WINDOW_HIST,figsize=(4,4))
    plt.clf()
    # Create title for histogram plot
    ttl = self.desc + '\n' +  \
          'Box: X=' + str(self.ix-self.mbox) + ':' \
                    + str(self.ix)           + ':' \
                    + str(self.ix+self.mbox) +     \
             ', Y=' + str(self.iy-self.mbox) + ':' \
                    + str(self.iy)           + ':' \
                    + str(self.iy+self.mbox)
    plt.title(ttl)
    plt.ylabel("% Frequency")
    tdisp = self.label in ( 'T9', 'T10', 'TS' )
    if tdisp: plt.xlabel("Pixel Temperature [K]")
    else:     plt.xlabel("Pixel Value [0:255]")
    xval = np.arange(imin,imax+1,dtype=int)
    # Set colour of histogram according to channel
    plt.bar(xval,h,color=plot_colours.get(self.label,'gray'))
    x0,x1 = plt.xlim()
    y0,y1 = plt.ylim()
    boxmean = np.mean(boxdata)
    boxsd   = np.std(boxdata)
    midpix = self.img[self.mbox,self.mbox]
    plt.plot( boxmean+[0,0], [y0,y1], ':', color='black' )
    plt.errorbar ( boxmean, 0.9*y1, xerr=boxsd, color='black', 
                   capsize=4 )
    plt.plot ( midpix, 0.9*y1, 's', color='black', 
               markerfacecolor='none' ) 
    plt.tight_layout()
    if boxmean > 0.5 * ( x1 + x0 ): xt = x0 + 0.4 * ( x1 - x0 )
    else: xt = x0 + 0.95*(x1-x0)
    yt = y0 + 0.95*(y1-y0)
    yd = 0.05*(y1-y0)
    text = 'Mean = {:6.2f}'.format(boxmean)
    plt.text(xt,yt,text,ha="right")
    yt -= yd
    text = 'S.D. = {:6.2f}'.format(boxsd)
    plt.text(xt,yt,text,ha="right")
    yt -= yd
    text = 'NPix = {:6n}'.format(n)
    plt.text(xt,yt,text,ha="right")
    yt -= yd
    if tdisp: text = 'MidPix = {:6.2f}'.format(midpix)
    else:     text = 'MidPix = {:6n}'.format(midpix)
    plt.text(xt,yt,text,ha="right")
    if geo.cal:
      lat,lon,zen = geo.locate(self.ix,self.iy) 
      text = 'Lat = {:6.2f}'.format(lat)
      yt -= yd
      plt.text(xt,yt,text,ha="right") 
      text = 'Lon = {:6.2f}'.format(lon)
      yt -= yd
      plt.text(xt,yt,text,ha="right") 
    if pfile: 
      file = input ( "Save to file (<CR>=hist.pdf): " ) or "hist.pdf"
      plt.savefig(file) 

# end of class Box ---------------------------------------------------------------

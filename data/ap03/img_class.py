# Standard modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Local modules
from ap03_utils import read_numbers   #  Read set of real numbers from terminal
from ap03_utils import tem_colours    # Colours used for each channel's plots

# Local constants 
WINDOW_IMG = 1  # Window# used for displaying main image throughout AP03 program

class Img:
  """ Full disk image data and methods 

  DATA
        nx int : No of horizontal pixels in images
        ny int : No of vertical pixels in images
    images flt : {nimg} Dictionary of full disk images
   imgdisp flt : np.(ny,nx) Currently displayed image
      desc str : {nimg} Dictionary of display titles for images
     label str : Label of currently displayed image (eg 'CV' )
     title str : Overall title of set of images

  METHODS 
      __init__ : Initialise new Img object
          read : Read in image for one channel from file
          disp : Display full disk image
   box_outline : Draw dotted line on main image around cursor box
        select : Menu to select different image for display
        radcal : Apply radiometric calibration to (re-)generate T9, T10 images
        temcal : Apply temperature correction to (re-)generate TS image
         click : User click on image to move cursor box

  USAGE 
      Called once at the start to intialise an Img object

  HISTORY
      v02Sep20 : AD Original version
  """

  def __init__(self):
    """ Initialise new Img object """

    self.nx = 0
    self.ny = 0
    self.images = {}
    self.imgdisp = 0
    self.desc = {}
    self.label = ''
    self.title = ''

  def read(self,imgfil,label):
    """ Read in image for one channel from file 

    PARAMETERS
      imgfil str : Name of file containing SEVIRI image
      label  str : Label/key assigned to image (eg 'CV')

    DESCRIPTION
      This requires all images to be the same size and contain integer values.
      Descriptions for each image are assigned internally, although it would
      make more sense if the description were contained in the title line of
      the image files themselves.
    """

    with open(imgfil) as f:
      if label == 'CV':  self.desc[label] = "Visible Channel - Pixel Values"
      if label == 'C9':  self.desc[label] = "Ch9  (11$\mu$m) - Pixel Values"
      if label == 'C10': self.desc[label] = "Ch10 (12$\mu$m) - Pixel Values"
      title = f.readline()
      nx, ny = np.fromfile(f,dtype=int,count=2,sep=" ")  
      if self.title == '':   # Take title and size from first image file 
        self.title = title
        self.nx = nx
        self.ny = ny
      elif nx != self.nx or ny != self.ny:  # size mismatch
        print("F-img.read: Images are not the same size")
        exit()
      imgdat = np.fromfile(f, dtype=int, count=nx*ny, sep=" ")
      self.images[label] = imgdat.reshape((ny,nx))

  def disp(self,label,box=None,lin=None):
    """ Display full image on screen 

    PARAMETERS
      label str : Selected option, may or may not be a valid label (eg 'CV')
      box   obj : Image cursor box data and methods 
      lin   obj : Image line analysis data and methods 

    RETURNS
      True  if valid image label supplied, including '', or
      False if label argumment was not recognised as an image label
    """

    if label in self.images: # called with unrecognised image label
      self.label = label
    # make a display image copy which will have cursor box or line drawn on it
    self.imgdisp = np.copy(self.images[self.label])
    if box is not None: self.box_outline(box)
    if lin is not None: 
      self.imgdisp[lin+1] = 255
      self.imgdisp[lin-1] = 255
    plt.figure(WINDOW_IMG)
    plt.clf()
    plt.title(self.desc[self.label])
    if self.label in ('T9','T10','TS'):   # use temperature colour scale
      tnorm, tcmap = tem_colours()
      plt.imshow(self.imgdisp, origin='lower', cmap=tcmap, norm=tnorm)
      plt.colorbar(label='Temperature [K]')
    else:                            # use gray scale
      plt.imshow(self.imgdisp, origin='lower', cmap='gray')
      plt.colorbar(label='Pixel Counts')
    plt.axis('off')                  # no axes required
    plt.tight_layout(pad=0.05)       # maximise size of image within window
    plt.show()
    return label == self.label       # True=changed image displayed

  def box_outline(self,box):
    """ Draw dotted line on main image around cursor box 

    PARAMETERS
      box obj : Image cursor box data and methods 

    DESCRIPTION
      This sets pixel values alternately 255,0 around the edges of the box
      These map to black/white for gray scale images.
      Note that this just overwrites the displayed image self.imgdisp rather
      than the original images in self.img
    """

    # Determine X,Y coordinates on main image at edges of box
    ixmin = box.ix - box.mbox
    ixmax = box.ix + box.mbox
    iymin = box.iy - box.mbox
    iymax = box.iy + box.mbox
    ival= 255   
    for i in range(box.nbox):
      self.imgdisp[iymin,ixmin+i] = ival
      self.imgdisp[iymax,ixmin+i] = ival
      self.imgdisp[iymin+i,ixmin] = ival
      self.imgdisp[iymin+i,ixmax] = ival
      ival = 255 - ival

  def menu(self):
    """ Menu to select different image for display 

    RETURNS
      opt  Selected image label (eg 'CV') or else ''

    DESCRIPTION
      This presents the user with an explicit list of available images excluding
      the currently displayed image. However, it is usually possible to bypass
      this by typing in an image label in response to any of the menus
    """

    opt = '?'
    while opt != '':    # repeat until valid option or ''
      print("  Currently displaying: " + self.desc[self.label] )
      print("  --- Image Selection Options ------")
      for label in self.images.keys():
        if label == self.label: continue
        print("    {:<3} - Switch to ".format(label) + self.desc[label] )
      print("   <CR> - Keep current image")
      print("  ----------------------------------")

      opt = input("  Enter Option: ").upper()
      if opt in self.images.keys(): return opt

  def radcal(self,rad):
    """ Apply Radiometric Calibration to (re-)generate T9, T10 images 

    PARAMETERS
      rad obj : Radiometric calibration data and methods

    DESCRIPTION  
      Create brightness temperature images T9, T10 from pixel counts C9, C10 
      using radiometric calibration data and applying rad.bright method 
      This adds images labelled T9 and T10 to self.images
    """ 
    if not rad.cal: return
    for tchn in ( 'T9', 'T10' ):
      if tchn == 'T9': 
        cchn = 'C9'
        desc = "Ch9  (11$\mu$m) - Brightness Temperature"
      else:
        cchn = 'C10'
        desc = "Ch10 (12$\mu$m) - Brightness Temperature"
      self.desc[tchn] = desc
      self.images[tchn] = rad.bright(tchn,self.images[cchn])

  def temcal(self,tem):
    """ Apply Temperature Correction to (re-)generate TS image 
  
    PARAMETERS 
      tem obj : Temperature correction data and methods

    DESCRIPTION
      Calculate surface temperature correcting for water vapour absorption by 
      applying TS = T9 + gamma * ( T9 - T10 ) using user's gamma value
      Only applied to pixels where both T9 and T10 > 0 K, otherwise TS=0.
    """
    if not tem.cal: return
    if 'T9' not in self.images or 'T10' not in self.images:
      print("  Ch9,10 not yet calibrated, so cannot apply Tem Correction ")
      return
    self.images['TS'] = np.ndarray([self.ny,self.nx],dtype='float')
    self.desc['TS'] = "Surface Temperature"
    for iy in range(self.ny):
      for ix in range(self.nx):
        if self.images['T9'][iy,ix] > 0.0 and self.images['T10'][iy,ix] > 0.0:
          self.images['TS'][iy,ix] = self.images['T9'][iy,ix] + \
               tem.gamma * ( self.images['T9'][iy,ix] - self.images['T10'][iy,ix] )
        else:
          self.images['TS'][iy,ix] = 0.0

  def click(self):
    """ User click on image to move cursor box

    DESCRIPTION
      Uses pyplot ginput module assuming a click on mouse left button. See the
      on-line documentation for other button options. Also requires display to
      be set to the full disk image.
    """
    print("left click on image...")
    self.disp('')
    pts = plt.ginput()[0]    # select first pair from list
    ix = int(round(pts[0]))
    iy = int(round(pts[1]))
    return (ix,iy)

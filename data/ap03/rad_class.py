# Standard modules
import math
import time
import numpy as np
import matplotlib.pyplot as plt

# Local modules
from box_class  import Box           # Image cursor box data and methods
from ap03_utils import inputxy       # Input new x,y coordinates to move cursor box
from ap03_utils import movepix       # Number of pixels to move
from ap03_utils import pixloc        # Text string containing pixel location
from ap03_utils import read_numbers  # Read set of numbers from terminal

class Rad:
  """ Radiometric calibration data and methods 

  DATA
    cal    boo : True = Radiometric Calibration set
    a9     flt : Ch9 Gain 
    a10    flt : Ch10 Gain
    b9     flt : Ch9 Offset 
    b10    flt : Ch10 Offset
    w9     flt : Ch9 Characteristic wavelength
    w10    flt : Ch10 Characteristic wavelength
    radfil str : Name of file containing RadCal data (eg 'rad.txt')
    
  METHODS
    __init__ : Initialise new object
      menu   : Screen menu for RadCal part of practical 
      save   : Write RadCal data to file
      input  : User-input of Radiometric Calibration parameters
      bright : Calculate Brightness Temperature image

  USAGE 
    Called once at the start of AP03 to initialise a Rad object
      
  HISTORY
    v24Nov20 : AD Explicitly set R1,R2 in bright to match manuscript values
    v02Sep20 : AD Original version
  """

  def __init__(self,radfil,img):
    """ Initialise new object

    PARAMETERS
      radfil str : Name of file containing RadCal data 
      img    obj : Full disk image data and methods 

    DESCRIPTION
      Try opening and reading an existing file radfil which ought to contain
      any previously established radiometric calibration parameters
    """

    self.cal = False
    self.a9  = 0.0
    self.a10 = 0.0
    self.b9  = 0.0
    self.b10 = 0.0
    self.w9  = 0.0
    self.w10 = 0.0
    self.radfil = radfil
    try:
      f = open(radfil,"r")
      rec = f.readline()  
      rec = f.readline()
      flds = rec.split()
      self.w9    = float(flds[0])  # Ch9 Characteristic wavelength
      self.a9    = float(flds[1])  # Ch9 Gain 
      self.b9    = float(flds[2])  # Ch9 Offset 
      rec = f.readline()
      flds = rec.split()
      self.w10   = float(flds[0])  # Ch10 Characteristic wavelength
      self.a10   = float(flds[1])  # Ch10 Gain
      self.b10   = float(flds[2])  # Ch10 Offset
      f.close()
      self.cal = True              #  Flag Calibration as set
      print(" *** RadCal data loaded from file: " + radfil)
    except:
      print(" *** RadCal data file not found/read: " + radfil)

  def menu ( self, ix, iy, img, geo, tem ):
    """ Screen menu for RadCal part of practical

    PARAMETERS
      ix  int : x-coordinate of centre of cursor box
      iy  int : y-coordinate of centre of cursor box
      img obj : Full disk image data and methods
      geo obj : Geometric calibration data and methods 
      tem obj : Temperature correction data and methods

    DESCRIPTION
      Print Rad menu and execute selection until the user types 'X'
      The displayed images are updated each time, and the current box-centre
      location is printed at the start of the menu
    """

    mbox = 10             # initial box half-size in pixels
    zoomc = False         # True = set zoom display to maximum contrast
    pfile = False         # True = save histogram plot to file
    opt = ''

    while opt != 'X':     # Repeat until 'X' option selected
      box = Box(ix,iy,img,geo,mbox=mbox,hist=True,zoomc=zoomc,pfile=pfile)
      img.disp(opt,box=box)
      pfile = False       # Reset if required
      ix = box.ix         # Box may adjust ix,iy to fit box within image
      iy = box.iy

      print("Main> Rad Menu>")
      print("    Ship: SST=301.3K WVC=56mm LAT=4.0N LON=-22.0E")
      print("    Cursor Box centre " + pixloc(ix,iy,geo) )
      print("    --- Image Selection ---------------------")
      print("      I - Select another image")
      print("      Z - Toggle Zoom display contrast")
      print("    --- Cursor Positioning ------------------")
      print("      P - Position box using mouse")
      print("      U - Move up one line ")
      print("      D - Move down one line ")
      print("      L - Move left one column ")
      print("      R - Move right one column ")
      print("      G - Go to specific (x,y) ")
      print("      B - Expand box size ")
      print("      S - Reduce box size ")
      print("    --- Radiometric Calibration -------------")
      print("      C - Enter/view Rad.Cal parameters")
      print("      F - Save histogram plot to file ")
      print("    -----------------------------------------")
      print("      X - Return to Main Menu")
      print("    -----------------------------------------")

      opt = input ( "    Enter Option: " ).upper()

      if opt == 'I': opt = img.menu()     # select menu to switch image, or
      if img.disp(opt,box=box): pass      # direct switch of displayed change
      elif opt == 'I': img.select()
      elif opt == 'Z': zoomc = not zoomc   
      elif opt == 'P': ix,iy = img.click()
      elif opt[0] == 'U': iy += movepix(opt)
      elif opt[0] == 'D': iy -= movepix(opt)
      elif opt[0] == 'L': ix -= movepix(opt)
      elif opt[0] == 'R': ix += movepix(opt)
      elif opt == 'G': ix,iy = inputxy((ix,iy))
      elif opt == 'B': mbox += 1
      elif opt == 'S': mbox -= 1
      elif opt == 'C': self.input(img,tem)
      elif opt == 'F': pfile=True

  def save(self):
    """ Write RadCal data to file 

    DESCRIPTION
      Write out data to the radcal file (eg 'rad.txt') in plain text format
      consisting of a header record and then two records containing the 
      radiometric calibration parameters for Ch9 then Ch10. 
      This same file is read as part of the initialisation.
    """

    f = open(self.radfil,"w")
    f.write("      W        A    B\n")
    f.write("{:6.2f}".format(self.w9))
    f.write("{:9.5f}".format(self.a9))
    f.write("{:9.5f}".format(self.b9))
    f.write("\n")
    f.write("{:6.2f}".format(self.w10))
    f.write("{:9.5f}".format(self.a10))
    f.write("{:9.5f}".format(self.b10))
    f.close()
    print("RadCal data written to file: " + self.radfil)

  def input(self,img,tem):
    """ User-input of Radiometric Calibration parameters 
  
    PARAMETERS
      img obj : Full disk image data and methods 
      img tem : Temperature correction data and methods 

    RETURNS
      True  : if radiometric calibration has been updated 
      False : if radiometric calibration left unchanged 

    DESCRIPTION
      If calibration data already exist, these are printed to the screen.
      The user then has the option of updating Ch9 and Ch10 parameters separately
      A warning message is printed if the inputs are well outside the expected
      ranges and the user has to type <CR> to continue.
      A full radiometric calibration is performed if either Ch9 or Ch10 
      parameters are updated (assuming valid entries are available for both), 
      generating new 'T9' and 'T10 images.
      If the Temperature Correction has been applied (ie 'TS' image available)
      then this is also recalculated with the new calibration values.
    """

    if self.cal:
      print("    Current values:")
      print("      Ch9  WL, A, B = {:6.2f},{:9.5f},{:9.5f}".format( \
             self.w9, self.a9, self.b9 ) )
      print("      Ch10 WL, A, B = {:6.2f},{:9.5f},{:9.5f}".format( \
             self.w10, self.a10, self.b10 ) )
      print("    Enter new values, or <CR> to keep current values")

    w9,  a9,  b9  = read_numbers(3, "     Enter Ch9  WL, A, B: ")
    if np.isfinite(w9): 
      if w9 < 10.5 or w9 > 11.0:
        dummy = input(" *** Your Ch9 WL value looks wrong! (<CR> to continue)")
      self.w9  = w9
      if a9 < -0.09 or a9 > -0.05:
        dummy = input(" *** Your Ch9 A value looks wrong! (<CR> to continue)")
      self.a9  = a9
      if b9 < 15.0 or b9 > 20.0:
        dummy = input(" *** Your Ch9 B value looks wrong! (<CR> to continue)")
      self.b9  = b9

    w10, a10, b10 = read_numbers(3, "     Enter Ch10 WL, A, B: ")
    if np.isfinite(w10): 
      if w10 < 11.5 or w10 > 12.5:
        dummy = input(" *** Your Ch10 WL value looks wrong! (<CR> to continue)")
      self.w10 = w10
      if a10 < -0.08 or a10 > -0.04:
        dummy = input(" *** Your Ch10 A value looks wrong! (<CR> to continue)")
      self.a10 = a10
      if b10 < 12.0 or b10 > 18.0:
        dummy = input(" *** Your Ch10 B value looks wrong! (<CR> to continue)")
      self.b10 = b10

    # if self.w9,w10 values are both still zero, no data yet provided 
    # update also requires either a valid w9 or w10 to have been entered
    update = self.w9 != 0.0 and self.w10 != 0.0 and \
            ( np.isfinite(w9) or np.isfinite(w10) ) 
    if update:
      print(" *** Updating Radiometric Calibration ... ")
      time.sleep(3)
      self.cal = True
      self.save()
      img.radcal(self)
      if tem.cal:
        print("  *** Updating Surface Temperature Calculation ... ") 
        img.temcal(tem)
        time.sleep(3)
    else:
      print(" *** Radiometric Calibration not updated ")
      time.sleep(3)
    return update

  def bright(self,tchn,piximg):
    """ Calculate Brightness Temperature image

    PARAMETERS
      tchn   str : Bright.Temp channel to be created ('T9' or 'T10')
      piximg ing : np.array(ny,nx) of raw pixel counts ('C9' or 'C10') 

    RETURNS
      temimg flt : np.array(ny,nx) of brightness temperatures

    DESCRIPTION
      Convert infrared image from pixel counts first to radiance [W/(m2 sr m), 
      by applying radiometric calibration parameters. 
      Then convert from radiance to brightness temperature using the inverse of
      the Planck function at the characteristic wavelength. To avoid problems 
      with underflow, set any radiance < 1 W/(m2 sr m) to 1
    """

    # Local constants
    H = 6.63e-34       # Planck constant       [m2 kg / s]
    C = 3.00e8         # Speed of light        [m / s]
    K = 1.38e-23       # Boltzmann constant    [m2 kg /s2 /K]
#    R1 = H * C / K     # Intermediate Constant [m K]
    R1 = 0.014309       # Ensure match with value in manuscript
#    R2 = 2 * H * C**2  # Intermediate Constant [m4 kg / s3]
    R2 = 1.19e-16       # Ensure match with value in manuscript

    if tchn == 'T9':        # Select Ch9 radiometric calibration parameters
      a = self.a9
      b = self.b9
      w = self.w9 * 1.0e-6  # convert microns to metres
    elif tchn == 'T10':     # Select Ch9 radiometric calibration parameters
      a = self.a10
      b = self.b10
      w = self.w10 * 1.0e-6 # convert microns to metres
    else: 
      print(" Rad.bright called with unrecognised argument tch=' + str(tchn)")
      exit()

    # First convert pixel counts to radiance image, setting a min value=1.0
    # 'a' is in units of W/(m2 sr um ct) and 'b' W/(m2 sr um) so need to multipy
    # by 1e6 to convert /um to /m
    radimg = ( a * piximg + b ) * 1.0e6    # W/(m2 sr m)
    radimg = np.clip(radimg,1.0,None)   # limit to min value of 1
    # Convert radiance to brightness temperature
    temimg = R1 / w / np.log( 1.0 + R2/(w**5 * radimg) )
    return temimg

# end of Rad class --------------------------------------------------------------

# Standard modules
import math
import time
import numpy as np
import matplotlib.pyplot as plt

# Local modules
from box_class  import Box        # Image cursor box data and methods
from ap03_utils import inputxy    # Input new x,y coordinates to move cursor box
from ap03_utils import movepix    # Number of pixels to move
from ap03_utils import pixloc     # Text string containing pixel location

class Tem:
  """ Temperature correction data and methods 

  DATA
    temfil str : Name of file containing TemCal data
    cal    boo : True = TemCal data set
    gamma  flt : Temperature correction coefficient

  METHODS
    __init__ : Initialise new Tem object
      menu   : User menu for TemCal part of practical
      save   : Write TemCal data to file 

  USAGE  
    Called once at the start of AP03 to initialise a Tem object

  HISTORY
    v31Aug20 : AD Original version
  """

  def __init__ ( self, temfil ):
    """ Initialise new Tem object

    PARAMETERS
      temfil str : name of file containing tem.cal data, 'tem.txt'
    
    DESCRIPTION
      If the Tem Cal data file exists it is read on initialisation.
      Otherwise it can be created/amended via the self.save method
    """

    self.temfil = temfil         # save name of TemCal file
    self.cal    = False
    self.gamma  = 0.0
    try:
      f = open(temfil,"r")
      rec = f.readline()  
      rec = f.readline()
      self.gamma = float(rec)    # Gamma value for Tem Correction
      f.close()
      self.cal = True            # Tem Correction Gamma value loaded
      print(" *** TemCal data loaded from file: " + temfil)
    except:
      print(" *** TemCal data file not found/read: " + temfil)


  def menu ( self, ix, iy, img, geo, rad ):
    """ User menu for TemCal part of practical 

    PARAMETERS
      ix  int : x-coordinate of centre of cursor box
      iy  int : y-coordinate of centre of cursor box
      img obj : Full disk image data and methods
      geo obj : Geometric calibration data and methods
      rad obj : Radiometric calibration data and methods
    """

    mbox  = 10           # Initial half-size of cursor box
    zoomc = False        # True = set zoom diplay to max contrast
    pfile = False        # True = save histogram plot to file
    opt   = ''
    while opt != 'X':    # Repeat menu unti 'X' selected
      box = Box(ix,iy,img,geo,mbox=mbox,hist=True,zoomc=zoomc,pfile=pfile)
      img.disp(opt,box=box)
      pfile = False      # Reset if necessary
      ix = box.ix        # ix,iy may have been modified by Box to fit box within 
      iy = box.iy        # full image

      print("Main> Tem Menu>")
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
      print("      E - Expand box size ")
      print("      S - Shrink box size ")
      print("    --- Temperature Correction --------------")
      print("      C - Enter/view Temperature Correction")
      print("      F - Save histogram to file")
      print("    -----------------------------------------")
      print("      X - Return to Main Menu")
      print("    -----------------------------------------")

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
      elif opt == 'E': mbox += 1
      elif opt == 'S': mbox -= 1
      elif opt == 'C': self.input(img)
      elif opt == 'F': pfile = True

  def save(self):
    """ Write TemCal data to file 

    DESCRIPTION
      Opens temfil (eg 'tem.txt') and writes Gamma value to the file
    """ 

    f = open(self.temfil,"w")
    f.write(" Gamma\n")
    f.write("{:6.2f}".format(self.gamma))
    f.close()
    print(" *** TemCal data written to file: " + self.temfil)
    time.sleep(3)     # pause for 3 seconds

  def input(self,img):
    """ User-input of TemCal parameter 

    PARAMETERS
      img obj : Full disk image data and methods

    DESCRIPTION
      If a Gamma value for the temperature correction is already stored, the user
      is asked if they want to replace it. If a new value is supplied it is saved
      to the temcal file and the correction applied to generate a new TS image
    """

    if self.cal:
      print("  Current value of Gamma = {:6.2f}".format(self.gamma))
      gamma = input ( "  Enter new value, or <CR> to keep: ")
    else:
      gamma = input ( "  Enter value of Gamma, or <CR> to skip: ")
    try:
      self.gamma = float(gamma)
      if self.gamma < 2.0 or self.gamma > 4.0:
        input(" *** Your Gamma value looks wrong! (<CR> to continue)")
      self.cal   = True
      print(" *** Updating Surface Temperature Calculation ... ")
      img.temcal(self)      # Apply Tem Correction to create full image of TS
      self.save()           # Save new Gamma value to TemCal file
    except:
      print(" *** Gamma value not updated ")
      time.sleep(3)
 
# end of Tem class ---------------------------------------------------------------

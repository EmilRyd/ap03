""" AP03 Meteosat Practical  """

VERSION = "24Nov20" # AD Adjust BT values calculated in rad_class.bright 
# 27Oct20 AD Change IMGDIR from ../seviri to ./seviri
# 02Sep20 AD Initial python conversion from IDL program image.pro

# Standard modules
import matplotlib.pyplot as plt   # standard Python plotting 
from pathlib import Path          # Windows/Unix filesystem paths

# Local modules
from img_class import Img     # Full disk image data and methods
from geo_class import Geo     # Geometric calibration data and methods
from lin_class import Lin     # Image line analysis data and methods
from rad_class import Rad     # Radiometric calibration data and methods
from tem_class import Tem     # Temperature correction data and methods

# Local constants
IMGDIR = Path("./seviri/")   # Directory holding images
GEOFIL = 'geo.txt'            # Geographical Calibration data filename
RADFIL = 'rad.txt'            # Radiometric Calibration data filename
TEMFIL = 'tem.txt'            # Temperature Correction data filename

print("AP03: Remote Sensing from Satellites v" + VERSION )
plt.ion()    # Turn on 'interactive mode' for plots

# Create "img" object and load the 3 basic images
img = Img()
img.read ( IMGDIR / 'msg_c01.img', 'CV' )
img.read ( IMGDIR / 'msg_c09.img', 'C9' )
img.read ( IMGDIR / 'msg_c10.img', 'C10' )

# Start with display of visible channel image 
img.disp('CV')
print("Images from: " + img.title)

# Create "geo" object and load any existing Geographical Calibration data
geo = Geo(GEOFIL)

# Create "rad" object and load any existing Radiometric Calibration data
rad = Rad(RADFIL,img)
if rad.cal: img.radcal(rad)  # create T9,T10 images

# Create "tem" object and load any existing Temperature Correction data
tem = Tem(TEMFIL)
if tem.cal: img.temcal(tem)  # create TS image
 
# Create "lin" object
lin = Lin()

# Set initial location to centre of image
ix = img.nx // 2 
iy = img.ny // 2

# Cycle through the main menu until "Exit program" option selected
opt = ''
while opt != 'Q':
  img.disp(opt)   # opt may also be a valid image label, eg 'CV'
  print("Main Menu>")
  print("  --- Main Program Options -------------")
  print("    I - Image selection options")
  print("    G - Geometric Calibration")
  print("    R - Radiometric Calibration")
  print("    T - Temperature Measurement")
  print("    L - Image Line Analysis")
  print("")
  print("    Q - Exit program")
  print("  --------------------------------------")
  opt = input ( "    Enter Option: " ).upper()

  if   opt == 'I': opt = img.menu()
  elif opt == 'G': geo.menu(ix,iy,img)
  elif opt == 'R': rad.menu(ix,iy,img,geo,tem)    
  elif opt == 'T': tem.menu(ix,iy,img,geo,rad)    
  elif opt == 'L': lin.menu(iy,img,geo)    

print("AP03 Program terminated normally")

# end of ap03 --------------------------------------------------------------------

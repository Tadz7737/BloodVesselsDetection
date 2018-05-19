import skimage
import cv2
from skimage import data, io, filters, exposure, feature, img_as_float, exposure
from skimage.morphology import square, dilation, erosion
from matplotlib import pyplot as plt
from PIL import Image
from os.path import abspath, exists
import numpy as np

FILE = "01.jpg"
FILE_MASK = "01.tif"

fig = plt.figure()

#f, ax = plt.subplots(1)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

img_src = cv2.imread(FILE)
ax1.imshow(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB))

#wczytanie maski
img_mask = cv2.imread(FILE_MASK,0)

height = img_src.shape[0]
width = img_src.shape[1]


img_gray = cv2.imread(FILE,0)
#zwiekszenie kontrastu
img_rescale = exposure.rescale_intensity(img_gray, in_range=(0,105))
ax2.imshow(img_rescale, cmap="gray")

#rozmycie
blur = cv2.blur(img_rescale,(3,3)) 

#korekcja gamma
#gamma_corrected = exposure.adjust_gamma(blur,2)

#korekcja logarytmiczna
#log_corrected = exposure.adjust_log(gamma_corrected,1)

#normalizacja histogramu
equ = cv2.equalizeHist(blur)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
cl1 = clahe.apply(equ)
ax3.imshow(cl1, cmap="gray")

#detekcja krawedzi
image = img_as_float(cl1)
image = feature.canny(image, sigma=3)

#dylatacja
#kernel1 = np.ones((20,20),np.uint8)
#kernel2 = np.ones((5,5),np.uint8)

#image = cv2.dilate(image,kernel1,iterations=1)
#image = cv2.erode(image,kernel2,iterations=1)
for i in range(0,height):
    for j in range(0,width):
        if(image[i][j]==True):
            if(img_src[i][j][2]<125):
                image[i][j]=False
image = dilation(image, square(29))
image = erosion(image,square(22))




ax4.imshow(image, cmap="gray")
ax1.set_title("Original")
ax2.set_title("Adjust contrast")
ax3.set_title("Blur, Histogramic equalization")
ax4.set_title("Blood vessels")
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')


plt.show()

fig2 = plt.figure()
bx1 = fig2.add_subplot(1,1,1)

for i in range(0,height):
    for j in range(0,width):
        if(image[i][j]==True):
           img_src[i][j][0] = 0
           img_src[i][j][1] = 0
           img_src[i][j][2] = 255
           
bx1.imshow(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB))
bx1.set_title('Highlighted vessels')
bx1.axis('off')
plt.show()

countCorrect = 0
countPixels = 0
for i in range(0,height):
    for j in range(0,width):
        countPixels +=1
     #   print(img_mask[i][j])
        if((image[i][j]==True) and (img_mask[i][j]==255)) or ((image[i][j]==False) and (img_mask[i][j]==0)):
            countCorrect +=1

averageQuality = (float(countCorrect) / countPixels) *100.0
print("Srednia jakosc: ",averageQuality,"\n")


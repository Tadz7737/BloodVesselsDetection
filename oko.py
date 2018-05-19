import skimage
import cv2
from skimage import data, io, filters, exposure, feature, img_as_float, exposure
from skimage.morphology import square, dilation, erosion
from matplotlib import pyplot as plt
from PIL import Image
from os.path import abspath, exists
import numpy as np

#nazwa pliku oraz maski
FILE = "01.jpg"
FILE_MASK = "01.tif"

#stworzenie planu
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.set_title("Original")
ax2.set_title("Adjust contrast")
ax3.set_title("Blur, Histogramic equalization")
ax4.set_title("Blood vessels")

ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
#wczytanie obrazu
img_src = cv2.imread(FILE)
ax1.imshow(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB))

#wczytanie maski
img_mask = cv2.imread(FILE_MASK,0)

height = img_src.shape[0]
width = img_src.shape[1]

#wczytanie obrazu w skali szarosci
img_gray = cv2.imread(FILE,0)

#zwiekszenie kontrastu
img_rescale = exposure.rescale_intensity(img_gray, in_range=(0,105))
ax2.imshow(img_rescale, cmap="gray")

#rozmycie
blur = cv2.blur(img_rescale,(3,3)) 

#normalizacja histogramu
equ = cv2.equalizeHist(blur)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
cl1 = clahe.apply(equ)
ax3.imshow(cl1, cmap="gray")

#detekcja krawedzi
image = img_as_float(cl1)
image = feature.canny(image, sigma=3)

#filtracja pixeli na podstawie czerwieni
for i in range(0,height):
    for j in range(0,width):
        if(image[i][j]==True):
            if(img_src[i][j][2]<125):
                image[i][j]=False

#dylatacja
image = dilation(image, square(29))

#erozja
image = erosion(image,square(22))
ax4.imshow(image, cmap="gray")

#wyswietlenie planu
plt.show()

#stworzenie nowego planu
fig2 = plt.figure()
bx1 = fig2.add_subplot(1,1,1)

#pokrywanie oryginalnego obrazu naczyniami krwionosnymi
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

#obliczenia sredniego poprawnego pokrycia (rowniez z czarnymi pixelami)
countCorrect = 0
countPixels = 0
for i in range(0,height):
    for j in range(0,width):
        countPixels +=1
        if((image[i][j]==True) and (img_mask[i][j]==255)) or ((image[i][j]==False) and (img_mask[i][j]==0)):
            countCorrect +=1

averageQuality = (float(countCorrect) / countPixels) *100.0
print("Srednia jakosc: ",averageQuality,"\n")
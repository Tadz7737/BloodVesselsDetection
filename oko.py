import skimage
import cv2
from skimage import data, io, filters, exposure, feature, img_as_float
from skimage.morphology import square, dilation
from matplotlib import pyplot as plt
from PIL import Image
from os.path import abspath, exists
import numpy as np

f, ax = plt.subplots(1)
img_src = cv2.imread("15_h.jpg",0)

#rozmycie
blur = cv2.blur(img_src,(3,3)) 

#normalizacja histogramu
equ = cv2.equalizeHist(blur)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
cl1 = clahe.apply(equ)

#detekcja krawedzi
image = img_as_float(cl1)
image = feature.canny(image, sigma=3)

#dylatacja
image = dilation(image, square(3))
ax.imshow(image, cmap="gray")
ax.set_title("15_h.jpg")
ax.axis('off')

 
plt.show()

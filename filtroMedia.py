from math import  floor, ceil
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

from numpy.lib.type_check import imag

image_name = 'noise.jpeg'
def filtroMediaOpenCV(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32)/(kernel_size**2)
    dst = cv.filter2D(img, -1, kernel)
    return dst

def thresholding(image, threshold):
    new_image = image.copy()
    if len(new_image.shape) == 3:
        new_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if(new_image[i][j] > threshold):
                new_image[i][j] = 255
            else:
                new_image[i][j] = 0
    return new_image

def filtroMediaMio(image, kernel_size):
    start = time.time()
    new_image = image.copy()
    if(len(image.shape) == 3): #se l'immagine non è in scala di grigi convertila
        new_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    a = floor((kernel_size - 1)/2)
    b = floor((kernel_size - 1)/2)
    new_pixel = 0
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            for r in range (-a, a):
                for s in range(-b, b):
                    if( i+r < 0 or j+s <0 or i+r > image.shape[0]-1 or j+s > image.shape[1]-1):
                        pass
                    else:
                        new_pixel +=  (new_image[i+r][j+s])
            new_image[i][j] = new_pixel * (1/kernel_size**2)
            new_pixel = 0
    end = time.time()
    elapsed = round(end-start, 10)
    print("Tempo impiegato senza slicing: ", elapsed)
    return new_image

def localEnhancement(image, level, E):
    new_image = image.copy()
    if(len(image.shape) == 3): #se l'immagine non è in scala di grigi convertila
        new_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if new_image[i][j] <= level:
                new_image[i][j] = (new_image[i][j]+1)*E
    return new_image

def filtroMediaSlicing(image, kernel_size):
    '''Filtro media implementato tramite slicing'''
    start = time.time()
    new_image = image.copy()
    if(len(image.shape) == 3): #se l'immagine non è in scala di grigi convertila
        new_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    a = floor((kernel_size - 1)/2)
    b = floor((kernel_size - 1)/2)

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            new_image[i][j] = np.sum(new_image[i-a:i+a+1, j-b:j+b+1]) * (1/(kernel_size**2))
    end = time.time()
    elapsed = round(end-start, 10)
    print("Tempo impiegato con Slicing: ", elapsed)
    return new_image


def calcHist(image):
    histogram = [0]*256
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if histogram[image[i][j]] == None:
                histogram[image[i][j]] = 1
            else:
                histogram[image[i][j]] += 1
    return histogram

def filtroMedianaHuang(image, filter_size):
    '''Applica filtro mediana all'immagine (in bianco e nero) con la tecnica di Huang con complessita O(n)'''
    start = time.time()
    new_image = image.copy()
    histogram = calcHist(image)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            for k in range(ceil(-filter_size/2), ceil(filter_size/2)):
                if(i+k>=image.shape[0] or j+filter_size>=image.shape[1] or j-filter_size-1>=image.shape[1]):
                    pass
                else:
                    histogram[image[i+k][j-filter_size-1]]-=1
                    histogram[image[i+k][j+filter_size]]+=1
            hist2 = histogram.copy()
            new_image[i][j] = np.median(hist2)
    end = time.time()
    print("Tempo impiegato senza slicing: ", round(end-start, 10))   
    return image



threshold = 50
enchancement = 3.5
level = 25
image = cv.imread(image_name)
gray_im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
new_image = filtroMedianaHuang(gray_im, 9)
#new_image_2 = filtroMediaOpenCV(gray_im, 9)
plt.subplot(222), plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB)),plt.title('Original') #OpenCv usa BRG invece che RBG, quindi bisogna invertire il colore
plt.subplot(221), plt.imshow(new_image, cmap='gray'), plt.title('Filtro Mediana')
#plt.subplot(223), plt.imshow(new_image_2, cmap='gray'), plt.title('Filtro Media')
plt.show()
# plt.subplot(221), plt.imshow(gray_im, cmap='gray'), plt.title('Gray')
# plt.subplot(222), plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB)),plt.title('Original') #OpenCv usa BRG invece che RBG, quindi bisogna invertire il colore
# plt.subplot(223), plt.imshow(filtroMediaMio(gray_im, kernel_size), cmap='gray'), plt.title('Filtro Media Senza Slice ' +   str(kernel_size) +"x" + str(kernel_size))
# plt.subplot(224), plt.imshow(filtroMediaSlicing(gray_im, kernel_size), cmap='gray'), plt.title('Filtro Media Slice ' +   str(kernel_size) +"x" + str(kernel_size))
# plt.show()

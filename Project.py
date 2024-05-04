# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:05:51 2024

@author: jla
"""
# All the imports
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from scipy import ndimage
from scipy import interpolate

# %%

# =============================================================================
# Retrieval of images
# =============================================================================


cwd = os.getcwd()
folder_images = "C://Users//pms20//OneDrive//Escritorio//Imatges Lab//Trabajo Imatges//bicl_project//Dataset 2"
folder_image_path = os.path.join(cwd, folder_images)

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            image = plt.imread(path)
            if image is not None:
                images[filename] = image
    return images

images = load_images_from_folder(folder_image_path)

im_001 = images["001.bmp"]

plt.imshow(im_001[:,:,2])
plt.axis("off")
plt.show()


def images_to_gray_scale(ims):
    images_gray = {}
    i = 1
    for filename, image in ims.items():
        if i % 2 != 0:
            im_g = images[filename][:,:,0]
            images_gray[filename] = im_g / im_g.max()
        i = i + 1 
    return images_gray
        
images_gray = images_to_gray_scale(images)


im_001_gray = images_gray["001.bmp"]
plt.imshow(im_001_gray, cmap = "gray")
plt.axis("off")
plt.show()

#%%
# =============================================================================
# Threshold estimation for nucleus segmentation
# =============================================================================

# Take values of maximum and minimum from the image 
maxIm = im_001_gray.max()
minIm = im_001_gray.min()

# The formula to compute the density for n1
n1 = np.linspace(1,100,100)
#n2 = 3
#n2_2 = minIm * n1 / maxIm
# using the condition of the maximum threshold 
n2_3 = minIm / (1 - maxIm/n1)

# computing the sensibility 
sensN1 =  (maxIm) / (maxIm + minIm * n1/n2_3)

# plotting the sensibility values as a function of the n values
plt.plot(n1[0:10], -sensN1[0:10])
plt.xlabel("n1")
plt.ylabel("Υn1")
plt.title("Sensivity for image 1")
plt.show()

# Using the images and performing the previous operation for all the images
for n in images_gray:
    im = images_gray[n]   
    maxIm = im.max()
    minIm = im.min()
    print(maxIm, minIm, "\n")
    
    n1 = np.linspace(1,100,100)
    #n2 = 3
    #n2_2 = minIm * n1 / maxIm
    n2 = minIm / (1 - maxIm/n1)
    
    sensN1 =  (maxIm) / (maxIm + minIm * n1/n2)
    
    plt.plot(n1[0:10], -sensN1[0:10])
 
plt.xlabel("n1")
plt.ylabel("Υn1")
plt.show()

# Same procedure for the n2:
for n in images_gray:
    im = images_gray[n]
    
    maxIm = im.max()
    minIm = im.min()
    
    n2 = np.linspace(1,100,100)
    #n2 = 3
    #n2_2 = minIm * n1 / maxIm
    n1 = maxIm / (1 - minIm/n2)
    
    sensN2 =  (minIm) / (minIm + maxIm * n2/n1)
    
    plt.plot(n2[0:10], -sensN2[0:10])
 
plt.xlabel("n2")
plt.ylabel("Υn2")
plt.show()

# %%
def Tnco_images(ims):
    thresholds = {}
    for filename, image in ims.items():
        th = image.max()/n1 + image.min()/n2
        thresholds[filename] = np.round(th, 3)
    return thresholds


Tncos = Tnco_images(images_gray)

# %%

# =============================================================================
# Approximating threshold ε˜t
# =============================================================================


def histogram_image(image, plot=True):
    hist = ndimage.histogram(image, 0, 1, 255)
    bins = np.linspace(0, 1, 255)
    if plot:
        plt.plot(bins, hist)
        plt.title('Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    return hist


histogram_image(images_gray["001.bmp"])


def histogram_image_zoomed(image, plot=True):
    size = 255
    hist = ndimage.histogram(image, 0, 1, size)
    bins = np.linspace(0, 1, size)
    if plot:
        plt.plot(bins[0:np.uint8(size*0.5)], hist[0:np.uint8(size*0.5)])
        plt.title('Histogram zoomed')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    return hist


histogram_image_zoomed(images_gray["001.bmp"])


# %%

def histogram_image_zoomed_bspline(image, degree, s, plot=True):
    size = 256
    hist = ndimage.histogram(image, 0, 1, size)
    bins = np.linspace(0, 1, size)
    
    zoom_fact = 0.6
    hist_zoom = hist[0:np.uint8(size*zoom_fact)]
    bins_zoom_sp = np.round(bins[0:np.uint8(size*zoom_fact)],3)
    bins_zoom = bins[0:np.uint8(size*zoom_fact)]

    t, c, k = interpolate.splrep(bins_zoom_sp, hist_zoom, k=degree, s = s)
    spline = interpolate.BSpline(t, c, k, extrapolate=True)
    
    max_spline = np.argmax(spline(bins_zoom_sp))
    size_zoom = len(bins_zoom_sp)
    spline_local = spline(bins_zoom_sp[max_spline-2:size_zoom-1])
    bins_local = bins_zoom_sp[max_spline-2:size_zoom-1]
    
    idxc = np.argmax(spline_local)
    idxc1 = np.argmin(spline_local)
    
    xc = bins_local[idxc]
    xc1 = bins_local[idxc1]
    
    #local_minima = bins_zoom[np.argmin(spline(bins_zoom_sp[idxc:len(bins_zoom_sp)-1]))]

    if plot:
        plt.plot(bins_zoom, hist_zoom, color="blue", label="Histogram")
        plt.plot(bins_zoom_sp, spline(bins_zoom_sp), color="red", label='B-spline Approximation')
        plt.axvline(x=xc, color='green', linestyle='--', label='Min value')
        plt.axvline(x=xc1, color='purple', linestyle='--', label='Max value')
        plt.title('Histogram zoomed')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        

    return spline, xc, xc1


sp, xmax, xmin = histogram_image_zoomed_bspline(images_gray["004.bmp"], 2, 15000)

#%%

for filename, image in images_gray.items():
    histogram_image_zoomed_bspline(images_gray[filename], 2, 15000)

# %%
'''
Now what we can do is to compute the value pf the threshold in segmentation for all the images and apply it, 
taking into account that for all the images, the values of n that will bw chosen will be the ones aat which the 
graphs start converging and stabilize. These values are n1=4 and n2=4. Then the threshold values are used to segmentate the nucleus
'''

# Capturing the threshold values
keys_images_gray = list(images_gray.keys())
values_images_gray = list(images_gray.values())
thresholded_images = {}
N1 = 2
N2 = 2
for i in range(len(values_images_gray)):
    image = values_images_gray[i]
    
    max_Im = image.max()
    min_Im = image.min()
    Tc0 = max_Im/N1 + min_Im/N2
    im_th = image < Tc0
    im_filled = ndimage.binary_fill_holes(im_th) # this fills all the holes of the nuclei segmented image
    im_cleaned = morphology.opening(im_filled, morphology.square(7)) # this removes all the trash detected around the nucleus
    thresholded_images[keys_images_gray[i]] = im_cleaned
    
 # Plotting the 1st thresholded image
   
im_001_thresholded = thresholded_images["001.bmp"]
plt.imshow(im_001_thresholded, cmap= 'binary')
plt.title('Thresholded image')
plt.axis('off')   
plt.show()











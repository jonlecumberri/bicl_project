# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:05:51 2024

@author: jla
"""


import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import interpolate
from scipy import signal
from skimage import morphology
from skimage import filters
from skimage import measure

# %%
# =============================================================================
# Retrieval of images
# =============================================================================


cwd = os.getcwd()
folder_images = "Dataset 2"
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

plt.imshow(im_001[:, :, 2])
plt.axis("off")
plt.show()


def images_to_gray_scale(ims):
    images_gray = {}
    i = 1
    for filename, image in ims.items():
        if i % 2 != 0:
            im_g = images[filename][:, :, 0]
            images_gray[filename] = im_g / im_g.max()
        i = i + 1
    return images_gray


images_gray = images_to_gray_scale(images)


im_001_gray = images_gray["001.bmp"]
plt.imshow(im_001_gray, cmap="gray")
plt.axis("off")
plt.show()


# %%
# =============================================================================
# Threshold estimation for nucleus segmentation
# =============================================================================

maxIm = im_001_gray.max()
minIm = im_001_gray.min()


n1 = np.linspace(1, 100, 100)
n2 = 3
n2_2 = minIm * n1 / maxIm
n2_3 = minIm / (1 - maxIm/n1)

sensN1 = (maxIm) / (maxIm + minIm * n1/n2_3)

plt.plot(n1[0:10], -sensN1[0:10])
plt.xlabel("n1")
plt.ylabel("Υn1")
plt.title("Sensivity for image 1")
plt.show()

for n in images_gray:
    im = images_gray[n]

    maxIm = im.max()
    minIm = im.min()
    print(maxIm, minIm, "\n")

    n1 = np.linspace(1, 100, 100)
    #n2 = 3
    #n2_2 = minIm * n1 / maxIm
    n2 = minIm / (1 - maxIm/n1)

    sensN1 = (maxIm) / (maxIm + minIm * n1/n2)

    plt.plot(n1[0:10], -sensN1[0:10])

plt.xlabel("n1")
plt.ylabel("Υn1")
plt.show()


for n in images_gray:
    im = images_gray[n]

    maxIm = im.max()
    minIm = im.min()

    n2 = np.linspace(1, 100, 100)
    #n2 = 3
    #n2_2 = minIm * n1 / maxIm
    n1 = maxIm / (1 - minIm/n2)

    sensN2 = (minIm) / (minIm + maxIm * n2/n1)

    plt.plot(n2[0:10], -sensN2[0:10])

plt.xlabel("n2")
plt.ylabel("Υn2")
plt.show()

n1 = 2
n2 = 4


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


histogram_image_zoomed(images_gray["004.bmp"])


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
   # if idcx1 == (len(spline_local) -1))
    
    xc = bins_local[idxc]
    xc1 = bins_local[idxc1]
    
    #local_minima = bins_zoom[np.argmin(spline(bins_zoom_sp[idxc:len(bins_zoom_sp)-1]))]

    if plot:
        plt.plot(bins_zoom, hist_zoom, color="blue", label="Histogram")
        plt.plot(bins_zoom_sp, spline(bins_zoom_sp), color="red", label='B-spline Approximation')
        plt.axvline(x=xc, color='green', linestyle='--', label='Max value')
        plt.axvline(x=xc1, color='purple', linestyle='--', label='Min value')
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

#%%

local_extremes = {}
for filename, image in images_gray.items():
    sp, xc, xc1 = histogram_image_zoomed_bspline(images_gray[filename], 2, 15000, plot = False)
    local_extremes[filename] = (xc, xc1)
    
#%% improved version

def histogram_image_zoomed_bspline2(image, degree, s, plot=True):
    size = 256
    hist = ndimage.histogram(image, 0, 1, size)
    bins = np.linspace(0, 1, size)
    
    t, c, k = interpolate.splrep(bins, hist, k=degree, s = s)
    spline = interpolate.BSpline(t, c, k, extrapolate=True)
    spline_values = spline(bins)
    
    index1 = np.argmax(spline_values > 1)
    index2 = np.uint8(256/2)
    
    idxc = np.argmax(spline_values[index1:index2])
    maxima = bins[idxc + index1]
    
    not_finished = True
    while not_finished:
        idxc1 = np.argmin(spline_values[(idxc + index1):index2])
        minima = bins[idxc1 + idxc + index1]
        if idxc1 == (len(spline_values[idxc:index2]) -1):
            #print("In")
            index2 = index2 + 10
        else:
            not_finished = False
        
    #print(maxima, minima)
    plt.plot(bins[index1:index2+20], hist[index1:index2+20], color="blue", label="Histogram")
    plt.plot(bins[index1:index2+20], spline_values[index1:index2+20], color="red", label='B-spline Approximation')
    plt.axvline(x=maxima, color='green', linestyle='--', label='Max value')
    plt.axvline(x=minima, color='blue', linestyle='--', label='Min value')
    plt.show()
    
    return maxima, minima
    
    
#%%
local_extremes = {}
for filename, image in images_gray.items():
    xc, xc1 = histogram_image_zoomed_bspline2(images_gray[filename], 2, 45000, plot = False)
    local_extremes[filename] = (xc, xc1)

#%%

keys_to_keep = ['001.bmp', '002.bmp', '003.bmp', '004.bmp', '005.bmp', '006.bmp', '007.bmp', '008.bmp', '009.bmp', '010.bmp']

#keys_to_keep = ["007.bmp"]

first_10_images = {key: images_gray[key] for key in keys_to_keep if key in images_gray}


#%% conditions for et
Ers_final = []

for filename, image in images_gray.items():
    Tnco_i = Tncos[filename]
    max_local_i = local_extremes[filename][0]
    min_local_i = local_extremes[filename][1]
    
    print("Tnco:", Tnco_i, ", Min:", round(min_local_i,3), ", Max: ", round(max_local_i,3))
    
    diff = np.abs(Tnco_i - min_local_i)
    print("Diff: ", round(diff,3))
    
    Ers = np.linspace(0.12, 0.18, 5)
    #Ers_final = []
    
    for Er in Ers:
    
        if max_local_i < Tnco_i:
            th = Tnco_i
        if (np.abs(Tnco_i) - min_local_i) < Er:
            th = (Tnco_i + min_local_i)/2
            
        if (np.abs(Tnco_i) - min_local_i) > Er:
            Er_final = Er
            print("Er computation")
            if Tnco_i > min_local_i:
                th = ((Tnco_i + min_local_i + Er)/2)
            else:
                th = ((Tnco_i + min_local_i - Er)/2)
            Ers_final.append(Er_final)
        
        
        w = image.shape[0]
        h = image.shape[1]
        im_new = image.copy()
        for y in range(h):
            for x in range(w):
                pix = image[x,y]
                if pix < th:
                    i = image.min()
                else:
                    i = pix
                
                im_new[x,y] = i
        
    
        plt.imshow(im_new, cmap = "gray")
        plt.title("Image: " + str(filename) +  " Er:"  + str(round(Er,3)))
        plt.axis("off")
        plt.show()
        

print(Ers_final)

mean_ers_final = np.mean(Ers_final)

print("Mean value fo all Er parameters used:", mean_ers_final)
#%%

def compute_thresholded_nucleus(images_set):
    th_images = {}
    for filename, image in images_set.items():
        Tnco_i = Tncos[filename]
        max_local_i = local_extremes[filename][0]
        min_local_i = local_extremes[filename][1]
        
        print("Tnco:", Tnco_i, ", Min:", round(min_local_i,3), ", Max: ", round(max_local_i,3))
        
        diff = np.abs(Tnco_i - min_local_i)
        print("Diff: ", round(diff,3))
        
        
        Er = 0.1388
        
        if max_local_i < Tnco_i:
            th = Tnco_i
        if (np.abs(Tnco_i) - min_local_i) < Er:
            th = (Tnco_i + min_local_i)/2
                
        if (np.abs(Tnco_i) - min_local_i) > Er:
           
            if Tnco_i > min_local_i:
                th = ((Tnco_i + min_local_i + Er)/2)
            else:
                th = ((Tnco_i + min_local_i - Er)/2)
                
            
            
        w = image.shape[0]
        h = image.shape[1]
        im_new = image.copy()
        for y in range(h):
            for x in range(w):
                pix = image[x,y]
                if pix < th:
                    i = image.min()
                else:
                    i = 1
                    
                im_new[x,y] = i
        
        black_mask = (im_new == image.min())
        filled_black_regions = ndimage.binary_fill_holes(black_mask)
        im_filled = np.where(filled_black_regions, image.min(), im_new)
        
        
        # labels, num_features = ndimage.label(filled_black_regions)
        # sizes = ndimage.sum(black_mask, labels, range(num_features + 1))
        # min_size = 15
        # filtered_black_regions = np.where(sizes > min_size, labels, 0)
        # result_image = np.where(filtered_black_regions > 0 , image.min(), im_new)
        
        # im_filled_bool = im_filled.astype(bool)
        # min_size_threshold = 5
        # filtered_image_bool  = morphology.remove_small_objects(im_filled_bool, min_size=min_size_threshold, connectivity=1)
        # result_image = filtered_image_bool.astype(int)
        
        # labeled_image, num_features = ndimage.label(filtered_image)
        # sizes = ndimage.sum(filtered_image, labeled_image, range(1, num_features + 1))
        # largest_index = np.argmax(sizes) + 1
        # largest_component_mask = labeled_image == largest_index
        # result_image = np.where(largest_component_mask, im_filled.min(), 1)
        
        inverted_image = im_filled.max() - im_filled
        labeled_image, num_features = ndimage.label(inverted_image)
        sizes = ndimage.sum(inverted_image, labeled_image, range(1, num_features + 1))
        sorted_indices = np.argsort(sizes)[::-1]
        top_indices = sorted_indices[:3]
        filtered_image = np.zeros_like(im_filled)
        for idx in top_indices:
            filtered_image[labeled_image == idx + 1] = 1
        result_image = im_filled.max() - filtered_image
        
        threshold_size = sizes[top_indices[-1]] * 0.5
        filtered_indices = [idx for idx in top_indices if sizes[idx] >= threshold_size]
        filtered_image2 = np.zeros_like(im_filled)
        for idx in filtered_indices:
            filtered_image2[labeled_image == idx + 1] = 1
        result_image2 = im_filled.max() - filtered_image2

        
        # plt.imshow(im_filled, cmap = "gray")
        # plt.title("Image: " + str(filename))
        # plt.axis("off")
        # plt.show()
        
        plt.subplot(121)
        plt.imshow(image, cmap = "gray")
        plt.title("Image: " + str(filename))
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(result_image, cmap = "gray")
        plt.title("Image: " + str(filename))
        plt.axis("off")
        
        plt.show()
        
        
        # plt.imshow(result_image2, cmap="gray")
        # plt.title("Result Image with Retained Biggest Black Regions")
        # plt.axis("off")
        # plt.show()
        
        th_images[filename] = im_filled

    return th_images


th_10_first_images = compute_thresholded_nucleus(first_10_images)


#%% Threshold estimation for WBC segmentation


for filename, image in first_10_images.items():
    
    print(filename)
    Tnco_i = Tncos[filename]
    al = image.min()
    au = image.max()/2
    n = 5
    
    ers = np.linspace(al, au, n)
    Twbcs = (image.max() + Tnco_i)/2 + ers
    
    ims_th_wbc = []
    for Twbc in Twbcs:
        print(Twbc)
        w = image.shape[0]
        h = image.shape[1]
        im_new = image.copy()
        for y in range(h):
            for x in range(w):
                pix = image[x,y]
                if pix < Twbc:
                    i = Twbc
                else:
                    i = pix
                    
                im_new[x,y] = i
                
        ims_th_wbc.append(im_new)
    
    ims_mean_th_wbc = sum(ims_th_wbc) / n
    threshold_value = filters.threshold_otsu(ims_mean_th_wbc)
    binary_image = ims_mean_th_wbc > threshold_value
    
    
    
    plt.imshow(binary_image, cmap="gray")
    plt.title("Result Image mean Th WBC: " + str(filename))
    plt.axis("off")
    plt.show()
    




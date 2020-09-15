import os
import time

import cv2
import numpy as np
from skimage import segmentation

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import tarfile
import skimage
from sklearn.mixture import GaussianMixture
import warnings
import scipy.ndimage as ndimage
from skimage import measure
import cv2 as cv
from sklearn.mixture import GaussianMixture
import scipy.stats 
from skimage.segmentation import active_contour
from skimage.filters import gaussian
warnings.filterwarnings('ignore')
import matplotlib.patches as patchess
from collections import  Counter


    
def find_neighbor(image, start_x, start_y,  threshold_1=127):
    img_len = image.shape[0]
    img_width = image.shape[1]
    #new_image=np.zeros((img_len,img_width))
    new_x=np.arange(start_x-1,start_x + 2)
    new_y=np.arange(start_y-1,start_y + 2)
    candidate=[]
    for temp_x in new_x:
        for temp_y in new_y:
            # discard neighbor points that are beyond the size of image
            if temp_x<0 or temp_x>=img_len:
                continue
            if temp_y<0 or temp_y>=img_width:
                continue
            # only get new neighbor points that don't exist in our candidate list
            if [temp_x, temp_y] in candidate:
                continue
            # no need to consider the center point itself
            if temp_x==start_x and temp_y==start_y:
                continue
            #print(image[start_x][start_y],image[temp_x][temp_y],np.abs(image[temp_x][temp_y]-image[start_x][start_y]))
            '''
            if  image[start_x][start_y]>threshold_1:
                if image[temp_x][temp_y] > threshold_1:
                    candidate.append([temp_x,temp_y])
            else:
                if image[temp_x][temp_y] > threshold_2 and image[temp_x][temp_y] < threshold_1 :
                    candidate.append([temp_x,temp_y])
            '''
            if image[temp_x][temp_y] > threshold_1:
                candidate.append([temp_x,temp_y])
            
                    
    return candidate


def region_growing(image,start_x,start_y,threshold_1=127):
    # initialization
    img_len = image.shape[0]
    img_width = image.shape[1]
    new_image=np.zeros((img_len,img_width))
    candidates=[[start_x,start_y]]
    
    while len(candidates)>0:
        temp_point = candidates.pop()
        candidate_neigbor=find_neighbor(image, temp_point[0], temp_point[1],threshold_1=threshold_1)
        for temp_neibor in candidate_neigbor:
            if new_image[temp_neibor[0]][temp_neibor[1]]==0:
                # if the candidate neignbor is not in current list, we append the point
                # and set the corresponding pixel in image to 1
                new_image[temp_neibor[0]][temp_neibor[1]]=1
                candidates.append(temp_neibor)
    return new_image,candidates


def run(input_image_path = 'image/wheat.png',avg_image_path='image/rice_image/rice_avg.png',kernel_size=(15,8),min_elongation=1.8, ref_area=1500):
    start_time0 = time.time()

    image = cv2.imread(input_image_path)
    image_avg=cv2.imread('image/rice_image/rice_avg.png')
    image=image-image_avg
    
    new_width = int(image.shape[1]/4)
    new_height = int(image.shape[0]/4)
    new_dim = (new_width, new_height)
    
    image = cv2.resize(image, new_dim, interpolation = cv2.INTER_AREA)
    
    gray_image=256-skimage.color.rgb2gray(image)*256
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
    eroded_image_gray=cv2.erode(gray_image,kernel)

    
    
    #time0 = time.time() - start_time0
    #time1 = time.time() - start_time1
    #print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))
    #cv2.imwrite("seg_%s_%ds.jpg" % (args.input_image_path[6:-4], time1), show)
    
    origin_img=cv2.imread(input_image_path)
    new_width = int(origin_img.shape[1]/4)
    new_height = int(origin_img.shape[0]/4)
    new_dim = (new_width, new_height)
    origin_img = cv2.resize(origin_img, new_dim, interpolation = cv2.INTER_AREA)
    origin_img_gray = cv2.cvtColor(origin_img,cv2.COLOR_RGB2GRAY)
    
    plt.imshow(origin_img)
    plt.title('Original Image')
    plt.show()
    
    threshold_1=127
    
    ellipse_x=[]
    ellipse_y=[]
    ellipses=[]
    grain_pixels=[[temp_x, temp_y] for temp_x in range(eroded_image_gray.shape[0]) for temp_y in range(eroded_image_gray.shape[1]) 
                  if eroded_image_gray[temp_x][temp_y]>threshold_1] 
    i=0
    while True:
        temp_len=len(grain_pixels)
        #print(temp_len)
        temp_index=np.random.randint(temp_len)
        temp_x,temp_y=grain_pixels[temp_index]
        mask,_ = region_growing(eroded_image_gray, threshold_1=threshold_1, start_x=temp_x,start_y=temp_y)
        region=np.sum(mask)
        temp_cand=[[temp_x, temp_y] for temp_x in range(mask.shape[0]) for temp_y in range(mask.shape[1]) 
                  if mask[temp_x][temp_y]>0] 
        #print(temp_cand)
        for temp_coord in temp_cand:
            grain_pixels.remove(temp_coord)
        #kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
        new_mask=cv2.dilate(mask,kernel)
        region=np.sum(new_mask)
        if region>100. and region<1200:
            #plt.imshow(new_mask*100+origin_img_gray, cmap='gray')
            #plt.title('the size of pixels of grain is '+str(region))
            #plt.show()

            temp_new_mask=new_mask.astype('uint8')
            temp_contours, _ = cv2.findContours(temp_new_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            temp_ellipse = cv2.fitEllipse(temp_contours[0])
            temp_image=origin_img_gray.copy()
            cv2.ellipse(temp_image, temp_ellipse, (255, 255, 0), 2)
            ellipses.append(temp_ellipse)
            '''
            plt.imshow(temp_image)
            plt.show()
            '''
            ellipse_x.append(temp_ellipse[1][0])
            ellipse_y.append(temp_ellipse[1][1])
            i+=1
        if i>300 or len(grain_pixels)<20:
            break
    
    temp_image=origin_img_gray.copy()
    
    for i in range(len(ellipses)):
        temp_ellipse=ellipses[i]
        temp_x=temp_ellipse[1][0]
        temp_y=temp_ellipse[1][1]
        temp_area=np.pi*temp_x*temp_y
        if not(temp_x/temp_y>1/min_elongation and temp_x/temp_y<min_elongation and temp_area>ref_area):
            cv2.ellipse(temp_image, temp_ellipse, (255, 255, 0), 2)
    plt.imshow(temp_image)
    plt.title('Ellipse Kernel '+str(kernel_size))
    plt.show()
    
    
    
    _,_,_=plt.hist(x=ellipse_x)
    plt.title('Long Axis of Ellipses')
    plt.show()
    
    _,_,_=plt.hist(x=ellipse_y)
    plt.title('Short Axis of Ellipses')
    plt.show()
    
    ellipse_area=np.array(ellipse_x)*np.array(ellipse_y)*np.pi
    _,_,_=plt.hist(x=ellipse_area)
    plt.title('Areas of Fit Ellipses')
    plt.show()
    

if __name__ == '__main__':
    run(input_image_path = 'image/rice_image/rice.png',avg_image_path='image/rice_image/rice_avg.png')

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

class Args(object):
    #input_image_path = 'image/wheat.png'  # image/coral.jpg image/tiger.jpg
    #input_image_path = 'seg_bean_min_8.jpg'
    
    
    train_epoch = 2 ** 6
    mod_dim1 = 128  #
    mod_dim2 = 32
    gpu_id = 0
    
    min_label_num = 8   # if the label number small than it, break loop
    max_label_num = 256  # if the label number small than it, start to show result image.


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),)

        
    def forward(self, x):
        return self.seq(x)
    
# define a funcion that return 8 neighbors of a start point that satisfied the set threshold
def find_neighbor(image, start_x, start_y,  threshold_1,threshold_2):
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
            if image[temp_x][temp_y] > threshold_2:
                candidate.append([temp_x,temp_y])
            
                    
    return candidate


def region_growing(image,start_x,start_y,threshold_1,threshold_2):
    # initialization
    img_len = image.shape[0]
    img_width = image.shape[1]
    new_image=np.zeros((img_len,img_width))
    candidates=[[start_x,start_y]]
    
    while len(candidates)>0:
        temp_point = candidates.pop()
        candidate_neigbor=find_neighbor(image, temp_point[0], temp_point[1],threshold_1,threshold_2)
        for temp_neibor in candidate_neigbor:
            if new_image[temp_neibor[0]][temp_neibor[1]]==0:
                # if the candidate neignbor is not in current list, we append the point
                # and set the corresponding pixel in image to 1
                new_image[temp_neibor[0]][temp_neibor[1]]=1
                candidates.append(temp_neibor)
    return new_image,candidates


def run(input_image_path = 'image/wheat.png'):
    start_time0 = time.time()

    args = Args()
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0
    image = cv2.imread(input_image_path)
    
    new_width = int(image.shape[1]/4)
    new_height = int(image.shape[0]/4)
    new_dim = (new_width, new_height)
    
    image = cv2.resize(image, new_dim, interpolation = cv2.INTER_AREA)

    '''segmentation ML'''
    seg_map = segmentation.felzenszwalb(image, scale=32, sigma=0.5, min_size=64)
    # seg_map = segmentation.slic(image, n_segments=10000, compactness=100)
    plt.imshow(seg_map,cmap="gray")
    #cv2.imwrite("superpixel", seg_map)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)

    model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-1, momentum=0.0)

    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image

    '''train loop'''
    start_time1 = time.time()
    model.train()
    for batch_idx in range(args.train_epoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        '''show image'''
        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:  # update show
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)
        cv2.imshow("seg_pt", show)
        cv2.waitKey(1)

        print('Loss:', batch_idx, loss.item())
        if len(un_label) < args.min_label_num:
            break

    '''save'''
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
    
    plt.imshow(show)
    plt.title('Segmentation Result after Unsupervised CNN')
    plt.show()
    
    image=show.copy()
    gray_image=skimage.color.rgb2gray(image)*256
    
    
    temp_img=gray_image.copy()
    temp_img=-temp_img
    threshold_1=-skimage.filters.threshold_minimum(temp_img)

    temp_img=gray_image.copy()
    temp_img=temp_img[temp_img<threshold_1]
    temp_img=-temp_img
    threshold_2=-skimage.filters.threshold_minimum(temp_img)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    eroded_image_gray=cv2.erode(gray_image,kernel)
    
    
    ellipse_x=[]
    ellipse_y=[]
    grain_pixels=[[temp_x, temp_y] for temp_x in range(eroded_image_gray.shape[0]) for temp_y in range(eroded_image_gray.shape[1]) 
                  if eroded_image_gray[temp_x][temp_y]>threshold_2] 
    i=0
    while True:
        temp_len=len(grain_pixels)
        #print(temp_len)
        temp_index=np.random.randint(temp_len)
        temp_x,temp_y=grain_pixels[temp_index]
        mask,_ = region_growing(eroded_image_gray,temp_x,temp_y, threshold_1,threshold_2)
        #region=np.sum(mask)
        temp_cand=[[temp_x, temp_y] for temp_x in range(mask.shape[0]) for temp_y in range(mask.shape[1]) 
                  if mask[temp_x][temp_y]>0] 
        #print(temp_cand)
        for temp_coord in temp_cand:
            grain_pixels.remove(temp_coord)
        new_mask=cv2.dilate(mask,kernel)
        region=np.sum(new_mask)
        if region>100. and region<1200:
            plt.imshow(new_mask*100+origin_img_gray, cmap='gray')
            #plt.subplot(1, 2, 1).imshow(new_mask*100+gray_image , cmap = 'gray')
            #plt.subplot(1, 2, 2).imshow(gray_image , cmap = 'gray')
            plt.title('the size of pixels of grain is '+str(region))
            plt.show()

            temp_new_mask=new_mask.astype('uint8')
            temp_contours, _ = cv2.findContours(temp_new_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            temp_ellipse = cv2.fitEllipse(temp_contours[0])
            temp_image=origin_img_gray.copy()
            cv2.ellipse(temp_image, temp_ellipse, (255, 255, 0), 2)
            plt.imshow(temp_image)
            plt.show()
            ellipse_x.append(temp_ellipse[1][0])
            ellipse_y.append(temp_ellipse[1][1])
            i+=1
        if i>100 or len(grain_pixels)<50:
            break
    
    
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
    run(input_image_path = 'image/wheat_image/wheat.png')
    #run(input_image_path = 'image/wheat_image/frame_00350.png')

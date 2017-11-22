import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import random
import scipy.misc
import os
random.seed(234)

import cv2
from multiprocessing import Pool
import numpy as np
datadir='./maskdata/fake1/'
target='./maskdata/faketrans/'
tmimg=cv2.imread('./maskdata/fake1/14238_2.jpg')


tmimg=cv2.imread('./maskdata/fake1/14238_2.jpg')
size = (tmimg.shape[0])/2
img,mask = tmimg[:size,:,:].astype('uint8'),tmimg[size:,:,:].astype('uint8')
minx,maxx,miny,maxy=10000,0,100000,0
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i,j,0]>200:
            minx=min(i,minx)
            maxx=max(i,maxx)
            miny=min(j,miny)
            maxy=max(j,maxy)
            mask[i,j,:]=255
        else:
            mask[i,j,:]=0
center = ((miny+maxy)/2,(minx+maxx)/2)
nmask = mask[minx:maxx,miny:maxy,:]
src = img[minx:maxx,miny:maxy]
# plt.figure()
# plt.imshow(src)
src= np.multiply(img[minx:maxx,miny:maxy],nmask/255)

# plt.figure()
# plt.imshow(nmask)
# plt.figure()
# plt.imshow(src)
final = cv2.seamlessClone(src,img,nmask,center,cv2.NORMAL_CLONE)
# plt.figure()
# plt.imshow(final)


import math
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import rescale
import time
import cv2

PATH_IMG = '/home/thangnx/code/human-count/part_A/train_data/images/IMG_'
PATH_DMAP = '/home/thangnx/code/human-count/part_A/train_data/dmap/DMAP_IMG_'
PATH_TRAIN = '/home/thangnx/code/human-count/train/' 

def ReadImage(imPath,mirror = False,scale=1.0):
    """
    Read gray images.
    """ 
    imArr = np.array(Image.open(imPath).convert('L'))

    imArr = imArr[:,:, np.newaxis]
    imArr = np.tile(imArr,(1,1,3))
    
    return imArr

def ReadDmap(dmapPath,mirror = False,scale = 1.0):
    """
    Load the density map from matfile.
    """ 
    dmap = sio.loadmat(dmapPath)
    densitymap = dmap['DMAP']
    return densitymap
    



def CropSubImage(imArr,dmapArr,downscale = 8.0):
    """
    imArr:  height * width *channel(rgb = 3,gray = 1) *
    dmapArr: corrsponding downscaled density map
    return  9 sub-images and sub-densitymap
    """
    imShape = imArr.shape
    dmapHeight = dmapArr.shape[0]
    dmapWidth = dmapArr.shape[1]
    imHeight = imShape[0]
    imWidth = imShape[1]
    subimHeight = imHeight/2
    subimWidth = imWidth/2
    subdmapHeight = int(subimHeight/downscale)
    subdmapWidth = int(subimWidth/downscale)
    
    #print subimHeight,subimWidth
    #print dmapHeight,dmapWidth
    #print subdmapHeight,subdmapWidth
    #time.sleep(100)
    
    subimArr1 = imArr[0:subimHeight,0:subimWidth,:]
    subimArr2 = imArr[0:subimHeight,imWidth-subimWidth:imWidth,:]
    subimArr3 = imArr[imHeight-subimHeight:imHeight,0:subimWidth,:]
    subimArr4 = imArr[imHeight-subimHeight:imHeight,\
                      imWidth-subimWidth:imWidth,:]
    subimArr5 = imArr[subimHeight/2:subimHeight+subimHeight/2,\
                      0:subimWidth,:]
    subimArr6 = imArr[0:subimHeight,subimWidth/2:\
                      subimWidth+subimWidth/2,:]
    subimArr7 = imArr[imHeight-subimHeight:imHeight,\
                      subimWidth/2:subimWidth+subimWidth/2,:]
    subimArr8 = imArr[subimHeight/2:subimHeight+subimHeight/2,\
                      imWidth-subimWidth:imWidth,:]
    subimArr9 = imArr[subimHeight/2:subimHeight+subimHeight/2,\
                      subimWidth/2:subimWidth+subimWidth/2,:]
                      
    subdmapArr1 = dmapArr[0:subdmapHeight,0:subdmapWidth]
    subdmapArr2 = dmapArr[0:subdmapHeight,dmapWidth-subdmapWidth:dmapWidth]
    subdmapArr3 = dmapArr[dmapHeight-subdmapHeight:dmapHeight,0:subdmapWidth]
    subdmapArr4 = dmapArr[dmapHeight-subdmapHeight:dmapHeight,\
                      dmapWidth-subdmapWidth:dmapWidth]
    subdmapArr5 = dmapArr[subdmapHeight/2:subdmapHeight+subdmapHeight/2,\
                      0:subdmapWidth]
    subdmapArr6 = dmapArr[0:subdmapHeight,subdmapWidth/2:\
                      subdmapWidth+subdmapWidth/2]
    subdmapArr7 = dmapArr[dmapHeight-subdmapHeight:dmapHeight,\
                      subdmapWidth/2:subdmapWidth+subdmapWidth/2]
    subdmapArr8 = dmapArr[subdmapHeight/2:subdmapHeight+subdmapHeight/2,\
                      dmapWidth-subdmapWidth:dmapWidth]
    subdmapArr9 = dmapArr[subdmapHeight/2:subdmapHeight+subdmapHeight/2,\
                      subdmapWidth/2:subdmapWidth+subdmapWidth/2]
    return subimArr1,subimArr2,subimArr3,subimArr4,subimArr5,subimArr6,\
           subimArr7,subimArr8,subimArr9,subdmapArr1,subdmapArr2,subdmapArr3,\
           subdmapArr4,subdmapArr5,subdmapArr6,subdmapArr7,subdmapArr8,subdmapArr9
  
def make_train_set(path_to_img, path_to_dmap,path_to_train):
    for i in range(1,301):
        img = ReadImage(path_to_img+str(i)+'.jpg')
        dmap = ReadDmap(path_to_dmap+str(i)+'.mat')

        li = CropSubImage(img,dmap)

        for id in range(0,18):
            if id < 9:
                #cv2.imwrite(path_to_train+'images/IMG_'+str(i*9-9 + id + 1) + '.jpg', li[id])
		continue
            else:
                sio.savemat(path_to_train+'dmap/DMAP_IMAGE'+str(i*9-9 + id + 1 - 9)+'.mat',{'DMAP':li[id]})

make_train_set(PATH_IMG, PATH_DMAP, PATH_TRAIN)

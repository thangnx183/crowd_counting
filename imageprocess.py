import cv2
import scipy.io as io
import matplotlib.pyplot as plt 
from os import listdir
from os.path import join, isfile
import numpy as np 

PATH_IMG = '/home/thangnx/code/human-count/part_B/train_data/images/'
PATH_DMAP = '/home/thangnx/code/human-count/part_B/train_data/dmap/'
PATH_TRAIN = '/home/thangnx/code/human-count/train_B/' 

def crop(img,dmap, downscale= 8.0):
    
    imHeight = img.shape[0]
    imWidth  = img.shape[1]
    
    subimHeight = img.shape[0] / 2
    subimWidth  = img.shape[1] / 2
    
    #subdmapHeight = int(subimHeight / downscale)
    #subdmapWidth  = int(subimWidth / downscale)
    
    #imsub1 = img[0:subimHeight,0:subimWidth,:]
    #imsub2 = img[0:subimHeight,imWidth-subimWidth:imWidth, :]
    #imsub3 = img[imHeight-subimHeight:imHeight,0:subimWidth,:]
    #imsub4 = img[imHeight-subimHeight:imHeight,imWidth-subimWidth:imWidth,:]
    #imsub5 = img[subimHeight/2:subimHeight+subimHeight/2,0:subimWidth, :]
    #imsub6 = img[0:subimHeight,subimWidth/2:subimWidth+subimWidth/2, :]
    #imsub7 = img[imHeight-subimHeight:imHeight, subimWidth/2:subimWidth+subimWidth/2, :]
    #imsub8 = img[subimHeight/2:subimHeight+subimHeight/2, imWidth-subimWidth:imWidth, :]
    #imsub9 = img[subimHeight/2:subimHeight+subimHeight/2,subimWidth/2:subimWidth+subimWidth/2, :]
    
    subim1, subdmap1 = crop_piece(img,dmap,0,0, downscale)
    subim2, subdmap2 = crop_piece(img,dmap,0,imWidth-subimWidth, downscale)
    subim3, subdmap3 = crop_piece(img,dmap,imHeight-subimHeight,0, downscale)
    subim4, subdmap4 = crop_piece(img,dmap,imHeight-subimHeight,imWidth-subimWidth, downscale)
    subim5, subdmap5 = crop_piece(img,dmap,subimHeight/2 ,0, downscale)
    subim6, subdmap6 = crop_piece(img,dmap,0,subimWidth/2, downscale)
    subim7, subdmap7 = crop_piece(img,dmap,imHeight-subimHeight,subimWidth/2, downscale)
    subim8, subdmap8 = crop_piece(img,dmap,subimHeight/2,imWidth-subimWidth, downscale)
    subim9, subdmap9 = crop_piece(img,dmap,subimHeight/2,subimWidth/2, downscale)

    return [subim1, subim2, subim3, subim4, subim5, subim6, subim7, subim8, subim9,
            subdmap1, subdmap2, subdmap3, subdmap4, subdmap5, subdmap6, subdmap7, subdmap8, subdmap9 ]
    
    

def crop_piece(img, dmap, x, y, downscale=8.0):
    #imHeight = img.shape[0]
    #imWidth  = img.shape[1]
    
    subimHeight = img.shape[0] / 2
    subimWidth  = img.shape[1] / 2
    
    subdmapHeight = int(subimHeight / downscale)
    subdmapWidth  = int(subimWidth / downscale)
    
    subimg = img[x:x+subimHeight, y: y+ subimWidth, :]
    #subdmap = dmap[dmapx: dmapx+subdmapHeight, dmapy: dmapy+subdmapWidth]
    
    #dmapx = x + int(subimHeight / 2.0 - subimHeight/downscale/2.0 )
    #dmapy = y + int(subimWidth /2.0 - subimWidth/downscale/2.0)
    
    #dmapx = x + int(subimHeight - subimHeight/downscale)
    #dmapy = y
    subdmap = 0
    count = 0
    dmapx = x + int(subimHeight - downscale*subdmapHeight)
    #dmapy = y + int(subimWidth -  downscale*subdmapWidth)
    #print 'x , y : {},{}'.format(x,y)

    while dmapx < x + subimHeight and count < downscale*downscale :
        dmapy = y + int(subimWidth -  downscale*subdmapWidth)
        while dmapy < y + subimWidth  and count < downscale*downscale:
            #print "dmapx, dmapy : {},{}".format(dmapx,dmapy)
            subdmap = subdmap + dmap[dmapx: dmapx+subdmapHeight, dmapy: dmapy+subdmapWidth]
            count += 1
            dmapy += subdmapWidth
            #print count
        dmapx += subdmapHeight
        #print 'dmapx : {} and x +subimHeight: {}'.format(dmapx, x + subimHeight)

    #subdmap = dmap[dmapx: dmapx+subdmapHeight, dmapy: dmapy+subdmapWidth]
    print count 
    return subimg, subdmap/(downscale*downscale)


def crop_piece_2(img, dmap, x, y, downscale=8.0):
    #imHeight = img.shape[0]
    #imWidth  = img.shape[1]
    
    subimHeight = img.shape[0] / 2
    subimWidth  = img.shape[1] / 2
    
    subdmapHeight = int(subimHeight / downscale)
    subdmapWidth  = int(subimWidth / downscale)
    
    subimg = img[x:x+subimHeight, y: y+ subimWidth, :]
    #subdmap = dmap[dmapx: dmapx+subdmapHeight, dmapy: dmapy+subdmapWidth]
    
    dmapx = x + int(subimHeight / 2.0 - subimHeight/downscale/2.0 )
    dmapy = y + int(subimWidth /2.0 - subimWidth/downscale/2.0)

    subdmap = dmap[dmapx: dmapx+subdmapHeight, dmapy: dmapy+subdmapWidth]

    return subimg, subdmap
    



def make_train_set(path_to_img, path_to_dmap,path_to_train):
    imgs = [f for f in listdir(path_to_img) if isfile(join(path_to_img,f))]
    dmaps = [f for f in listdir(path_to_dmap) if isfile(join(path_to_dmap,f))]

    print len(imgs)
    print len(dmaps)

    #img = cv2.imread(path_to_img+'IMG_'+str(1)+'.jpg')
    #dmap = io.loadmat(path_to_dmap+'DMAP_IMG_'+str(1)+'.mat')
    #dmap = np.array(dmap['DMAP'])

    #li = crop(img, dmap)
    
    for id in range(len(imgs)):
        img = cv2.imread(path_to_img+'IMG_'+str(id + 1)+'.jpg')
        dmap = io.loadmat(path_to_dmap+'DMAP_IMG_'+str(id + 1)+'.mat')
        dmap = np.array(dmap['DMAP'])

        li = crop(img, dmap)
        
        for i in range(0,18):
            if i < 9:
                #cv2.imwrite(path_to_train+'images/IMG_'+str(id*9 + i + 1) +'.jpg', li[i])
		continue
            else:
                io.savemat(path_to_train+'avg_dmap/DMAP_IMAGE'+str(id*9 + i + 1 - 9) +'.mat', {'DMAP':li[i] } )
		continue
        



#make_train_set(PATH_IMG, PATH_DMAP, PATH_TRAIN)
img_train = [f for f in listdir(PATH_TRAIN+'images') if isfile(join(PATH_TRAIN+'images',f))] 
H, W  = 0 , 0

for i in range(len(img_train)):
    img = cv2.imread(PATH_TRAIN+'images/'+img_train[i])
    h,w, _ = img.shape
    print h,w
    #img = cv2.resize(img, (436,298), interpolation = cv2.INTER_AREA)
    #print img.shape
    H += h
    W += w 
print "avg"
print H/len(img_train), W/len(img_train)

# 298 436 


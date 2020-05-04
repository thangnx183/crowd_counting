import numpy as np 
import scipy.io as sio
from oct2py import Oct2Py

img_path = './part_B/train_data/images/IMG_'
gt_path = './part_B/train_data/ground-truth/GT_IMG_' 
dmap_path = './part_B/train_data/dmap/DMAP_IMG_'

with Oct2Py() as oc:
    oc.eval('pkg load statistics')
    for i in range(1,301):
        img = img_path+str(i)+'.jpg'
        head = gt_path+str(i)+'.mat'
        dmap = dmap_path + str(i)+'.mat'
        d = oc.density(img,head,7)
        print sum(sum(d))
        sio.savemat(dmap,{"DMAP":d} )




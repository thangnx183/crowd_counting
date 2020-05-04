import numpy as np
import cv2
import scipy.io as io
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 

PATH_TRAIN = '/home/thangnx/code/human-counting/ShanghaiTech/part_A/train_data/images/'
PATH_GT_TRAIN = '/home/thangnx/code/human-counting/ShanghaiTech/part_A/train_data/ground-truth/'
PATH_DMAP_TRAIN = '/home/thangnx/code/human-counting/ShanghaiTech/part_A/train_data/dmap/'

PATH_TEST = '/home/thangnx/code/human-counting/ShanghaiTech/part_A/test_data/images/'
PATH_GT_TEST = '/home/thangnx/code/human-counting/ShanghaiTech/part_A/test_data/ground-truth/'
PATH_DMAP_TEST = '/home/thangnx/code/human-counting/ShanghaiTech/part_A/test_data/dmap/'


def get_heads_location(gt_arr):

    return gt_arr['image_info'][0][0][0][0][0]

def dis_map(heads, k = 7):
    num = len(heads)
    dis_matrix = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            if j > i:
                dis_matrix[i,j] = sum((heads[i] - heads[j])**2)
            
            if j < i :
                dis_matrix[i,j] = dis_matrix[j,i]

    dis_matrix.sort()
    return np.sqrt(dis_matrix[:,:k])
    
def norm2d(i, sigma, center):
    isize = i.shape

    X1, X2 = np.meshgrid(np.linspace(0,isize[0] - 1, isize[0]), np.linspace(0,isize[1] - 1, isize[1]))

    X1 = np.reshape([X1[:,i] for i in range(X1.shape[1])], (X1.shape[0] * X1.shape[1], 1))    
    X2 = np.reshape([X2[:,i] for i in range(X2.shape[1])], (X2.shape[0] * X2.shape[1], 1))

    X = np.column_stack((X1,X2))

    Sigma = np.zeros((len(sigma), len(sigma)))

    for j in range(len(sigma)):
        Sigma[j,j] = sigma[j]**2
    #print Sigma
    #print X
    dense = multivariate_normal.pdf(X, mean=np.array([center[1], center[0]]),cov=Sigma)
    dense = np.reshape(dense, isize)

    return dense

def density(image, heads, k = 7):
    density_map = np.zeros((image.shape[0], image.shape[1]))
    density_map = density_map.astype(np.float64)
    
    distance = dis_map(heads, k)
    
    for i in range(len(heads)):
        var = 0.3 * np.mean(distance[i,:])
        dh = norm2d(density_map,np.array([var, var]), heads[i])
        if sum(sum(dh)) == 0 :
            print 'invalid number'
            print 'var : {0}'.format(var)
            print 'head: {0} , {1}'.format(heads[i,0], heads[i,1])
            break
        dh = dh / sum(sum(dh))
        #print sum(sum(dh))
        if sum(sum(dh)) > 1.1:
            print "bug"
            break
        density_map = density_map + dh
        #print density_map
        #print 'sum : {0}'.format(sum(sum(density_map)))
    
    return density_map

def num_file(path):
    from os import listdir
    from os.path import isfile, join

    files = [f for f in listdir(path) if isfile(join(path,f))]

    return len(files)

'''
generate density map from image and ground truth file
out : store density map
'''

def make_dmap(path_to_image, path_to_gt, path_to_dmap):
    num_image = num_file(path_to_image)
    num_gt = num_file(path_to_gt)

    if num_gt != num_gt:
        print 'no match file'
        pass
    
    for i in range(111,num_gt,1):
        img = cv2.imread(path_to_image+'IMG_'+str(i+1)+'.jpg')
        gt = io.loadmat(path_to_gt+'GT_IMG_'+str(i+1) + '.mat')
        gt = get_heads_location(gt)

        dense = density(img,gt)
        print "human : {0}".format(sum(sum(dense)))

        io.savemat(path_to_dmap+'DMAP_IMG_'+str(i+1)+'.mat', {'DMAP':dense})



img = cv2.imread('./test/IMG_TRAIN_1.jpg')

plt.imshow(img)
plt.show()

#mat = io.loadmat('./test/GT_IMG_TRAIN_1.mat')
#heads = get_heads_location(mat)


#d = density(img,heads )
#print sum(d)
#io.savemat('demo.mat',{'density' : d})

#t = np.ones([img.shape[0], img.shape[1]])
#print t.shape


#norm = norm2d(t, np.array([1.41467951618,1.41467951618]), np.array([834.660679539 , 427.943122287]))
#print sum(sum(norm))

#print num_file(PATH_TEST)

#make_dmap(PATH_TEST, PATH_GT_TEST, PATH_DMAP_TEST)









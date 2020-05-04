import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import scipy.io as io 
import cv2
from os import listdir
from os.path import join, isfile
from PIL import Image

params_path = './count_net'

mat = io.loadmat(params_path)
#print len(mat)

#w,b = mat['conv1_1'][0][0]

#print w.shape
#print b.shape
mean_pixel = [ 123.68, 116.779, 103.939]

def get_img(img):
    img.astype(np.float32)
    print img.shape
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]
    
    return img

def get_num(gt_path):
    gt = io.loadmat(gt_path)

    return gt['image_info'][0][0][0][0][0].shape[0]


def get_params(name):
    w,b = mat[name][0][0]
    b = b.reshape(-1)

    return w,b

def conv2d(input, kernel_size, num_filter, stride=1, pad=0,dilation=1, name=""):
    
    if name :
        weight, bias = get_params(name)


    input = tf.pad(input,[[0,0],[pad,pad],[pad,pad],[0,0]], "CONSTANT")
    conv = tf.nn.conv2d(input, filter=weight, strides=(1,stride, stride,1), padding='VALID',dilations=[1,dilation, dilation,1], name='conv'+name)
    #print bias
    conv = tf.nn.bias_add(conv, bias)
    print conv.get_shape()

    return tf.nn.relu(conv) 

def group_conv2d(input, kernel_size, num_filter, group,stride=1,pad=0,name=''):
    #in_channels = int(input.get_shape()[-1])
    #weight = init_weight([kernel_size, kernel_size, in_channels, num_filter], 0.0001)
    #bias   = init_bias(num_filter)

    input = tf.pad(input,[[0,0],[pad,pad],[pad,pad],[0,0]], "CONSTANT")
    input_group = tf.split(axis=3, value=input, num_or_size_splits=group)
    #print 'input:'
    #print input_group[0]
    
    #weight_group  tf.split(axis=3, value=weight, num_split=group)
    weights, bias = get_params(name)

    weights = tf.split(axis=3, value=weights, num_or_size_splits=group)
    #print bias.shape
    #print '*'*10
    #bias    = tf.split(axis=1, value=bias, num_or_size_splits = group)
    #bias    = [i.reshape(-1) for i in bias]


    out_group = []

    for i in range(len(input_group)):
        #pass
        conv_i = tf.nn.conv2d(input_group[i], filter=weights[i], strides=(1,stride, stride,1), padding='VALID')
        out_group.append(conv_i)
        #out_group.append(conv2d(input_group[i], kernel_size=kernel_size, num_filter=num_filter/group, stride=stride, pad=pad,filter=[weights[i], bias[i]]))
        #conv_i = tf.nn.conv2d()
    #out_group = [conv2d(i,kernel_size=kernel_size, num_filter=num_filter/group, stride=stride, pad=pad) for i in input_group]
    conv = tf.concat(axis=3, values=out_group)
    
    conv = tf.nn.bias_add(conv, bias)
    
    return tf.nn.relu(conv)


def pooling(input, kernel_size, stride, pad=0, name=""):
    input = tf.pad(input,[[0,0],[pad, pad],[pad, pad],[0,0]], "CONSTANT")
    return tf.nn.max_pool(input, ksize=(1,kernel_size,kernel_size,1), strides=(1,stride, stride,1), padding="VALID", name='pool'+name)


X = tf.placeholder(dtype=tf.float32, shape=[1,768, 1024, 3], name="input")
Y = tf.placeholder(dtype=tf.float32, shape=[1,None, None, 1], name='label')

conv1_1 = conv2d(X,kernel_size=3, num_filter=64, pad=1, name="conv1_1")
conv1_2 = conv2d(conv1_1, kernel_size=3,num_filter=64, pad=1,name='conv1_2')

pool1   = pooling(conv1_1,kernel_size=2,stride=2,name='1')

conv2_1 = conv2d(pool1,kernel_size=3, num_filter=128, pad=1,name='conv2_1')
conv2_2 = conv2d(conv2_1, kernel_size=3, num_filter=128, pad=1, name='conv2_2')
pool2   = pooling(conv2_2, kernel_size=2, stride=2, name='2')

conv3_1 = conv2d(pool2, kernel_size=3, num_filter=256, pad=1, name='conv3_1')
conv3_2 = conv2d(conv3_1, kernel_size=3, num_filter=256,pad=1,name='conv3_2')
conv3_3 = conv2d(conv3_2, kernel_size=3, num_filter=256, pad=1, name='conv3_3')
pool3   = pooling(conv3_3, kernel_size=2, stride=2,name='3')

conv4_1 = conv2d(pool3, kernel_size=3,num_filter=512,pad=1, name='conv4_1')
conv4_2 = conv2d(conv4_1, kernel_size=3, num_filter=512, pad=1,name='conv4_2')
conv4_3 = conv2d(conv4_2, kernel_size=3, num_filter=512,pad=1,name='conv4_3')
pool4   = pooling(conv4_3, kernel_size=3, stride=1, pad=1, name='4')

conv5_1 = conv2d(pool4, kernel_size=3, num_filter=512,pad=2, dilation=2, name='conv5_1')
conv5_2 = conv2d(conv5_1, kernel_size=3, num_filter=512, pad=2, dilation=2, name='conv5_2')
conv5_3 = conv2d(conv5_2, kernel_size=3, num_filter=512, pad=2, dilation=2, name='conv5_3')

#print conv5_3

conv_score = group_conv2d(conv5_3, kernel_size=1,num_filter=64, group=64, name='conv_score')
print conv_score.get_shape()
scores = tf.split(axis=3, value=conv_score, num_or_size_splits=64)
sum_scores = sum(scores)
avg_scores = sum_scores / 64.0

predicts = [0.0] *64

for i in range(64):
    predicts[i] = tf.square(scores[i] - avg_scores)

predict1 = tf.reduce_sum(predicts) / 64.0
predict2 = tf.reduce_sum(avg_scores)

print avg_scores

path_to_test_img = '/home/thangnx/code/Deep-NCL/examples/crowd/shanghaiA/part_A/test_data/images'
#path_to_test_img = '/home/thangnx/code/human-counting/train/img'
#path_to_test_dmap = '/home/thangnx/code/human-counting/ShanghaiTech/part_A/test_data/dmap'
#path_to_test_dmap = '/home/thangnx/code/human-counting/train/dmap4'
#path_save = '/home/thangnx/code/human-counting/save/save4'
gtPath = '/home/thangnx/code/Deep-NCL/examples/crowd/shanghaiA/part_A/test_data/ground-truth/GT_IMG_'
li_img = [f for f in listdir(path_to_test_img) if isfile(join(path_to_test_img,f))]
print len(li_img)
#li_dmap = [f for f in listdir(path_to_test_dmap) if isfile(join(path_to_test_dmap,f))]
#print len(li_dmap)

'''
with tf.Session() as sess:
    #imNum = 0
    for i in range(len(li_img)):
        imNum = 0
        #img = cv2.imread(path_to_test_img+'/IMG_'+str(i+1) +'.jpg')
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
        img = np.array(Image.open(path_to_test_img+'/IMG_'+str(i+1) +'.jpg').convert('L'))
        img = img[:,:,np.newaxis]
        img = np.tile(img,(1,1,3))

        #mat = io.loadmat(path_to_test_dmap+'/DMAP_IMG_'+str(i+1)+'.mat')
        #mat = np.array(mat['DMAP'])
        #dmap_sum = sum(sum(mat))
        num = get_num(gtPath+ str(i+1)+'.mat')

        X_batch = np.zeros((1, img.shape[0], img.shape[1],3), dtype=np.float32)
        X_batch[0] = img

        #avg = sess.run(predict2, feed_dict={X:X_batch})

        #im_sum1 = 0
        #im_sum2 = 0

        #for id in range(len(scoresArr)):
        #    subim_sum = scoresArr[id]
        #    temp1 = (subim_sum - dmap_sum)**2
        #    temp2 = (subim_sum - avg)**2

        #    im_sum1 = im_sum1+temp1.sum()
        #    im_sum2 = im_sum2+temp2.sum()
        
        #imNum = imNum + im_sum2/64
        #print 'test {}: {}/{}'.format(i+1, avg,num )
        #print imNum
'''

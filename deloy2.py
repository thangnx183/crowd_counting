import tensorflow as tf
import numpy as np
from os import listdir
from os.path import join , isfile
import scipy.io as io
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import math

mat = io.loadmat('./vgg16.mat')
mat  = mat['layers'][0]
decay = 0.0005

def get_num(gt_path):
  gt = io.loadmat(gt_path)
  return gt['image_info'][0][0][0][0][0].shape[0]

def init_weight(shape, std):
    #print shape
    return tf.Variable(tf.truncated_normal(shape, stddev=std), dtype=tf.float32)

def init_bias(length):
    return tf.Variable(tf.zeros(length), dtype=tf.float32)

# define convolution layer
def conv2d(input, kernel_size, num_filter,std=0.01, stride=1, pad=0,dilation=1, name=""):
    
    if name and int(name[0]) < 5:
        for i in range(23):
            temp = 'conv'+name
            #print temp
            if mat[i][0][0][0][0] == temp:
                weight ,bias = mat[i][0][0][2][0]
                
                weight = np.transpose(weight, (1,0,2,3))
                weight = tf.Variable(weight)
                
                bias = bias.reshape(-1)    
                bias = tf.Variable(bias)
        
    else:   
        in_channels = int(input.get_shape()[-1])

        weight = init_weight([kernel_size, kernel_size,in_channels, num_filter], std)
        bias   = init_bias(num_filter)
        
        #weight_decay = decay * tf.nn.l2_loss(weight)
        #tf.add_to_collection("wd", weight_decay) 
        #print "trainable"
    
    weight_decay = decay * tf.nn.l2_loss(weight)
    tf.add_to_collection("wd", weight_decay)    

    input = tf.pad(input,[[0,0],[pad,pad],[pad,pad],[0,0]], "CONSTANT")
    conv = tf.nn.conv2d(input, filter=weight, strides=(1,stride, stride,1), padding='VALID',dilations=[1,dilation, dilation,1], name='conv'+name)
    
    conv = tf.nn.bias_add(conv, bias)
    #pass
    return tf.nn.relu(conv, name='relu'+name) 

#define max pooling layer 
def pooling(input, kernel_size, stride, pad=0, name=""):
    input = tf.pad(input,[[0,0],[pad, pad],[pad, pad],[0,0]], "CONSTANT")
    return tf.nn.max_pool(input, ksize=(1,kernel_size,kernel_size,1), strides=(1,stride, stride,1), padding="VALID", name='pool'+name)

#define group convolution layer
def group_conv2d(input, kernel_size, num_filter, group, std=0.001,stride=1,pad=0,name=''):
    #in_channels = int(input.get_shape()[-1])
    #weight = init_weight([kernel_size, kernel_size, in_channels, num_filter], 0.001)
    #bias   = init_bias(num_filter)
    
    input_group = tf.split(axis=3, value=input, num_or_size_splits=group)
    #weight_group  tf.split(axis=3, value=weight, num_split=group)

    out_group = [conv2d(i,kernel_size=kernel_size, num_filter=num_filter/group, std=std, stride=stride, pad=pad) for i in input_group]
    #out_group = []
    
    #for i in range(len(inut_group)):
    #  in_channels = int(input_group[i].get_shape()[-1])
    #  weight = init_weight([kernel_size, kernel_size,in_channels,num_filter/group], std = 0.0001)
    #  conv_i = tf.nn.conv2d(input)
    
    
    conv = tf.concat(axis=3, values=out_group)
    
    #conv = tf.nn.bias_add(conv, bias)

    return conv


  
  
X = tf.placeholder(dtype=tf.float32, shape=[1,None, None, 3], name="input")

Y = tf.placeholder(dtype=tf.float32, shape=[1,None, None, 1], name='label')

#c_train = tf.placeholer()

conv1_1 = conv2d(X,kernel_size=3, num_filter=64, pad=1, name="1_1")
conv1_2 = conv2d(conv1_1, kernel_size=3,num_filter=64, pad=1,name='1_2')
pool1   = pooling(conv1_1,kernel_size=2,stride=2,name='1')

conv2_1 = conv2d(pool1,kernel_size=3, num_filter=128, pad=1,name='2_1')
conv2_2 = conv2d(conv2_1, kernel_size=3, num_filter=128, pad=1, name='2_2')
pool2   = pooling(conv2_2, kernel_size=2, stride=2, name='2')

conv3_1 = conv2d(pool2, kernel_size=3, num_filter=256, pad=1, name='3_1')
conv3_2 = conv2d(conv3_1, kernel_size=3, num_filter=256,pad=1,name='3_2')
conv3_3 = conv2d(conv3_2, kernel_size=3, num_filter=256, pad=1, name='3_3')
pool3   = pooling(conv3_3, kernel_size=2, stride=2,name='3')

conv4_1 = conv2d(pool3, kernel_size=3,num_filter=512,pad=1, name='4_1')
conv4_2 = conv2d(conv4_1, kernel_size=3, num_filter=512, pad=1,name='4_2')
conv4_3 = conv2d(conv4_2, kernel_size=3, num_filter=512,pad=1,name='4_3')
pool4   = pooling(conv4_3, kernel_size=3, stride=1, pad=1, name='4')

conv5_1 = conv2d(pool4, kernel_size=3, num_filter=512,pad=2, dilation=2, name='5_1')
conv5_2 = conv2d(conv5_1, kernel_size=3, num_filter=512, pad=2, dilation=2, name='5_2')
conv5_3 = conv2d(conv5_2, kernel_size=3, num_filter=512, pad=2, dilation=2, name='5_3')

conv_score = group_conv2d(conv5_3, kernel_size=1,num_filter=64, group=64)

scores = tf.split(axis=3, value=conv_score, num_or_size_splits=64)
#print len(scores)
#print scores
sum_scores = sum(scores)
#print sum_scores
avg_scores = sum_scores / 64.0
#print avg_scores
#predicts = [0.0]*64

#for i in range(64):
#  predicts[i] = tf.square(scores[i] - avg_scores)

#predict1 = tf.reduce_sum(predicts) / 64.0
predict = tf.reduce_sum(avg_scores)
#print avg_scores.get_shape()


save_path = './save'



test_img_path = './part_A/test_data/images/IMG_'
test_gt_path  = './part_A/test_data/ground-truth/GT_IMG_'
#li_img_test = [f for f in listdir(test_img_path[:-5]) if isfile(join(test_img_path[:-5], f))]
#print li_img_test


config = tf.ConfigProto(allow_soft_placement = True)
saver = tf.train.Saver()

#test_img = cv2.imread('/content/drive/My Drive/human_counting/IMG_17.jpg')
#test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
#num = 285
#save_loss = []

#saver = tf.train.Saver()
import skvideo.io
import imageio
'''
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    #cap = skvideo.io.vread('videoplayback.mp4')
    source = cv2.imread('./ss/ss3.png',1)

    img = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    print sum(sum(img))
    img = img[:,:,np.newaxis]
    img = np.tile(img,(1,1,3))
    img = cv2.resize(img, (436,298), interpolation = cv2.INTER_AREA)
    print img.shape

    X_batch = np.zeros((1, 298, 436,3), dtype=np.float32)
    X_batch[0] = img

    num = sess.run(predict, feed_dict={X:X_batch})
    num = num *64

    frame = Image.fromarray(source)
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 50, encoding="unic")
    (x, y) = (50, 50)
    color = 'rgb(255, 0, 0)'

    draw.text((x, y), str(int(num)), fill=color, font=font)
    
    cv2.imshow('frame',np.array(frame))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
def get_order_pts(l_pts):
    l_pts = np.array(l_pts, dtype='float').tolist()
    cent=(sum([p[0] for p in l_pts])/len(l_pts),sum([p[1] for p in l_pts])/len(l_pts))
    l_pts.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))

    print '*'*10
    print l_pts

    return np.array(l_pts)


pts = []
dst = np.array([
    [0,0],
    [435,0],
    [435,297],
    [0,297]] , dtype='float32')

matrix = np.array([])

def get_pts(event, x,y, flags, params):
    global pts 
    global matrix

    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x,y])
        print pts  
        if len(pts) == 4:
            pts = get_order_pts(pts)
            matrix = cv2.getPerspectiveTransform(np.array(pts,dtype='float32'), dst)
            #print matrix
            



with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_path+'/net.ckpt')

    cap = imageio.get_reader('/home/thangnx/Videos/cam_full.mp4')
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('frame',get_pts)
    cv2.resizeWindow('frame', 600, 600)

    writer  = imageio.get_writer('out.mp4')

    num = 0

    for i , frame in enumerate(cap):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #cv2.putText(img=frame,  text='hello !!', org=(10,40),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,color=2, thickness=2)
        #cv2.imshow('frame', frame)
        
        if i == 0 :
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
        
        if i % 8 == 0:
            #cv2.imwrite('./image/img_'+str(i)+'.jpg', frame)
            i_frame = cv2.warpPerspective(frame, matrix,(436,298))

            cv2.imshow("perspective", i_frame)
            #print i_frame.shape
            cv2.imwrite('./perspective/per_'+str(i)+'.jpg', i_frame)

            i_frame = cv2.cvtColor(i_frame, cv2.COLOR_BGR2GRAY)    
            i_frame = i_frame[:,:,np.newaxis]
            i_frame = np.tile(i_frame,(1,1,3))
            
            X_batch = np.zeros((1, 298, 436,3), dtype=np.float32)
            X_batch[0] = cv2.resize(i_frame, (436,298), interpolation = cv2.INTER_AREA)

            

            #num = sess.run(predict, feed_dict={X:X_batch})

        #cv2.putText(img=frame,  text=str(int(num*64)), org=(10,40),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,color=2, thickness=2)
        cv2.imshow('frame', frame)
        #writer.append_data(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #if cv2.waitKey(1) & 0xFF == ord('c'):
        #    cv2.imwrite('./image/img_'+str(i)+'.jpg', frame)
    cv2.destroyAllWindows()
        # num = sess.run(predict, feed_dict={X:X_batch})  



    

















'''
    cap = imageio.get_reader('kim3.mp4')
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 600,600)
    #print cap
    #print 'reading video'
    num = 0
    writer = imageio.get_writer("out2.mp4")
    for i, frame in enumerate(cap):
        #print frame.shape
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      
        
        if i % 50 == 0:
            img = frame[:,:,np.newaxis]
            img = np.tile(img,(1,1,3))
            img = cv2.resize(img, (436,298), interpolation = cv2.INTER_AREA)
            X_batch = np.zeros((1, 298, 436,3), dtype=np.float32)
            X_batch[0] = img
            num = sess.run(predict, feed_dict={X:X_batch})
            num = num *64

        frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 50, encoding="unic")
        (x, y) = (50, 50)
        color = 'rgb(0, 0, 0)'

        draw.text((x, y), str(num), fill=color, font=font)
        
        cv2.imshow('frame',np.array(frame))
        writer.append_data(np.array(frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    writer.close()
    cv2.destroyAllWindows()
'''






'''
with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())
    #train_arr = [ i+1 for i in range(len(li_img))]

    #Xtest_batch = np.zeros((1,test_img.shape[0], test_img.shape[1], 3), dtype = np.float32)
    #Xtest_batch[0] = get_train_img(test_img)

    saver.restore(sess, save_path+'/net.ckpt')
  

    li_img_test = [f for f in listdir(test_img_path[:-5]) if isfile(join(test_img_path[:-5], f))]
    mse = 0
    len_test = 0
    for i in range(len(li_img_test)):
        img = cv2.imread(test_img_path+str(i+1)+'.jpg',0)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #print sum(img)
        img = img[:,:,np.newaxis]
        img = np.tile(img,(1,1,3))
        #deloy_img = cv2.resize(img, (436,298), interpolation = cv2.INTER_AREA)
        #li_img = get_sub_deloy_img(img)
        #dmap = io.loadmat(test_path+'/dmap/DMAP_IMG_'+str(i+1)+ '.mat')
        #dmap = np.array(dmap['DMAP'])
        num_gt = get_num(test_gt_path+str(i+1)+ '.mat') 
        if num_gt > 400:
            continue

        X_batch = np.zeros((1, 298, 436,3), dtype=np.float32)
        #X_batch[0] = deloy_img
        #out_num = 0
        #for id in range(len(li_img)):
        #  X_batch[0] = li_img[id]
        #  out_avg = sess.run(predict, feed_dict={X:X_batch})
        #  out_num += out_avg
        #print sum(sum(X_batch))

        X_batch[0] = cv2.resize(img, (436,298), interpolation = cv2.INTER_AREA)
        uncut_count = sess.run(predict, feed_dict={X:X_batch})

        out_avg = sess.run(predict, feed_dict={X:X_batch})

        print 'test {}  shape {} {} : {} / {} '.format(i+1, img.shape[0], img.shape[1] ,uncut_count*64, num_gt )
        with open('out.txt', 'a') as f:
            f.write('image {} : {} / {} \n'.format(i+1,uncut_count*64,num_gt ))
        
        mse += abs(uncut_count*64 - num_gt)
        len_test += 1
    mse = mse / len_test
    print 'mse : {} / len testset {}'.format(mse, len_test)
    with open('out.txt','a') as f:
        f.write('MAE : {} \n'.format(mse))
'''


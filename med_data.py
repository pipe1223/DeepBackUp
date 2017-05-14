import numpy as np
import cv2
import os

MAIN_PATH = "D:/Work/data/2dslice/"
#MAIN_PATH = "/home/chen-lab/pipe_workspace/data/medical/multi_organ/Cystdata/Cystdata/"

PATH1 = MAIN_PATH + "train_224/"


NUM_CLASSES = 10
image_size = 224


def binarylab(labels):
    x = np.zeros([image_size,image_size,NUM_CLASSES])
    #print np.amin(labels)
    #print np.amax(labels)

    for i in range(image_size):
        for j in range(image_size):
            if (labels[i][j] >= NUM_CLASSES):
                labels[i][j] = 1
            x[i,j,labels[i][j]]=labels[i][j]+1
            
    return x

import math

def distance_mark(im_size=(image_size,image_size)):
    
    #seed point coordinates
    fmark = np.zeros((2,2))
    fmark[0] = [ image_size*0.5,  image_size*0.25]
    fmark[1] = [ image_size*0.5,  image_size*0.75]
    delta = image_size/(2*fmark.shape[0])

    def sigmoid(x):
        return 1/(1+math.exp(-x))

    fmap = np.zeros((2,image_size,image_size))
    for x in range (image_size):
        for y in range (image_size):
            for mark_index in range (fmark.shape[0]):
                fmap[mark_index,x,y] = sigmoid(np.amin(np.sqrt(pow(x-fmark[mark_index,0],2)+pow(y-fmark[mark_index,1],2)))/delta)
    return fmap

def distance_map(name):

    path_to_f_seed = MAIN_PATH + "test/f_seed_"+str(image_size)+"/"
    path_to_b_seed = MAIN_PATH + "test/b_seed_"+str(image_size)+"/"

    img_f = cv2.imread(path_to_f_seed+name)
    img_b = cv2.imread(path_to_b_seed+name)
    #gray_scale

    img_f = img_f.astype('float32')
    img_b = img_b.astype('float32')


    img_f[img_f[:, :, 0] != 0] = 1
    img_b[img_b[:, :, 0] != 0] = 1    



    # seed point coordinates
    fg_point = np.argwhere(img_f[:, :, 0] != 0)
    bg_point = np.argwhere(img_b[:, :, 0] != 0)
    delta = image_size/8.

    def sigmoid(x):
        return x#1 / (1 + math.exp(-x))

    fmap = np.zeros((image_size, image_size))
    bmap = np.zeros((image_size, image_size))
    for x in range(fmap.shape[0]):
        for y in range(fmap.shape[1]):

            if fg_point.shape[0]:
                fmap[x, y] = sigmoid(np.amin(np.sqrt(pow(x - fg_point[:, 0], 2) + pow(y - fg_point[:, 1], 2))) / delta)
            else:
                fmap[x,y] = 0.5


            if bg_point.shape[0]:
                bmap[x, y] = sigmoid(np.amin(np.sqrt(pow(x - bg_point[:, 0], 2) + pow(y - bg_point[:, 1], 2))) / delta)
            else:
                bmap[x, y] = 0.5

    return fmap, bmap


def load_data(num=0):
    import glob
    #print "load data"
    train_data = []
    train_label = []
    label_train = [os.path.basename(x) for x in glob.glob(PATH1+ "*")]
    
    if (num==0):
        num = len(label_train)
    for i in range(num):
        img, target = process_line_load(label_train[i])
        #if np.sum(target) < 5000:
            #print "remove " + label_train[i]
        #else:
        #print label_train[i]+":"+str(np.sum(target))
        train_data.append(img)
        train_label.append(target)
        #print "done:"+str(i)
    return np.array(train_data), np.array(train_label)

def load_data_auto_seg(num=0):
    import glob
    #print "load data"
    train_data = []
    train_label = []
    label_train = [os.path.basename(x) for x in glob.glob(PATH1+ "*")]

    if (num==0):
        num = len(label_train)
    for i in range(num):
        img, target = process_line_load_auto_seg(label_train[i])
        #if np.sum(target) < 5000:
            #print "remove " + label_train[i]
        #else:
        #print label_train[i]+":"+str(np.sum(target))
        train_data.append(img)
        train_label.append(target)
        #print "done:"+str(i)
    return np.array(train_data), np.array(train_label)


def generate_arrays_from_file(path):
    while 1:
        with open(path,"r") as f:
            names = f.readlines()
        ls = [line.rstrip('\n') for line in names]
        
        for name in ls:
            img, target = process_line(name)
            yield (img, target)

def generate_val_from_file(path):
    while 1:
        with open(path,"r") as f:
            names = f.readlines()
        ls = [line.rstrip('\n') for line in names]

        for name in ls:
            img, target = process_line(name)
            yield (img, target)



def process_line_load(name):
    fmap, bmap = distance_map(name)
    path_to_train = MAIN_PATH + "train_"+str(image_size)+"/"
    path_to_target = MAIN_PATH + "npy_target_"+str(image_size)+"/"
    img = cv2.imread(path_to_train+ name)

    #gray_scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = img.astype('float32')
    img /= 255
    #img = img.reshape((3,image_size,image_size))

    #img = img[np.newaxis, ...]

    mark_total = np.zeros(( 3, image_size, image_size))
    mark_total[0] = img
    mark_total[1] = fmap
    mark_total[2] = bmap


    #print mark_total.shape
    target = np.load(path_to_target+ name +".npy")
    target = np.asarray(binarylab(target))
    target = target.reshape((image_size*image_size,NUM_CLASSES))
    return mark_total, target

def process_line_load_auto_seg(name):
    path_to_train = MAIN_PATH + "train_"+str(image_size)+"/"
    path_to_target = MAIN_PATH + "npy_target_"+str(image_size)+"/"
    img = cv2.imread(path_to_train+ name)

    #gray_scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype('float32')
    img /= 255
    #img = img.reshape((3,image_size,image_size))

    #img = img[np.newaxis, ...]

    mark_total = np.zeros(( 1, image_size, image_size))
    mark_total[0] = img
    #mark_total[1] = fmap
    #mark_total[2] = bmap


    #print mark_total.shape
    target = np.load(path_to_target+ name +".npy")
    target = np.asarray(binarylab(target))
    target = target.reshape((image_size*image_size,NUM_CLASSES))
    return mark_total, target



def process_line(name):
    path_to_target = MAIN_PATH + "npy_target_"+str(image_size)+"/"

    path_to_train = MAIN_PATH + "train_"+str(image_size)+"/"
    
    img = cv2.imread(path_to_train+ name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32')
    img /= 255
    mark_total = np.zeros(( 1,1, image_size, image_size))
    #img = img.reshape((1,1,image_size,image_size))
    
    mark_total[0] = img
    
    target = np.load(path_to_target+ name +".npy")
    target = np.asarray(binarylab(target))
    target = target.reshape((1,image_size*image_size,NUM_CLASSES))
    return mark_total, target

import numpy as np
import cv2
from multiprocessing import Pool
import math
import os
import glob 
import random



class Preprocess:
    'Common base class for all employees'
    dataDir = "data/"
    labelDir = "label/"
    image_size = 224
    num_classes = 21
    
    def __init__(self, mainPath, dataFolder=dataDir, labelFolder = labelDir, img_size = image_size, number_classes = 21):
        
        Preprocess.dataDir = dataFolder
        Preprocess.labelDir = labelFolder
        Preprocess.image_size = img_size
        Preprocess.num_classes = number_classes
        
        self.mainPath = mainPath
        self.dataPath = mainPath + Preprocess.dataDir
        self.labelPath = mainPath + Preprocess.labelDir
        self.trainPath = mainPath + 'train_'+str(img_size)
        self.targetPath = mainPath + 'target_'+str(img_size)
        self.npy_targetPath = mainPath + "npy_target_"+str(img_size)
   
    def displayPathDetail(self):
        print "Main Path:",  self.mainPath
        print "Data Folder:", Preprocess.dataDir
        print "Label Folder:", Preprocess.labelDir

    def resize_img(self,debug = False):
        ls = [os.path.basename(x) for x in glob.glob(self.dataPath + "*")]
        for imgname in ls:
            path_to_train = self.dataPath + imgname
            path_to_teacher = self.labelPath + imgname

            train_img = cv2.imread(path_to_train)
            target_img = cv2.imread(path_to_teacher)

            #resize
            shape = train_img.shape[:-1]
            shorter = shape[0] if shape[0] < shape[1] else shape[1]
            length = int(shorter/2)
            xc, yc = int(shape[0]/2), int(shape[1]/2)

            train_img = train_img[xc-length:xc+length, yc-length:yc+length]
            target_img = target_img[xc-length:xc+length, yc-length:yc+length]

            train_img = cv2.resize(train_img,(Preprocess.image_size,Preprocess.image_size))
            target_img = cv2.resize(target_img,(Preprocess.image_size,Preprocess.image_size))

            cv2.imwrite(self.trainPath + '/{}'.format(imgname), train_img)
            cv2.imwrite(self.targetPath + '/{}'.format(imgname), target_img)
            
            Preprocess.make_target_dataset(self, imgname)
            if debug: 
                print("Done {}".format(imgname))

    def resize_img_with_effect(self,debug = False):
        ls = [os.path.basename(x) for x in glob.glob(self.dataPath + "*")]
        for imgname in ls:
            path_to_train = self.dataPath + imgname
            path_to_teacher = self.labelPath + imgname

            train_img = cv2.imread(path_to_train)
            target_img = cv2.imread(path_to_teacher)

            #resize
            shape = train_img.shape[:-1]
            shorter = shape[0] if shape[0] < shape[1] else shape[1]
            length = int(shorter/2)
            xc, yc = int(shape[0]/2), int(shape[1]/2)

            train_img = train_img[xc-length:xc+length, yc-length:yc+length]
            target_img = target_img[xc-length:xc+length, yc-length:yc+length]

            train_img = cv2.resize(train_img,(Preprocess.image_size,Preprocess.image_size))
            target_img = cv2.resize(target_img,(Preprocess.image_size,Preprocess.image_size))
            
            
            #save and write on disk##################
            preprocess_type = {'s&p', 'speckle'}
            for i in preprocess_type:
                noise_image = Preprocess.noisy(self, i,train_img)
                cv2.imwrite(self.trainPath + '/{}'.format(i+'_'+imgname), noise_image)
                
                #save label
                cv2.imwrite(self.targetPath + '/{}'.format(i+'_'+imgname), target_img)
                Preprocess.make_target_dataset(self, i+'_'+imgname)
                #########################################
            if debug: 
                print("Done {}".format(imgname))

            
    def make_target_dataset(self, imgname):
        target = np.zeros((Preprocess.image_size, Preprocess.image_size),dtype="int32")

        path = self.targetPath + "/"
        #for imgname in imgs:
        path_to_target = path + imgname
        target_img = cv2.imread(path_to_target)
        dim = target_img.shape
        for x in range(dim[0]):
            for y in range(dim[1]):
                target[x,y] = target_img[x][y][0]

        np.save(self.npy_targetPath + "/{}.npy".format(imgname),target)
        #print("Done {}".format(imgname))
    
    def generate_txt(self, ratio=0.8):
        f = open(self.mainPath+"train.txt","w")
        f_val = open(self.mainPath+"val.txt","w")
        ls = [os.path.basename(x) for x in glob.glob(self.trainPath + "/*")]
        random.shuffle(ls)
        for i in range(len(ls)):
            imgname = ls[i]
            if i < len(ls)*ratio:
                f.write(imgname+"\n")
            else:
                f_val.write(imgname+"\n")
        f_val.close()
        f.close()
    
    def binarylab(self, labels):
        x = np.zeros([Preprocess.image_size, Preprocess.image_size,Preprocess.num_classes])

        for i in range(Preprocess.image_size):
            for j in range(Preprocess.image_size):
                if (labels[i][j] >= Preprocess.num_classes):
                    labels[i][j] = 1
                x[i,j,labels[i][j]]=labels[i][j]+1
        return x

    def generate_arrays_from_file(self,is_train = True):
        if is_train:
            path = self.mainPath + "train.txt"
        else:
            path = self.mainPath + "val.txt"
        while 1:
            with open(path,"r") as f:
                names = f.readlines()
            ls = [line.rstrip('\n') for line in names]

            for name in ls:
                img, target = Preprocess.process_line(self,name)
                yield (img, target)    

    def process_line(self, name):
        
        path_to_target = self.npy_targetPath+"/"

        path_to_train = self.trainPath+"/"
        
        mark_total = Preprocess.load_file(self,path_to_train + name)
        
        ###remove if no problem 
        ###4/5/2017
        #img = cv2.imread(path_to_train + name)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = img.astype('float32')
        #img /= 255
        #mark_total = np.zeros(( 1,1, Preprocess.image_size, Preprocess.image_size))
        
        #mark_total[0] = img
        ###
        
        target = np.load(path_to_target+ name +".npy")
        target = np.asarray(Preprocess.binarylab(self, target))
        target = target.reshape((1,Preprocess.image_size*Preprocess.image_size,Preprocess.num_classes))
        return mark_total, target
    
    def load_file(self,filePath):
        img = cv2.imread(filePath)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img.astype('float32')
        img /= 255
        mark_total = np.zeros(( 1,1, Preprocess.image_size, Preprocess.image_size))
        
        mark_total[0] = img
        return mark_total
    
    def noisy(self, noise_typ,image):
        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 0.5
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.05
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ =="speckle":
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = image + image * gauss
            return noisy

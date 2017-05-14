from keras.layers import merge, Lambda, Convolution2D, Deconvolution2D, MaxPooling2D, Input, Reshape, Permute, ZeroPadding2D, UpSampling2D, Cropping2D
from keras.layers.core import Activation
from keras.models import Model
#from keras.utils.visualize_util import model_to_dot, plot
from keras.utils.layer_utils import layer_from_config
from keras.utils import np_utils, generic_utils
from keras import backend as K

K.set_image_dim_ordering('th')

import cv2
import numpy as np

def FCN(FCN_CLASSES = 21):

    image_size = 224
    #(samples, channels, rows, cols)
    input_img = Input(shape=(3, image_size, image_size))
    #(3*224*224)
    x = Convolution2D(64, 3, 3, activation='relu',border_mode='same')(input_img)
    x = Convolution2D(64, 3, 3, activation='relu',border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(64*112*112)
    x = Convolution2D(128, 3, 3, activation='relu',border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu',border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(128*56*56)
    x = Convolution2D(256, 3, 3, activation='relu',border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu',border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(256*56*56)

    #split layer
    p3 = x
    p3 = Convolution2D(FCN_CLASSES, 1, 1,activation='relu')(p3)
    #(21*28*28)

    x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(512*14*14)

    #split layer
    p4 = x
    p4 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p4)
    
    p4 = Deconvolution2D(FCN_CLASSES, 4, 4,
            output_shape=(None, FCN_CLASSES, 30, 30),
            subsample=(2, 2),
            border_mode='valid')(p4)
    p4 = Cropping2D(cropping=((1, 1), (1, 1)))(p4)

    #(21*28*28)

    x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(512*7*7)

    p5 = x
    p5 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p5)
    p5 = Deconvolution2D(FCN_CLASSES, 8, 8,
            output_shape=(None, FCN_CLASSES, 32, 32),
            subsample=(4, 4),
            border_mode='valid')(p5)
    p5 = Cropping2D(cropping=((2, 2), (2, 2)))(p5)
    #(21*28*28)

    # merge scores
    merged = merge([p3, p4, p5], mode='sum')
    x = Deconvolution2D(FCN_CLASSES, 16, 16,
            output_shape=(None, FCN_CLASSES, 232, 232),
            subsample=(8, 8),
            border_mode='valid')(merged)
    x = Cropping2D(cropping=((4, 4), (4, 4)))(x)
    x = Reshape((FCN_CLASSES,image_size*image_size))(x)
    x = Permute((2,1))(x)
    out = Activation("softmax")(x)
    #(21,224,224)
    model = Model(input_img, out)
    return model

def FCN_2(FCN_CLASSES = 21):
    
    image_size = 224

    input_seed = Input(shape=(2, image_size, image_size))
    seed_pool = MaxPooling2D((2,2), strides=(2,2))(input_seed)#112
    seed_pool = MaxPooling2D((2,2), strides=(2,2))(seed_pool) #56
    seed_pool28 = MaxPooling2D((2,2), strides=(2,2))(seed_pool) #28
    seed_pool14 = MaxPooling2D((2,2), strides=(2,2))(seed_pool28) #14
    seed_pool7 = MaxPooling2D((2,2), strides=(2,2))(seed_pool14) #56


    
    #(samples, channels, rows, cols)
    input_img = Input(shape=(1, image_size, image_size))
    #(3*224*224)a
    x = Convolution2D(64, 3, 3, activation='relu',border_mode='same')(input_img)
    x = Convolution2D(64, 3, 3, activation='relu',border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(64*112*112)
    x = Convolution2D(128, 3, 3, activation='relu',border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu',border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(128*56*56)
    x = Convolution2D(256, 3, 3, activation='relu',border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu',border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(256*28*28)
    
    #split layer
    p3 = x
    p3 = merge([p3, seed_pool28], mode='concat', concat_axis=1)

    p3 = Convolution2D(FCN_CLASSES, 1, 1,activation='relu')(p3)
    #(21*28*28)

    x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(512*14*14)
    

    #split layer
    p4 = x
    p4 = merge([p4, seed_pool14], mode='concat', concat_axis=1)

    p4 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p4)

    p4 = Deconvolution2D(FCN_CLASSES, 4, 4,
            output_shape=(None, FCN_CLASSES, 30, 30),
            subsample=(2, 2),
            border_mode='valid')(p4)
    p4 = Cropping2D(cropping=((1, 1), (1, 1)))(p4)
    
    
    #(21*28*28)






    x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(512*7*7)

    p5 = x
    p5 = merge([p5, seed_pool7], mode='concat', concat_axis=1)

    p5 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p5)
    p5 = Deconvolution2D(FCN_CLASSES, 8, 8,
            output_shape=(None, FCN_CLASSES, 32, 32),
            subsample=(4, 4),
            border_mode='valid')(p5)
    p5 = Cropping2D(cropping=((2, 2), (2, 2)))(p5)
    #(21*28*28)

    # merge scores
    merged = merge([p3, p4, p5], mode='sum')
    x = Deconvolution2D(FCN_CLASSES, 16, 16,
            output_shape=(None, FCN_CLASSES, 232, 232),
            subsample=(8, 8),
            border_mode='valid')(merged)
    x = Cropping2D(cropping=((4, 4), (4, 4)))(x)


    #model_test =Model([input_img, input_seed],merge_seed)
    #print model_test.output_shape   
    x = Convolution2D(FCN_CLASSES, 1, 1, activation='relu', border_mode='same')(x)
    
    x = Reshape((FCN_CLASSES,image_size*image_size))(x)

    x = Permute((2,1))(x)
    out = Activation("softmax")(x)
    model = Model([input_img, input_seed], out)
    return model

def FCN_AlexNet(FCN_CLASSES = 21):
    image_size = 224
    
    
    #INPUT
    #(samples, channels, rows, cols)
    #input = (3,224,224)
    input_img = Input(shape=(1, image_size, image_size))

    #CONVOLUTION 1
    #output = (96,55,55)
    x = ZeroPadding2D((2,2))(input_img)
    x = Convolution2D(96, 11, 11, activation='relu', subsample=(4,4),)(x)
    ################
    
    #POOLING 1
    #output = (96,27,27)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    ################
    
    #CONVOLUTION 2
    #output = (256,28,28)
    x = ZeroPadding2D((2,2))(x)
    x = Convolution2D(256, 4, 4, activation='relu', subsample=(1,1))(x)
    ################
    
    model = Model(input_img, x)
    print (model.output_shape)


    #DECONVOLUTION 1 (SPLIT)
    p3 = x
    p3 = Convolution2D(FCN_CLASSES, 1, 1,activation='relu')(p3)
    ################
    
    #POOLING 2
    #output = (256,13,13)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    ################
    model = Model(input_img, x)
    print (model.output_shape)


    #DECONVOLUTION 2 (SPLIT)
    p4 = x
    p4 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p4)

    model = Model(input_img, p4)
    print (model.output_shape)


    p4 = Deconvolution2D(FCN_CLASSES, 4, 4,
            output_shape=(None, FCN_CLASSES, 30, 30),
            subsample=(2, 2),
            border_mode='valid')(p4)
    p4 = Cropping2D(cropping=((1, 1), (1, 1)))(p4)
    ################

    model = Model(input_img, p4)
    print (model.output_shape)

    
    #CONVOLUTION 3
    #output = (256,7,7)
    x = Convolution2D(384, 4, 4, activation='relu')(x)
    x = Convolution2D(384, 3, 3, activation='relu')(x)
    x = Convolution2D(256, 3, 3, activation='relu')(x)
    ################
    
    #DECONVOLUTION 3
    p5 = x
    p5 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p5)
    p5 = Deconvolution2D(FCN_CLASSES, 8, 8,
            output_shape=(None, FCN_CLASSES, 32, 32),
            subsample=(4, 4),
            border_mode='valid')(p5)
    p5 = Cropping2D(cropping=((2, 2), (2, 2)))(p5)
    ################
    
    #DECONVOLUTION FINAL (MERGE)
    merged = merge([p3, p4, p5], mode='sum')
    x = Deconvolution2D(FCN_CLASSES, 16, 16,
            output_shape=(None, FCN_CLASSES, 232, 232),
            subsample=(8, 8),
            border_mode='valid')(merged)
    x = Cropping2D(cropping=((4, 4), (4, 4)))(x)
    ################
    
    #RESHAPE (FINAL OUTPUT)
    #output = (50176, FCN_CLASSES)
    x = Reshape((FCN_CLASSES,image_size*image_size))(x)
    x = Permute((2,1))(x)
    out = Activation("softmax")(x)
    model = Model(input_img, out)
    print (model.output_shape) 
    return model
    
def to_json(model):
    json_string = model.to_json()
    with open('FCN_via_Keras_architecture.json', 'w') as f:
        f.write(json_string)

if __name__ == "__main__":
    model = FCN()
    #visualize_model(model)
    #to_json(model)

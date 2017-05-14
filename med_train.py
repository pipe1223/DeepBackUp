import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'

from keras import backend as K
from keras.optimizers import Adam

from med_FCN import FCN, FCN_2, FCN_AlexNet
from med_data import generate_arrays_from_file, load_data, distance_mark, load_data_auto_seg
from med_DeconvNetModel import DeconvNet

import cv2
import numpy as np


#import os
#os.environ['KERAS_BACKEND'] = 'theano'
#for using gpu
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'

def train(path,val,model_name="FCN"):
    model_dec = { "FCN" : FCN, "DeconvNet" : DeconvNet}
    model = model_dec[model_name](2)
    model.compile(loss="categorical_crossentropy",
                  optimizer='adadelta',
                  metrics=["accuracy"])
    #model.load_weights("weights_of_{}.hdf5".format(model_name),by_name=True)
    print model.output_shape
    model.fit_generator(generate_arrays_from_file(path),
        samples_per_epoch=2000, nb_epoch=5,validation_data=generate_arrays_from_file(val),nb_val_samples=300)
    model.save_weights('weights_of_{}_tm_med.hdf5'.format(model_name))
    return

def train_load(path,model_name="FCN"):
    print "train_load"

    nb_epoch = 1000
    batch_size = 20
    learning_rate = 1e-5
    decay_rate = 5e-5
    momentum_rate = 0.9
     
  
    from keras.optimizers import SGD
    sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum_rate, nesterov=True)

    
    smooth =1.


    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        #y_pred_f = K.update(y_pred_f[(y_pred_f >=0.5).nonzero()],1.)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

    model_dec = { "FCN" : FCN_2, "DeconvNet" : DeconvNet, "FCN_alex" : FCN_AlexNet}
    model = model_dec[model_name](10)
    model.compile(loss=dice_coef_loss,#"categorical_crossentropy",
                  optimizer=Adam(lr=1e-6),#'adadelta',
                  metrics=[dice_coef])


    #model.load_weights("weights_of_FCN_loadtype_ADAM_mask_po6.hdf5",by_name=True)

    trains_list,trainl_list = load_data_auto_seg()
    
    
       
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(filepath="weights_of_FCN_alex_ADAM_mask_po6.hdf5", verbose=1, save_best_only=True)
    print trains_list.shape
    history = model.fit(trains_list, trainl_list, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.3, callbacks=[checkpointer])
    return

if __name__ == "__main__":
    #train(train.txt,model_name="DeconvNet")
    train_load('./med_train.txt',model_name="FCN_alex")#"DeconvNet")

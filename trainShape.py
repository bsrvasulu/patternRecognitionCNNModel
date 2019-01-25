# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 12:53:03 2019

@author: Sreenivasulu Bachu
"""

import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, core, ZeroPadding2D, BatchNormalization, Activation, Flatten, Dense
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import keras.layers
from keras.layers.merge import concatenate
from keras.utils import *
from keras.initializers import glorot_uniform
from scipy.ndimage.filters import gaussian_filter, gaussian_laplace
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_local
from scipy.ndimage.morphology import binary_erosion
from sklearn.metrics import confusion_matrix, classification_report
import sys, os, fnmatch
import pandas as pd
import math
import statistics
import pickle
#from imageprocess import *


class LossObj(object):
    def __init__(self):
        self.losses = []
        self.dice_coef = []        
    
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.lossObj = LossObj() 
        self.lossObj.losses = []
        #self.lossObj.dice_coef = []
        self.lossObj.accuracy = []
        
    def on_epoch_end(self, batch, logs={}):
        self.lossObj.losses.append(logs.get('loss'))
        #self.lossObj.dice_coef.append(logs.get('dice_coef_mod'))  
        self.lossObj.accuracy.append(logs.get('acc'))  
        
    def on_batch_end(self, batch, logs={}):
        self.lossObj.losses.append(logs.get('loss'))
        #self.lossObj.dice_coef.append(logs.get('dice_coef_mod'))  
        self.lossObj.accuracy.append(logs.get('acc'))  
        
    def get_LossObj(self):
        return self.lossObj
    
    #def get_dice_coeff(self):
    #    return self.dice_coef

def dice_coef_mod(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred) 
    y_pred_f = K.clip(y_pred_f, 0., 1.)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss_mod(y_true, y_pred):
    return 1.0 -dice_coef_mod(y_true, y_pred)

class trainShapes(object):
    def __init__(self, shape, img_rows = 256, img_cols = 256, dataDir = './images'):
        self.dataDir = dataDir
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.shape = shape 
        self.num_channels = 1
    
    def prepare_network_chanal_last(self):
        inputs = Input((self.img_rows, self.img_cols, self.num_channels))

        # zero padding
        #zeroPadX = ZeroPadding2D(padding = (4, 4))(inputs)
        
        conv0 = Conv2D(8, (2, 2), strides=(1, 1), padding='same', data_format='channels_last')(inputs)
        conv0 = BatchNormalization(axis = 3)(conv0)
        conv0 = Activation('relu')(conv0)
        pool0 = MaxPooling2D((2, 2), strides=(1, 1), padding='same', data_format='channels_last')(conv0)        
        
        conv01 = Conv2D(8, (2, 2), strides=(1, 1), padding='same', data_format='channels_last')(pool0)
        conv01 = BatchNormalization(axis = 3)(conv01)
        conv01 = Activation('relu')(conv01)
        pool01 = MaxPooling2D((2, 2), strides=(1, 1), padding='same', data_format='channels_last')(conv01)
        
        conv1 = Conv2D(16, (4, 4), padding='same', data_format='channels_last')(pool01)
        conv1 = BatchNormalization(axis = 3)(conv1)
        conv1 = Activation('relu')(conv1)        
        pool1 = MaxPooling2D((4, 4), strides=(1, 1), padding='same', data_format='channels_last')(conv1)
        
        conv11 = Conv2D(16, (4, 4), padding='same', data_format='channels_last')(pool1)
        conv11 = BatchNormalization(axis = 3)(conv11)
        conv11 = Activation('relu')(conv11)        
        pool11 = MaxPooling2D((4, 4), strides=(1, 1), padding='same', data_format='channels_last')(conv11)
        
        conv2 = Conv2D(32, (8, 8), padding='same', data_format='channels_last')(pool11)
        conv2 = BatchNormalization(axis = 3)(conv2)
        conv2 = Activation('relu')(conv2)        
        pool2 = MaxPooling2D((8, 8), strides=(1, 1), padding='same', data_format='channels_last')(conv2)

        conv3 = Conv2D(1, (1, 1), padding='same',data_format='channels_last')(pool2)
        conv3 = Activation('relu')(conv3)

        model = Model(inputs = inputs, outputs = conv3)
        #model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss_mod,metrics=[dice_coef_mod])
        model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
        return model
 
    def prepare_network_chanal_last_classification_2(self, classes = 4):
        inputs = Input((self.img_rows, self.img_cols, self.num_channels))

        # zero padding
        #zeroPadX = ZeroPadding2D(padding = (4, 4))(inputs)
        
        conv0 = Conv2D(8, (2, 2), strides=(1, 1), padding='same', data_format='channels_last')(inputs)
        conv0 = BatchNormalization(axis = 3)(conv0)
        conv0 = Activation('relu')(conv0)
        pool0 = MaxPooling2D((2, 2), strides=(1, 1), padding='same', data_format='channels_last')(conv0)        
        
        conv01 = Conv2D(8, (2, 2), strides=(1, 1), padding='same', data_format='channels_last')(pool0)
        conv01 = BatchNormalization(axis = 3)(conv01)
        conv01 = Activation('relu')(conv01)
        pool01 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', data_format='channels_last')(conv01)
        
        conv1 = Conv2D(16, (4, 4), padding='same', data_format='channels_last')(pool01)
        conv1 = BatchNormalization(axis = 3)(conv1)
        conv1 = Activation('relu')(conv1)        
        pool1 = MaxPooling2D((4, 4), strides=(2, 2), padding='same', data_format='channels_last')(conv1)
        
        conv11 = Conv2D(16, (4, 4), padding='same', data_format='channels_last')(pool1)
        conv11 = BatchNormalization(axis = 3)(conv11)
        conv11 = Activation('relu')(conv11)        
        pool11 = MaxPooling2D((4, 4), strides=(2, 2), padding='same', data_format='channels_last')(conv11)
        
        conv2 = Conv2D(32, (8, 8), padding='same', data_format='channels_last')(pool11)
        conv2 = BatchNormalization(axis = 3)(conv2)
        conv2 = Activation('relu')(conv2)        
        pool2 = MaxPooling2D((8, 8), strides=(2, 2), padding='same', data_format='channels_last')(conv2)
        
        conv3 = Conv2D(32, (2, 2), padding='same', data_format='channels_last')(pool2)
        conv3 = BatchNormalization(axis = 3)(conv3)
        conv3 = Activation('relu')(conv3)        
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', data_format='channels_last')(conv3)
        

        conv4 = Conv2D(1, (1, 1), padding='same',data_format='channels_last')(pool3)
        conv4 = Activation('relu')(conv4)

        # output layer
        X = Flatten()(conv4)
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
        #X = Dense(classes, activation=keras.activations.softmax(X, dim=axis), name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
        

        # Create model
        model = Model(inputs = inputs, outputs = X, name='Shapes4')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def prepare_network_chanal_last_classification(self, classes = 4):
        inputs = Input((self.img_rows, self.img_cols, self.num_channels))

        # zero padding
        zeroPadX = ZeroPadding2D(padding = (4, 4))(inputs)        
        conv0 = Conv2D(8, (8, 8), strides=(1, 1), padding='same', data_format='channels_last')(zeroPadX)
        conv0 = BatchNormalization(axis = 3)(conv0)
        conv0 = Activation('relu')(conv0)
        pool0 = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same', data_format='channels_last')(conv0)        
        
        conv1 = Conv2D(16, (4, 4),strides=(1, 1), padding='same', data_format='channels_last')(pool0)
        conv1 = BatchNormalization(axis = 3)(conv1)
        conv1 = Activation('relu')(conv1)        
        pool1 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same', data_format='channels_last')(conv1)
       
        conv2 = Conv2D(32, (2, 2), strides=(1, 1), padding='same', data_format='channels_last')(pool1)
        conv2 = BatchNormalization(axis = 3)(conv2)
        conv2 = Activation('relu')(conv2)        
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last')(conv2)
        
        # output layer
        X = Flatten()(pool2)
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
        #X = Dense(classes, activation=keras.activations.softmax(X, dim=axis), name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
        

        # Create model
        model = Model(inputs = inputs, outputs = X, name='Shapes4')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model    
    
    def one_hot_encodeing(self, y):
        # convert integers to dummy variables (i.e. one hot encoded)
        #dummy_y = np_utils.to_categorical(y)
        encoded_y = to_categorical(y, num_classes=4, dtype='float32')
        #nb_classes = 6
        #targets = np.array([[2, 3, 4, 0]]).reshape(-1)
        #one_hot_targets = np.eye(nb_classes)[targets]
        #return one_hot_targets
        return encoded_y

    def randomData(self, X, Y):
        m = X.shape[0]
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :]#.reshape(Y.shape[0], m)
        return shuffled_X, shuffled_Y

    def save_object(self, obj, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def noiseImagesXY(self, count=20):
        X_data = []
        Y_data = []
        for entry in range(count): 
            X = np.zeros([256, 256, 1], dtype="float_")
            Y_data.append(X.copy())
            num_noise = np.random.randint(20,256*5)
            pt_random = np.random.randint(0, 255, (num_noise, 2))           
            for (i, j) in pt_random:
                X[i, j, 0] = 1
            X_data.append(X.copy())
        X_data = np.asarray(X_data)
        Y_data = np.asarray(Y_data)                
        return X_data, Y_data
    
    def noiseImagesXY_clasification(self, count=20):
        X_data = []
        y_data = []
        for entry in range(count): 
            X = np.zeros([256, 256, 1], dtype="float_")
            num_noise = np.random.randint(20,256*5)
            pt_random = np.random.randint(0, 255, (num_noise, 2))           
            for (i, j) in pt_random:
                X[i, j, 0] = 1
            X_data.append(X.copy())
        X_data = np.asarray(X_data)
        y_data = np.zeros(shape=(X_data.shape[0], 1))         
        return X_data, y_data
        
    def readImagesXY_classification(self):
        X_r = np.load('X_data_train_rectangle.npy')
        y_r = np.ones(shape=(X_r.shape[0], 1)) * 2#np.load('y_data_train_rectangle.npy')
        X_c = np.load('X_data_train_circle.npy')
        y_c = np.ones(shape=(X_c.shape[0], 1))
        X_l = np.load('X_data_train_line.npy')
        y_l = np.ones(shape=(X_l.shape[0], 1)) * 3
        X = np.concatenate((X_r, X_c, X_l), axis = 0)
        y = np.concatenate((y_r, y_c, y_l), axis = 0)
        return X, y
    
    def readImagesXY_test_classification(self):
        X_r = np.load('X_data_test_rectangle.npy')
        y_r = np.ones(shape=(X_r.shape[0], 1)) * 2#np.load('y_data_train_rectangle.npy')
        X_c = np.load('X_data_test_circle.npy')
        y_c = np.ones(shape=(X_c.shape[0], 1))
        X_l = np.load('X_data_test_line.npy')
        y_l = np.ones(shape=(X_l.shape[0], 1)) * 3
        X = np.concatenate((X_r, X_c, X_l), axis = 0)
        y = np.concatenate((y_r, y_c, y_l), axis = 0)
        return X, y    
    
    def readImagesXY(self):
        X = np.load('X_data_train_rectangle.npy')
        y = np.load('y_data_train_rectangle.npy')
        return X, y
    
    def readImagesXY_test(self):
        X = np.load('X_data_test_rectangle.npy')
        y = np.load('y_data_test_rectangle.npy')
        return X, y
    
    def readNonClassImagesXY(self):
        X = np.load('X_data_train_circle.npy')
        y = np.zeros(X.shape)
        return X, y
        
    def readNonClassImagesXY_test(self):
        X = np.load('X_data_test_circle.npy')
        y = np.zeros(X.shape)
        return X, y    
 
    def train_channel_last_classification(self):
        model = self.prepare_network_chanal_last_classification()
        print(model.summary())
        history = LossHistory()
        for iter in range(2):
            for fileCount in range(49):
                print('File: ' + ".\\npyXYFiles\\X_data_" + str(fileCount)+ '.npy')
                X = np.load(".\\npyXYFiles\\X_data_" + str(fileCount)+ '.npy')
                y = np.load(".\\npyXYFiles\\y_data_class_" + str(fileCount)+ '.npy')                    
                #X, y = train_shapes.readImagesXY_classification()
                X_n, y_n = train_shapes.noiseImagesXY_clasification(int(len(X)/4))
                X = np.concatenate((X, X_n), axis = 0)
                y = np.concatenate((y, y_n), axis = 0)
                y = train_shapes.one_hot_encodeing(y)
                X, y = train_shapes.randomData(X, y)
                model.fit(X, y, batch_size=8, nb_epoch=20, verbose=1, callbacks=[history])
                if fileCount % 10 == 0 :
                    model.save(self.shape + '_' + str(iter) + '_' + str(fileCount) + '.h5')
                    model_json = model.to_json()
                    with open(self.shape + ".json", "w") as json_file:
                        json_file.write(model_json)
                    self.save_object(history.get_LossObj(), self.shape + '.pkl')  
            
        # save model
        # serialize model to JSON
        model_json = model.to_json()
        with open(self.shape + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5       
        model.save(self.shape + '.h5')
        self.save_object(history.get_LossObj(), self.shape + '.pkl')
        print("Saved model to disk")         
    
    
    def train_channel_last(self):
        model = self.prepare_network_chanal_last()
        print(model.summary())
        history = LossHistory()
        #imageProc = imageprocess(dataDir = self.dataDir)
        X, y = self.readImagesXY()
        X_n, y_n = self.noiseImagesXY(len(X))
        #X_n2, y_n2 = self.readNonClassImagesXY()
        #X_n3, y_n3 = self.noiseImagesXY(len(X_n2))
        #append noise
        X = np.concatenate((X, X_n, X_n2, X_n3), axis = 0)
        y = np.concatenate((y, y_n, y_n2, y_n3), axis = 0)
        X, y = self.randomData(X, y)
        print('X.shape', X.shape)
        print('Y.shape', y.shape)
        model.fit(X, y, batch_size=8, nb_epoch=11, verbose=1, callbacks=[history])
        # save model
        # serialize model to JSON
        model_json = model.to_json()
        with open(self.shape + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5       
        model.save(self.shape + '.h5')
        self.save_object(history.get_LossObj(), self.shape + '.pkl')
        print("Saved model to disk")           

    def retrieve_fitmodel_channel_last(self):
        # load json and create model
        json_file = open(self.shape + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(self.shape + '.h5')
        print("Loaded model from disk")        
        # compile
        model.compile(optimizer = Adam(lr=2e-5), loss = dice_coef_loss_mod, metrics = [dice_coef_mod])
        print(model.summary())
        
        imageProc = imageprocess(dataDir = self.dataDir)
        X, Y = imageProc.convertImagesXY()
        print('X.shape', X.shape)
        print('Y.shape', Y.shape)
        model.fit(X, Y, batch_size=8, nb_epoch=20, verbose=1, callbacks=[history])

        # save model
        # serialize model to JSON
        model_json = model.to_json()
        with open(self.shape + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5       
        model.save(self.shape + '.h5')
        print("Saved model to disk")      

    def calculate_stats_classification(self):
        # load json and create model
        json_file = open(self.shape + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(self.shape + '.h5')
        
        print("Loaded model from disk")
        
        # compile
        model.compile(optimizer = 'adam', loss = dice_coef_loss_mod, metrics = [dice_coef_mod])
        print(model.summary())

        #X, y = train_shapes.readImagesXY_classification()
        #X, y = train_shapes.readImagesXY_test_classification()
        #X_n, y_n = train_shapes.noiseImagesXY_clasification(len(X))
        #X = np.concatenate((X, X_n), axis = 0)
        #y = np.concatenate((y, y_n), axis = 0)
        #y = train_shapes.one_hot_encodeing(y)
        y_t = []
        y_pt = []
        for fileCount in range(49):
            print('File: ' + ".\\npyXYFiles\\X_data_" + str(fileCount)+ '.npy')
            X = np.load(".\\npyXYFiles\\X_data_" + str(fileCount)+ '.npy')
            y = np.load(".\\npyXYFiles\\y_data_class_" + str(fileCount)+ '.npy')                    
            X_n, y_n = train_shapes.noiseImagesXY_clasification(int(len(X)/4))
            X = np.concatenate((X, X_n), axis = 0)
            y = np.concatenate((y, y_n), axis = 0)            
            #y = train_shapes.one_hot_encodeing(y)
            #X, y = train_shapes.randomData(X, y)
            #all shapes
            #print('--------------------------------')
            #print('X.shape: ', X.shape)
            #print('y.shape: ', y.shape)
            #print('--------------------------------')            
            y_predict = model.predict(X)
            y_predict_labels = np.argmax(y_predict, axis=1)
            #print('y_predict_labels.shape: ', y_predict_labels.shape)
            y = y.reshape(y_predict_labels.shape)
            y_t = np.append(y_t, y)
            y_pt = np.append(y_pt, y_predict_labels)
            #print('y_predict:', y_predict_labels)
            #print('y_predict:', y)
        confusion = confusion_matrix(y_t, y_pt)
        print('confusion matrix:\n', confusion)
        print('classification_report:\n', classification_report(y_t, y_pt))

    def calculate_stats(self):
        # load json and create model
        json_file = open(self.shape + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(self.shape + '.h5')
        
        print("Loaded model from disk")
        
        # compile
        model.compile(optimizer = 'adam', loss = dice_coef_loss_mod, metrics = [dice_coef_mod])
        print(model.summary())

        #imageProc = imageprocess(dataDir = self.dataDir)
        #X, Y = imageProc.convertImagesXY()   
        
        #X, y = self.readImagesXY()
        #X_n, y_n = self.noiseImagesXY(len(X))
        X_n2, y_n2 = self.readImagesXY_test()
        X_n3, y_n3 = self.noiseImagesXY(len(X_n2))
        #append noise
        X = np.concatenate((X_n2, X_n3), axis = 0)
        y = np.concatenate((y_n2, y_n3), axis = 0)
        X, Y = self.randomData(X, y)
        print('X.shape', X.shape)
        print('Y.shape', y.shape)

        '''        
        X, y = self.readImagesXY()
        X_n, y_n = self.noiseImagesXY(len(X))
        #append noise
        X = np.concatenate((X, X_n, X), axis = 0)
        y = np.concatenate((y, y_n, y), axis = 0)
        X, Y = self.randomData(X, y)        
        '''
        '''
        X_n, y_n = imageProc.noiseImagesXY(len(X))
        #append noise
        X = np.concatenate((X, X_n, X), axis = 0)
        y = np.concatenate((y, y_n, y), axis = 0)
        X, Y = self.randomData(X, y) 
        '''
        #all shapes
        print('--------------------------------')
        print('X.shape: ', X.shape)
        print('Y.shape: ', Y.shape)
        print('--------------------------------')            
        Yp = model.predict(X)
        tempYp = Yp[:,:,:,0]*X[:,:,:,0]
        idx = tempYp > 0.5
        tempYp[idx] = 1.0
        idx = tempYp <= 0.5
        tempYp[idx] = 0.0  
        
        tempActYp = Yp[:,:,:,0]
        idx = tempActYp > 0.5
        tempActYp[idx] = 1.0
        idx = tempActYp <= 0.5
        tempActYp[idx] = 0.0   
        
        tempY = Y[:,:,:,0]*X[:,:,:,0]
        yyp_result = tempYp*tempY            
        m_pts = 0
        e_points = 0
        detected_pts = 0
        missing_shapes = 0
        noPattern_shapes = 0;
        pattern_shapes = 0 
        extra_detected_shapes = 0            
        for i in range(0,len(X)):
            tempYp_sum = np.sum(tempYp[i])
            tempY_sum = np.sum(tempY[i])
            yp_sum = np.sum(tempActYp[i])
            if (tempY_sum == 0 and tempYp_sum > 5):
                extra_detected_shapes += 1
                                   
                print('Extra points detected: ', tempYp_sum)
                plt.figure(1)
                plt.subplot('121')
                plt.imshow(X[i,:,:,0])
                plt.subplot('122')
                plt.imshow(Y[i,:,:,0])
                plt.show() 
                plt.figure(2)                    
                plt.subplot('121')
                temp = tempActYp[i,:,:]#*X[i,:,:,0]
                idx = temp > 0.5
                temp[idx] = 1.0
                idx = temp <= 0.5
                temp[idx] = 0.0                
                plt.imshow(temp)
                plt.subplot('122')
                temp = tempActYp[i,:,:]*X[i,:,:,0]
                idx = temp > 0.5
                temp[idx] = 1.0
                idx = temp <= 0.5
                temp[idx] = 0.0                
                plt.imshow(temp)
                plt.show() 
                                                         
            elif tempY_sum != 0 and tempYp_sum == 0 :
                missing_shapes += 1
                '''
                print('Missing points count: ', tempY_sum)
                plt.figure(1)
                plt.subplot('121')
                plt.imshow(X[i,:,:,0])
                plt.subplot('122')
                plt.imshow(Y[i,:,:,0])
                plt.show() 
                plt.figure(2)                    
                plt.subplot('121')
                temp = tempActYp[i,:,:]#*X[i,:,:,0]
                idx = temp > 0.5
                temp[idx] = 1.0
                idx = temp <= 0.5
                temp[idx] = 0.0                
                plt.imshow(temp)
                plt.subplot('122')
                temp = tempActYp[i,:,:]*X[i,:,:,0]
                idx = temp > 0.5
                temp[idx] = 1.0
                idx = temp <= 0.5
                temp[idx] = 0.0                
                plt.imshow(temp)
                plt.show() 
                '''                    
            elif tempY_sum != 0 :
                pattern_shapes += 1
                
                print('Actual points count: ', tempY_sum)
                plt.figure(1)
                plt.subplot('121')
                plt.imshow(X[i,:,:,0])
                plt.subplot('122')
                plt.imshow(Y[i,:,:,0])
                plt.show() 
                plt.figure(2)                    
                plt.subplot('121')
                temp = tempActYp[i,:,:]#*X[i,:,:,0]
                idx = temp > 0.5
                temp[idx] = 1.0
                idx = temp <= 0.5
                temp[idx] = 0.0                
                plt.imshow(temp)
                plt.subplot('122')
                temp = tempActYp[i,:,:]*X[i,:,:,0]
                idx = temp > 0.5
                temp[idx] = 1.0
                idx = temp <= 0.5
                temp[idx] = 0.0                
                plt.imshow(temp)
                plt.show()                     
                
            else :
                noPattern_shapes += 1
                '''
                plt.figure(1)
                plt.subplot('121')
                plt.imshow(X[i,:,:,0])
                plt.subplot('122')
                plt.imshow(Y[i,:,:,0])
                plt.show() 
                plt.figure(2)                    
                plt.subplot('121')
                temp = tempActYp[i,:,:]#*X[i,:,:,0]
                idx = temp > 0.5
                temp[idx] = 1.0
                idx = temp <= 0.5
                temp[idx] = 0.0                
                plt.imshow(temp)
                plt.subplot('122')
                temp = tempActYp[i,:,:]*X[i,:,:,0]
                idx = temp > 0.5
                temp[idx] = 1.0
                idx = temp <= 0.5
                temp[idx] = 0.0                
                plt.imshow(temp)
                plt.show()    
                '''
                
            yyp_result_sum = np.sum(yyp_result[i])
            detected_pts += yyp_result_sum
            m_pts += tempY_sum - yyp_result_sum
            e_points += tempYp_sum - yyp_result_sum
        
        print('-------------------------------------------------')
        print('Stats - - -')
        print('detected points: ', detected_pts)
        print('Missing points: ', m_pts)
        print('Extra Detected points: ', e_points)
        print('Shape images: ', pattern_shapes)
        print('No shape image: ', noPattern_shapes)
        print('Missing images: ', missing_shapes)
        print('Extra Detected images: ', extra_detected_shapes)          
        print('-------------------------------------------------')
                     
    def showDIceCoeffientTrend(self):
        # retrive and show
        with open(self.shape + '.pkl', 'rb') as input:
            lossObj = pickle.load(input)  
            #print('lossObj.losses = ', lossObj.losses)
            #print('lossObj.dice_coef = ', lossObj.dice_coef)  
            plt.title('Dice Coefficient')
            plt.xlabel('epoch count')
            plt.ylabel('dice Coefficient')
            #plt.plot(lossObj.dice_coef)
            plt.plot(lossObj.losses)
            plt.show() 
            
    def saveDataXY(self, listDir=[]):
        X_data = np.zeros([1, 256, 256, 1], dtype="float_")
        y_data = np.zeros([1, 256, 256, 1], dtype="float_")
        imageProc = imageprocess(dataDir = '')
        for dirName in listDir:
            imageProc.dataDir = dirName
            X, y = imageProc.convertImagesXY()
            X_data = np.concatenate((X_data, X.copy()), axis = 0)
            y_data = np.concatenate((y_data, y.copy()), axis = 0)
        np.save('X_data', X_data)
        np.save('y_data', y_data)          
        
    def showDataXY(self, X_data, y_data):
        print(X_data.shape)
        print(y_data.shape)
        for i in range(0,len(X_data)):
            plt.figure(1)
            plt.subplot('121')
            plt.imshow(X_data[i, :, :, 0])
            plt.subplot('122')
            plt.imshow(y_data[i, :, :, 0])
            plt.show()  
        
if __name__ == '__main__':
    gpu_id = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    train_shapes = trainShapes(shape='CLASSIFICATION', dataDir = './train_images')
    #train_shapes.saveDataXY(listDir=['train_line_images_data', 'train_line_images _data_rotate_left', 'train_line_images_data_rotate_right', 'train_line_images_data_rotate_right_right'])
    #train_shapes.saveDataXY(listDir=['train_circles_images_data'])
    
    train_shapes.train_channel_last_classification()
    #train_shapes.calculate_stats_classification()
    
    #train_shapes.train_channel_last()
    #train_shapes.showDIceCoeffientTrend()
    #train_shapes = trainShapes(shape='CIRCLE', dataDir = './test_images')
    #train_shapes.calculate_stats()
    #print(train_shapes.onehot_encodeing())
    
    #X, y = train_shapes.readNonClassImagesXY()
    #train_shapes.showDataXY(X, y)
    '''
    X, y = train_shapes.readImagesXY_classification()
    X_n, y_n = train_shapes.noiseImagesXY_clasification(len(X))
    #append noise
    X = np.concatenate((X, X_n), axis = 0)
    y = np.concatenate((y, y_n), axis = 0)
    print(X.shape)
    print(y.shape)
    df = pd.DataFrame(y)
    print(df[0].unique())
    encode_y = train_shapes.one_hot_encodeing(y)
    X, y = train_shapes.randomData(X, encode_y)
    print(y[0:20])
    '''
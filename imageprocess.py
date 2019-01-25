# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:55:13 2019

@author: Sreenivasulu Bachu
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
import os, fnmatch
from scipy import ndimage
from PIL import Image
from random import shuffle

class imageprocess(object):
    def __init__(self, dataDir = 'image'):
        self.dataDir = dataDir
    
    def convertImage(self, imageFileName):
        image = imageio.imread(imageFileName)
        redImage = image[:, :, 0] == 0
        image[redImage] = 255
        redImage = image[:, :, 0] != 255
        X = np.zeros([256, 256, 1], dtype="float_")
        X[redImage] = 1.0
        #print(X.shape)
        #plt.imshow(X)
        #plt.show()
        return X
    
    def convertImagesXY(self):
        X_data = []
        Y_data = []
        fileList = os.listdir(self.dataDir)  
        pattern = "*.png"   
        for entry in fileList:  
            if fnmatch.fnmatch(entry, pattern):
                X = self.convertImage(".\\"+self.dataDir+"\\"+entry)
                Y_data.append(X.copy())
                num_noise = np.random.randint(20,1024)
                pt_random = np.random.randint(0, 255, (num_noise, 2))           
                for (i, j) in pt_random:
                    X[i, j, 0] = 1
                X_data.append(X.copy())
        X_data = np.asarray(X_data)
        Y_data = np.asarray(Y_data)                
        return X_data, Y_data
    
    def convertImagesXYImageFile(self, pngFileName):

        X_data = []
        Y_data = []

        X = self.convertImage(".\\"+self.dataDir+"\\"+pngFileName)
        Y_data.append(X.copy())
        num_noise = np.random.randint(20,1024)
        pt_random = np.random.randint(0, 255, (num_noise, 2))           
        for (i, j) in pt_random:
            X[i, j, 0] = 1
        X_data.append(X.copy())
        X_data = np.asarray(X_data)
        Y_data = np.asarray(Y_data)                
        return X_data, Y_data

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
    
    def rotateImage(self, rotation_angle=[90]):
        '''
        image = imageio.imread('C:\\NO_BACKUP\\gitHub\\deepLearning\\imageprocess\\train_line_images_data\\line_0.png')
        rotate_image = ndimage.rotate(image, 45)
        img = Image.fromarray(rotate_image)
        img = img.resize((256, 256))
        img.save('C:\\NO_BACKUP\\gitHub\\deepLearning\\imageprocess\\train_line_images_data\\line_0_rotate_45.png')
        '''
        fileList = os.listdir(self.dataDir)  
        pattern = "*.png"   
        for entry in fileList:  
            if fnmatch.fnmatch(entry, pattern):
                image = imageio.imread(".\\"+self.dataDir+"\\"+entry)
                for rotation in rotation_angle: 
                    rotate_image = ndimage.rotate(image, rotation)
                    img = Image.fromarray(rotate_image)
                    img = img.resize((256, 256))
                    img.save(".\\"+self.dataDir+"\\"+ entry.split('.')[0]+'_'+str(rotation)+'.'+entry.split('.')[1])   

    def saveDataXY(self, dirList=[], xyDataDir=""):
        X_data = []
        y_data = []
        y_data_classification = []
        filesList = []
        for dirName in dirList:
            fileList = os.listdir(dirName)  
            pattern = "*.png"   
            for entry in fileList:  
                if fnmatch.fnmatch(entry, pattern):
                    filesList.append(".\\"+dirName+"\\"+entry)
                    
        shuffle(filesList)
        dataCount = 0;
        npFileCount = 0;
        for patFile in  filesList: 
            X = self.convertImage(patFile)
            y_data.append(X.copy())
            #np.concatenate((y_data, X.copy()), axis = 0)
            num_noise = np.random.randint(20,1024)
            pt_random = np.random.randint(0, 255, (num_noise, 2))           
            for (i, j) in pt_random:
                X[i, j, 0] = 1
            X_data.append(X.copy())
            #np.concatenate((X_data, X.copy()), axis = 0)
            if 'circle_' in patFile:
                y_data_classification.append(1)
            elif 'rectangle_' in patFile:
                y_data_classification.append(2)
            elif 'rect_' in patFile:
                y_data_classification.append(2)
            elif 'line_' in patFile:
                y_data_classification.append(3)
            else:
                y_data_classification.append(0)
                
            y_data_classification
            dataCount = dataCount + 1
            if (dataCount == 200):
                np.save(".\\"+xyDataDir+"\\" + 'X_data_'+str(npFileCount), np.asarray(X_data))
                np.save(".\\"+xyDataDir+"\\" +'y_data_'+str(npFileCount), np.asarray(y_data))
                y_data_classification_ar = np.asarray(y_data_classification)
                y_data_classification_ar = y_data_classification_ar.reshape(y_data_classification_ar.shape[0], 1)
                np.save(".\\"+xyDataDir+"\\" +'y_data_class_'+str(npFileCount), y_data_classification_ar)
                dataCount = 0
                npFileCount = npFileCount + 1
                X_data = []
                y_data = []
                y_data_classification = []

                
if __name__ == '__main__':
    #gpu_id = 0
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    imageProc = imageprocess(dataDir = 'train_rectangle_data')
    #imageProc.saveDataXY(dirList=['train_circles_images_data', 'train_rectangle_data', 'train_line_images_data'], xyDataDir='npyXYFiles')
    imageProc.saveDataXY(dirList=['test_circle_images', 'test_line_images', 'test_rectangle_images'], xyDataDir='npyXYFiles-test')
    #imageProc.rotateImage([20])
    #imageProc.rotateImage([x * 5 for x in range(1, 73) ])
    '''
    #X_data, Y_data = imageProc.convertImagesXY()
    for n in range(0, 361):
        #fileName = 'rectangle_37_'+str(n)+'.png'
        fileName = 'rectangle_2_'+str(n)+'.png'
        #fileName = 'rectangle_0_60'+'.png'
        if(not os.path.isfile(".\\"+'train_rectangle_data'+"\\"+fileName)):
            print('File not exit: ' + fileName)
            continue;
        X_data, Y_data = imageProc.convertImagesXYImageFile(fileName)
        #X_data = np.asarray(X_data)
        #Y_data = np.asarray(Y_data)
        #print(X_data.shape)
        #print(Y_data.shape)
        print(fileName)
        for i in range(0,len(X_data)):
            plt.figure(1)
            plt.subplot('121')
            plt.imshow(X_data[i, :, :, 0])
            plt.subplot('122')
            plt.imshow(Y_data[i, :, :, 0])
            plt.show()      
    '''
        


# Pattern Recognition CNN Model
Many Hi-Tech or semiconductor industry classifying defects patterns and as well as location of defects are important to identify which resource/equipment causing these defects. Keeping this in mind Pattern recognition CNN models classify different shapes and as well as predicts where these shapes occurred. This problem divided into 2 steps.
* Classify shapes
* Predicts points

Here is the overall process:
* Generate synthetic image data of various images

      Generated 256x256 pixels images of shapes (circle, rectangle/square, line, and no shape)
 * Generate numpy array of images and randomize the data (which is very important)
 
      Shape of the arrays nx256x256x1
      As we are dealing with detects color is not important and hence drop other 2 channels and update image array such that cell which has shape pixel turn it to '1' otherwise '0', so that learning algorithm learn much faster.
* Train classification model. Here is the classification model network
      
      _________________________________________________________________
      Layer (type)                 Output Shape              Param #   
      =================================================================
      input_2 (InputLayer)         (None, 256, 256, 1)       0         
      _________________________________________________________________
      zero_padding2d_2 (ZeroPaddin (None, 264, 264, 1)       0         
      _________________________________________________________________
      conv2d_4 (Conv2D)            (None, 264, 264, 8)       520       
      _________________________________________________________________
      batch_normalization_4 (Batch (None, 264, 264, 8)       32        
      _________________________________________________________________
      activation_4 (Activation)    (None, 264, 264, 8)       0         
      _________________________________________________________________
      max_pooling2d_4 (MaxPooling2 (None, 33, 33, 8)         0         
      _________________________________________________________________
      conv2d_5 (Conv2D)            (None, 33, 33, 16)        2064      
      _________________________________________________________________
      batch_normalization_5 (Batch (None, 33, 33, 16)        64        
      _________________________________________________________________
      activation_5 (Activation)    (None, 33, 33, 16)        0         
      _________________________________________________________________
      max_pooling2d_5 (MaxPooling2 (None, 9, 9, 16)          0         
      _________________________________________________________________
      conv2d_6 (Conv2D)            (None, 9, 9, 32)          2080      
      _________________________________________________________________
      batch_normalization_6 (Batch (None, 9, 9, 32)          128       
      _________________________________________________________________
      activation_6 (Activation)    (None, 9, 9, 32)          0         
      _________________________________________________________________
      max_pooling2d_6 (MaxPooling2 (None, 5, 5, 32)          0         
      _________________________________________________________________
      flatten_2 (Flatten)          (None, 800)               0         
      _________________________________________________________________
      fc4 (Dense)                  (None, 4)                 3204      
      =================================================================

      ## Loss and accuracy trends
  
  ![Image](/images/loas_accuracy_trend.png)

* Evaluate the model using test data. Here is the train and test results (confusion matrix, F1 scores, etc.)

      ## train
      confusion matrix:
       [[2550    0    0    0]
       [   0 3526   36   11]
       [   0   33 3162    6]
       [   7    3    3 3413]]
       
      classification report:
                     precision    recall  f1-score   support

               0.0       1.00      1.00      1.00      2550
               1.0       0.99      0.99      0.99      3573
               2.0       0.99      0.99      0.99      3201
               3.0       1.00      1.00      1.00      3426

         micro avg       0.99      0.99      0.99     12750
         macro avg       0.99      0.99      0.99     12750
      weighted avg       0.99      0.99      0.99     12750

      ## test
      confusion matrix:
       [[1000    0    0    0]
       [   0 1358   67   35]
       [   0   41 1026   13]
       [  11    7    2 1440]]
       
      classification report:
                     precision    recall  f1-score   support

               0.0       0.99      1.00      0.99      1000
               1.0       0.97      0.93      0.95      1460
               2.0       0.94      0.95      0.94      1080
               3.0       0.97      0.99      0.98      1460

         micro avg       0.96      0.96      0.96      5000
         macro avg       0.96      0.97      0.97      5000
      weighted avg       0.96      0.96      0.96      5000


Based on the results model has overfits the training data.

## Next Steps:

      Retrain the model using regularization/dropout units. Retrain using different model/network.
      Train model to predict shape points

## Some of images
![Image](/images/shapeImage2.png) ![Image](/images/shapeImage1.png) 
![Image](/images/shapeImage3.png) ![Image](/images/shapeImage4.png) 
![Image](/images/shapeImage5.png)


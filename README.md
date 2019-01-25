# Pattern Recognition CNN Model
Many Hi-Tech or semi conductor industry classifying defects patterns and as well as location of defects are important to identify  
Train the breast cancer data using tensor flow and keras models. Both cases used same network.
Here are the steps:
* Extract input feature

      Randomize input data and randomize batches at the time of traing is required to avoid overfit the data.
      Make sure the matrix size matches with the network.

* Train and evaluate model to see how best it performs on test data
* Save model and weights
* Retrive model and evaluate with new data

Tried with different networks but the data fitted with very well with the following network
* Input has 9 features
* Hidden layers with the 8, 8, 4 and 4 nuerans
* Output layers with 1 nueran to predict cancer or non cancer

ReLu non linear activatation has used for input and hidden layers.

## TensorFlow Model
* Create place holders
* Initialize weights
* Farward propagation
* Compute cost
    
      Calculated cost using sigmoid cross entropy with logits by reducing mean.
    
* optimize
      
      Optimized using Adam optimizer


## TKeras Model
* Build network

      Build network using sequential model. Make sure first dense layer matched with input size.
      Add remaining dense layers with ReLu non linear activation.
      Final dense layer added with size one with sigmoid activation  
    
* Compile model
  
      Compile the model using rmsprop optimize and binary cross entropy loss with accuracy matrics
* Train and evaluate Model



## DNN Network
![Image](/images/nnNetwork.jpg)

### Challenges

      Current model unable to handle if input misses any feature. How to handle this situation? 

Note: Train the models using the data published by UCI - Machine Learning Repositsry (Breast Cancer Wisconsin (Diagnostic) Data Set). [Dataset](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

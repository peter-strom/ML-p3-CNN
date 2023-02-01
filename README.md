# Project III - Machine learning and sensor technology - Ela21, YRGO

Supervisor: Erik Pihl 

By: Daniel Mentzer and Peter Strömblad
            
## Convolutional Neural Network

## Introduction
This project is based on [![Project II - A neural network in an embedded system](https://github.com/peter-strom/ML-p2-Neural_net_embedded)] and was an optional task where we examined and learned about the convolutional layer.
Convolutional layers are used in neural networks as a tool to extract details and important context from images and at the same time remove non informative sections to make the image small and resource effective enough to use as training-data. 
## Discussion
This bonus project was full of fun problems to solve. The first one was to figure out how a simple bitmap was constructed using a hex editor, se Fig. 1. And what the purpose of each kernel that a CNN uses to extract data and or make the image smaller. The rest was straightforward nestling for-loops. But we didn’t quite understand how to (or if it was needed to) optimize the weights in the kernel. We tried with randomized values in the kernel but was easier to visualize when all weights were set to 0.5.  
Our time ran away and finally, we end up using the Convolutional layer simply as a tool to make training-data for our neural network. Our conclusion after this project is that we still don't understand everything about the kernels fully usage and its optimization. But this was a great exercise that was interesting to visualize.

![alt text](https://github.com/peter-strom/ML-p3-CNN/blob/222883ed8bd85c40ba873919ed640a5f684ecb9f/img/fig1.png)

Fig.1 - A simple bitmap viewed in a hex editor. 
  
 <br><br> 
![alt text](https://github.com/peter-strom/ML-p3-CNN/blob/222883ed8bd85c40ba873919ed640a5f684ecb9f/img/fig2.png)

Fig.2 - The convolutional layer extracted details from a 15x15 image. And resulted in a 7x7 image after the pooling was done. The image was then flattened to a 1-dimensional vector that could be fed to our neural network..
  
<br><br>  
![alt text](https://github.com/peter-strom/ML-p3-CNN/blob/222883ed8bd85c40ba873919ed640a5f684ecb9f/img/fig3.png)

Fig.3 - The neural network ran our training data through its nodes. The output layer had 4 nodes so it could represent a byte. Turned out to work very well with this simple test.   


---
title: "Transfer learning"
sequence: 5
---
# Introduction

In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest.

There are two major transfer learning scenarios:

- Finetuning the network: Instead of random initializaion, we initialize the network with a pretrained network, similar to the one that is trained on imagenet 1000 dataset. The rest of the training looks as usual. It is common to use a smaller learning rate because we expect that the ConvNet weights are relatively good, so we don’t wish to distort them too quickly and too much (especially while the new Linear Classifier above them is being trained from random initialization).


- ConvNet as fixed feature extractor: Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (this layer’s outputs are the 1000 class scores for a different task like ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. Finally train a linear classifier (e.g. Linear SVM or Softmax classifier) for the extracted ConvNet features.

source: Stanford computer vision course: CS231 http://cs231n.github.io/transfer-learning/

# Finetuning
Finetuning an existing network with Tensorflow can be done in different ways. The easiest way is to use the Slim environment, where you only have to define the model that you want to use and the desired folder structure. 
https://github.com/tensorflow/models/tree/master/slim#Pretrained

In the online Tensorflow repository there is description on how to train/ finetune your model with some arguments via the terminal.
https://github.com/tensorflow/models/tree/master/slim#Tuning

Still if you want to tweak the parameters yourself it is more advisable to implement the model itself so that you can tweak the learning rate, the dropout and some other loss operations. 

One layered network 
https://www.datacamp.com/community/tutorials/tensorflow-tutorial#gs.71yg_K4 (traffic sign detection code tensorflow)


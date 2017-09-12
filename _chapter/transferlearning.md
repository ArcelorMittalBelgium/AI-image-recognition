---
title: "Transfer learning"
sequence: 4
---

In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest.


# Introduction

There are two major transfer learning scenarios:

- **Finetuning the network**: Instead of random initializaion, we initialize the network with a pretrained network, similar to the one that is trained on imagenet 1000 dataset. The rest of the training looks as usual. It is common to use a smaller learning rate because we expect that the ConvNet weights are relatively good, so we don’t wish to distort them too quickly and too much (especially while the new Linear Classifier above them is being trained from random initialization).


- **ConvNet as fixed feature extractor**: Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (this layer’s outputs are the 1000 or more class scores for a different task like ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. Finally train a linear classifier (e.g. Linear SVM or Softmax classifier) for the extracted ConvNet features.

Source: [Stanford computer vision course: CS231](http://cs231n.github.io/transfer-learning/)

# Finetuning
Finetuning an existing network with Tensorflow can be done in different ways. The easiest way is to use the [Slim environment](https://github.com/tensorflow/models/tree/master/slim#Pretrained), where you only have to define the model that you want to use and the desired folder structure. 

In the online Tensorflow repository there is a [description on how to train/finetune your model](https://github.com/tensorflow/models/tree/master/slim#Tuning) with some arguments via the terminal.

Still if you want to tweak the parameters yourself it is more advisable to implement the model itself so that you can tweak the learning rate, the dropout and some other loss operations. 
The [tensorflow_vgg.py code example](../../code/tensorflow_vgg.py){:target="_blank"} gives an implementation of the VGG network as a fixed feature extractor and where the last fully connected layer is replaced and retrained on the new amount of classes.

# Dataset class imbalance
This is common problem in machine learning, the number of data samples for particular classes is extremely larger than the number of samples of another class. 

The solution to this problem can be tackled in two ways:

- A cost based function approach:
Set the punishment for the wrongly classified examples of the minority classes higher then those for the majority classes.

- Sampling strategy approach:
    - Oversampling, adding more of the minority classes in combination with data-augmentation,
    - Undersampling,  removing some of the majority classes or randomly selecting some of them,
    - Hybrid approach.

Furthermore, it is important to use alternative metrics (i.e. recall, precision, confusion-matrices) instead of the accuracy to validate and optimize the model. 

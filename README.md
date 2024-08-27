# Welcome to Deep Learning Tutorial using Keras 
In this tutorial you will learn how to build a deep learning sequential model using one of the famous deep learning API Keras. 

## Convolutional Neural Network (CNN) Model from Scratch

This repository contains step by step demonstration that how to build a Convolutional Neural Network (CNN) model from scratch using TensorFlow and Keras. The project also includes a comparison of building a model from scratch with a fine-tuned VGG16 model.

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Custom CNN Model](#custom-cnn-model)
  - [Model Architecture](#model-architecture)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [Fine-Tuning VGG16](#fine-tuning-vgg16)
- [Conclusion](#conclusion)
- [References](#references)

## Overview

This project involves building a CNN model from scratch to classify images. The dataset used in this project is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes.

## Data Preparation

The CIFAR-10 dataset is preprocessed before feeding it into the CNN model. The data is normalized and split into training and testing sets.

## Custom CNN Model

### Model Architecture

The CNN model is built from scratch with the following layers:

- **Convolutional Layers:** Extract features from the input images.
- **Pooling Layers:** Reduce the spatial dimensions of the feature maps.
- **Fully Connected Layers:** Perform classification based on the extracted features.

### Training the Model

The model is trained using the Adam optimizer and categorical cross-entropy loss function. The training process includes:

- **Batch Size:** 32
- **Epochs:** 50
- **Learning Rate:** 0.001

### Evaluating the Model

The performance of the CNN model is evaluated on the test dataset, with accuracy and loss being the primary metrics.

## Fine-Tuning VGG16

A pre-trained VGG16 model is fine-tuned on the CIFAR-10 dataset. The top layers are replaced with custom fully connected layers to adapt to the 10-class classification problem.

## Conclusion

The custom CNN model built from scratch demonstrates a solid understanding of deep learning concepts. Fine-tuning the VGG16 model further improves the performance, showcasing the benefits of transfer learning.

## References

- [Keras Documentation](https://keras.io)
- [TensorFlow Documentation](https://www.tensorflow.org)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

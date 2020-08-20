---
title: "Training a Convolutional Neural Network for Object Detection"
date: 2020-04-18T15:34:30-04:00
categories:
  - project
tags:
  - Machine Learning
  - Python
---

Once a dataset of images has been generated, we can begin training a model. Instead of creating a model's architecture from scratch we can use a premade one called, SSD Inception V2 based off of the Tensorflow Object Detection API. Single Shot Detection (SSD) is a type of architecture that focuses on speed of inferencing without sacrificing accuracy. We will perform transfer learning which means we will overwrite the weights in the output layer of the pre-trained model based on the classes we define from our dataset.
![image](/assets/images/ssdlayers.jpg)
*The SSD architecture with Inception V2 feature extractor blocks*
<br />
Our first step is to create our virtual environment with all the modules and dependecies needed to train the model. 

> - Python 3.6.8
> - Tensorflow 1.14
> - OpenCV 4.1.1
> - Numpy 1.7
> - Absl-py 0.8.1
> - Cython 0.29.13
> - IPython 7.8
> - Lxml 4.4.1
> - Matplotlib 3.1.1
> - Pandas 0.25.1
> - Pillow 6.2
> - Pip 19.2.3
> - Protobuf 3.10
> - Pycocotools 2.0
> - Wheel 0.33.6

Once the 

---
title: "Machine Learning Problem Framing"
date: 2020-04-16T15:34:30-04:00
categories:
  - Capstone Project
tags:
  - Machine Learning
  - Python
toc: true
toc_label:
toc_icon: 'bars'
classes: 

---

# Introduction
When faced with a business challenge that may require a machine learning solution, we must consider the following. What is the ML (machine learning) problem type? What are the data sources? How will the output be used? What metrics will deem the model successful? How will the finished model be properly implemented into the workflow? What are some possible non-ML solutions? These are some of the questions my collegues, Tisha, Chris, Mohammed, and I had to ask ourselves for our University of Illinois capstone project when we were tasked with exploring the first machine learning implementation at Fermi National Accelerator Labratory (Fermilab). 
# Fermilab's Problem
Fermilab has a particle accelerator that fires high power protons for various experiments. This means that the accelerator itself becomes too radioactive for maintenance workers to easily work on it. They currently need to use hot cells with mechanical arms and PPE to perform maintenance on the accelerator. This process takes a long time and requires extensive training for the workers. Even with PPE the workers are still at risk of accumulating radiation over time so implementing a ML approach to maintenance will not only save Fermilab time and money but also help promote the health and safety of its workers.
# Proposed Solution
## Input to ML Algorithm
The first step of any maintenance work on the accelerator is to remove any nuts and bolts to gain access to the inner parts. Therefore it would be helpful for a robotic arm to automatically remove these fasteners without the need of a human worker. In order to achieve this we need to implement a computer vision algorithm that would be able to locate these fasteners in three-dimensional space. The robot will "see" the fastener using a 3D camera and will calculate the coordinates of said fastener in order to travel that distance. The camera we used is the Intel Realsense D345 which has RGB sensing and stereoscopic depth sensing technology built in along with a robust SDK for scripting purposes. 
## The ML Algorithm
The task of "seeing" an object in a video feed requires using a computer vision solution. There are a number of computer vision technologies that don't utilize machine learning, but they are only applicible in a very uniform environment like a factory's assembly line where a camera wouldn't have much variation in the images it sees. For our situation there are varying angles, lighting conditions, backgrounds, colors, and even types of fasteners on the accelerator. In order to filter through all the unnecessary information to locate a nut or bolt in an image we will use a type of machine learning called object detection. Object detection utilizes a type of machine learning called deep learning which relies on big datasets and powerful hardware for the learning process. We will be supervising the learning process which means we will feed the model large quantities of images of nuts and bolts that are labeled accordingly. This is in contrast to unsupervised learning which lets the algorithm create its own classifications depending on the features it extracts from the dataset. That's not useful for us because we already know what kind of fasteners exist on the accelerator. Deep neural networks have multiple hidden layers between the input and output that extract complex features a human would never be able to find. The model will learn these features specific to a nut and a bolt making it so it shouldn't matter where the camera is positioned or what part of the accelerator the camera is pointed at.
## Output from ML Algorithm
The scope of our project was limited by the time and budget given to us as well as the early semester release due to the COVID-19 quarantine. However the future work built from our completed prototype looks quite promising. Our 3D camera using our trained model calculates the 3D coordinates of any nut and bolt it sees. This easily allows a robotic arm that is connected to our platform to reach the fastener once it is spacially calibrated to the environment. A high quality arm with six degrees of freedom and a force feedback gripper was out of our budget and beyond the focus of our project, but is necessary for the completion of the task at hand for semi-autonomous maintenance in hazardous environments.

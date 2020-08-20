---
title: "Training a Convolutional Neural Network for Object Detection"
date: 2020-04-18T15:34:30-04:00
categories:
  - Project
tags:
  - Machine Learning
  - Python
toc: true
toc_label:
toc_icon: 'bars'
classes: wide

---

Once a dataset of images has been generated, we can begin training a model. Instead of creating a model's architecture from scratch we can use a premade one called, SSD Inception V2 based off of the Tensorflow Object Detection API. Single Shot Detection (SSD) is a type of architecture that focuses on speed of inferencing without sacrificing accuracy. We will perform transfer learning which means we will overwrite the weights in the output layer of the pre-trained model based on the classes we define from our dataset.

<br />

|![image](/assets/images/ssdlayers.jpg)|
|:--:|
|*The SSD architecture with Inception V2 feature extractor blocks*|

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

Once the packages have been downloaded we must create a folder in *'tensorflow/models/research/object_detection'* called *'training'*. Place the test.record and train.record datasets in this folder. We also need to create a file called labelmap.pbtxt and fill it with the following.


```
item {
  id: 1
  name: 'nut'
}

item {
  id: 2
  name: 'bolt'
}
```

Then download the SSD Inception V2 model files found [here](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz). Place the pipeline.config and all the model.ckpt files in the training folder. These contain the pretrained model with weights based off of the Coco dataset which has 90 different objects it classifies. For this project we need to edit the pipeline.config file so num_classes is equal to 2. We also have to edit fine_tune_checkpoint, label_map_path, and input_path (lines 152-172) to point to the paths of the checkpoint files (CLARIFY THE CKPT FILE PATH THING), labelmap file and .record files respectively.

<br />

Now that we have the pipeline.config, labelmap.pbtxt, train.record, test.record and model.ckpt files in the training folder we can begin training the model. Open up the terminal and change the directory to *~/research* and run the following.
```bash
export PYTHONPATH =$PYTHONPATH:pwd:pwd/slim
```
You must do this everytime you open up a new terminal. To avoid this put that line at the end of your .bashrc file and replace *'pwd'* with the full path to the research folder in tensorflow.

<br />

Now change the directory to *~/object_detection* and start the training script by running the following in the terminal.

```bash
python3 model_main.py
--logtostderr
--model_dir=Training/
--pipeline_config_path=Training/pipeline.config
```


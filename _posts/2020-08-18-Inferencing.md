---
title: "Deploying Trained Object Detection Model"
date: 2020-08-18T15:34:30-04:00
categories:
  - Project
tags:
  - Machine Learning
  - Python
header:
  image:
  teaser: /assets/images/inference.jpg
toc: true
toc_label:
toc_icon: 'bars'
classes: wide

---
# Introduction
# Prepping the Loop
## Import Packages
Now that we've successfully trained the object detection model we can finally put it to use! We will create a script that will perform inferencing using a stereoscopic camera to detect the three dimensional coordinates of nuts and bolts in realtime. We will be using an Intel Realsense D435 Depth Camera and its corresponding SDK for this project. 
Before we do anything we must import the necessary libraries.
```python
import os
import cv2
import numpy as np
import tensorflow as tf
import pyrealsense2 as rs
from utils import visualization_utils as vis_util
from utils import label_map_util
```

<br />
## Initialization
Next we need to intitialize a few things. First we initialize the two sensors on the camera. One is the infrared depth sensor and the other is the color sensor. 

```python
# Video Dimensions
WIDTH = 848
HEIGHT = 480

# initialize realsense cam
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8,30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
```

<br />
## Loading Trained Model
Then we have to load the label map and trained model into memory in order to perform inferencing in realtime.

```python
# current directory
CWD_PATH = os.getcwd()

# Name of directory containing object detection model
MODEL_NAME = 'inceptionV2'

# Path to frozen graph .pb
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'labels', 'labelmap.pbtxt')

# Load the label map
NUM_CLASSES = 2
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# load TF model into memory
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    TFSess = tf.compat.v1.Session(graph=detection_graph)
```

<br />
## Defining Inputs and Outputs
Now we must define the variables that will act as the input and outputs of our model. The only input we will have is the array of pixels we recieve from each frame of the video. The three outputs we will get from our model are the top-left X,Y pixel coordinates of the bounding boxes, the class of object it detects, and the percent confidence in the object detection.

```python
# Define input and output tensors:
# input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
```

<br />
## Create the Window
Finally we prepare the window that will show the video.

```python
# Initialize framerate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Name the window
WIN_NAME = 'Fermi Detection'

# create window
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(WIN_NAME, 120, 500)
```

<br />
# Object Detection
Now that we've finished initializing everything, we can work on taking the data from the live video feed and feeding it through our model in order to detect the desired objects in frame.

```python
while(True):
    t1 = cv2.getTickCount()

    # retrieve video feed and its depth and color data
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
    	continue
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # get vertices (need to map depth to color)
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    #vtx = np.asanyarray(points.get_vertices())
    #txt = points.get_texture_coordinates()
    #txt2 = np.asanyarray(points.get_texture_coordinates())
    #print(txt2[0])
    #values = get_texcolor(color_frame, txt[0])
    #print(values)

    #  We get a frame from the video, and we expand its dimensions to the tensor shape
    #  [1, None, None, 3]
    frame_expanded = np.expand_dims(color_image, axis=0)

    # We perform the detection of objects, providing the video image as input
    (boxes, scores, classes, num) = TFSess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
        <br />
    # Draw the bounding box, class, confidence
    vis_util.visualize_boxes_and_labels_on_image_array(
        color_image,
        np.atleast_2d(np.squeeze(boxes)),
        np.atleast_1d(np.squeeze(classes).astype(np.int32)),
        np.atleast_1d(np.squeeze(scores)),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.70)

    intrin = color_frame.profile.as_video_stream_profile().intrinsics

    # retrieve coordinates for every box per frame (sometimes glitches and draws everywhere)
    for box in boxes[0]:
    	if box[0] != 0 and box[1] != 0 and box[2] != 0 and box[3] != 0:
            xmin = int(box[1]*848)
            ymin = int(box[0]*480)
            distance = depth_frame.get_distance(xmin,ymin)
            #dmm = int(distance*1000)
            #cv2.rectangle(color_image,(xmin,ymin-20),(xmin+80,ymin-40),(0,255,0),-1)
            #cv2.putText(color_image,str(dmm)+"mm",(xmin,ymin-20),font,0.4,(255,255,0),2)
            xyvalues = rs.rs2_deproject_pixel_to_point(intrin, [xmin,ymin], distance*depth_scale)
            xvalue = int(xyvalues[0]*1000000)
            yvalue = int(xyvalues[1]*1000000)
            zvalue = int(xyvalues[2]*1000000)
            cv2.putText(color_image,"(X,Y,Z): ("+str(xvalue)+", "+str(yvalue)+", "+str(zvalue)+") mm",(xmin, ymin-20), font,0.4,(255,255,0),2)

    # Calc and draw FPS
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    cv2.putText(color_image,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

    # Update window
    cv2.imshow(WIN_NAME, color_image) 
    # To show both color and depth camera:
    # images = np.hstack((color_image, depth_colormap))
    # cv2.imshow('RealSense', images) 

    # press escape to quit
    if cv2.waitKey(1) == 27:
        print("Exiting...")
        break

pipeline.stop()
cv2.destroyAllWindows()
```

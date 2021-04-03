---
layout: single
header:
  overlay_color: "#333"
  teaser: /assets/images/algorithm/mask.jpeg
title:  "Face Mask Detection Using Tensorflow"
excerpt: "Deep Learning"
breadcrumbs: true
share: true
permalink: /mask/
date:    2020-12-02
toc: false

---
# Welcome to Face Mask Detection 👋
### 🏠 [Homepage](https://github.com/devil-cyber/Mask-Detection)
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000)
[![Documentation](https://img.shields.io/badge/documentation-yes-brightgreen.svg)](https://github.com/devil-cyber/Mask-Detection/README.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
[![Twitter: Manikan31004419](https://img.shields.io/twitter/follow/Manikan09676833.svg?style=social)](https://twitter.com/Manikan09676833)
[![Demo](https://img.shields.io/badge/Demo-Live%20project%20demo-blue)](https://share.streamlit.io/devil-cyber/mask-detection/app.py)

> This is a Faster RCNN based object detection model that detects the person face mask.It can clearly detect face mask in group of people with a great ease.

![Demo](https://github.com/devil-cyber/Mask-Detection/raw/master/video.gif)

![](https://camo.githubusercontent.com/b472b8dbb334025196143f5e0480f1f609145adc1f51a1df844a0af66376e79c/68747470733a2f2f617070732e73747265616d6c697475736572636f6e74656e742e636f6d2f646576696c2d63796265722f6d61736b2d646574656374696f6e2f6d61737465722f6170702e70792f2b2f6d656469612f31613261396262653935373038643430396539643434343961616561343439366331313830653535633162346364316131303838613531382e6a706567)

 



## Install

```python

# To get started with this project first create env. if you have Anaconda then create env using below command:
conda create -n [env name] python=3.6
# If you does not have Anaconda the create env using virtualenv follow below command:
pip install virtualenv
python -m [env name] env
# After creating enivornment now install all requirements
pip install -r requirements.txt

```


## Run App

```sh
python mask_detection.py
```
## Dataset
- `You can get dataset from kaggle` [Face Mask Data](https://www.kaggle.com/andrewmvd/face-mask-detection)
- `You can also create own dataset by clicking own pic with Mask and Without Mask using`  create_dataset.py file
> The code look like :

```python

import sys
import os
import cv2
desc = '''Script to gather data images with a particular label.
Usage: python gather_images.py <label_name> <num_samples>
The script will collect <num_samples> number of images and store them
in its own directory.
Only the portion of the image within the box displayed
will be captured and stored.
Press 'a' to start/pause the image collecting process.
Press 'q' to quit.
'''


try:
    label_name = sys.argv[1]
    num_samples = int(sys.argv[2])
except:
    print("Arguments missing.")
    print(desc)
    exit(-1)

IMG_SAVE_PATH = 'image_data'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass
try:
    os.mkdir(IMG_CLASS_PATH)
except FileExistsError:
    print("{} directory already exists.".format(IMG_CLASS_PATH))
    print("All images gathered will be saved along with existing items in this folder")

cap = cv2.VideoCapture(0)

start = False
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if count == num_samples:
        break

    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

    if start:
        roi = frame[100:500, 100:500]
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, roi)
        count += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
                (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start

    if k == ord('q'):
        break

print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()

```
> The setup Code

```python
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import warnings

from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

class Main:
    def __init__(self):
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_FROZEN_GRAPH = os.getcwd() + '/inference_graph/frozen_inference_graph.pb'
        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = os.getcwd() + '/training/labelmap.pbtxt'
    def graph(self):
          print("> ====== Loading frozen graph into memory")
          detection_graph = tf.Graph()
          with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
              serialized_graph = fid.read()
              od_graph_def.ParseFromString(serialized_graph)
              tf.import_graph_def(od_graph_def, name='')
          return detection_graph
    def index(self):
          category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)
          return category_index
```
> The main Code 

```python

import cv2
from imutils.video import VideoStream
from main import Main
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from distutils.version import StrictVersion
import datetime
from io import StringIO

from PIL import Image


main = Main()
detection_graph = main.graph()
category_index = main.index()
print(category_index)
cap = VideoStream(0).start()
start_time = datetime.datetime.now()
num_frames = 0
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = (num_frames / elapsed_time)
            fps = str(round(fps, 2))
            cv2.putText(image_np, f"fps:{fps}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

            cv2.imshow('Face Mask Detection', cv2.resize(image_np, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

cap.release()

```

### Note : Clone this repo and create env and install requirements.txt to follow these steps


## Author

👤 **Manikant Kumar**

 
* Github: [@devil-cyber](https://github.com/devil-cyber)
* LinkedIn: [@manikant-kumar-550998192](https://linkedin.com/in/manikant-kumar-550998192)

## Show your support

Give a ⭐️ if this project helped you!


***
 
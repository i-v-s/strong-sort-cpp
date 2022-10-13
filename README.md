strong-sort-cpp
===============

This is the C++ implementation of the paper "StrongSORT: Make DeepSORT Great Again". Code is reproduced from repository https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet

Known restrictions
------------------

 - ReId model is not included
 - ECC is not implemented

Installation
------------

Install C++ requirements:
 - `apt install libeigen3-dev libopencv-dev gcc`

Install Python bindings:

 - `python setup.py install`

Using example 
---------

```python
from torch import hub
from torchreid.utils import FeatureExtractor
from strong_sort_cpp import StrongSort

ss = StrongSort()
yolo = hub.load('ultralytics/yolov5', 'yolov5m')
reid = FeatureExtractor('osnet_ain_x1_0', ...)

for image in source:
    pred = yolo(image).pred[0].cpu().numpy()
    
    ltwhs = pred[:, :4]
    confs = pred[:, 4]
    classes = pred[:, 5]

    rois = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in ltwhs.astype(int)]
    features = reid(rois).detach().cpu().numpy()
    tracks = ss.update(ltwhs, confs, classes, features, (w, h))
    ...
```

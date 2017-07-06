SSD Implementation 
==============

# Overview

This is an implementation of [Single Shot MultiBox Detector (SSD)]()
paper for pedestrian detection. This is still very much a work in progress. 

# Datasets

The code supports the VOC2012 dataset and the Caltech Pedestrian
dataset. 
The code has been written so you can add  support for your own
datasets as well and mix & match as necessary. 

## Requirements

- OpenCV 3.0+ (with Python binding)
- Python 2.7+, 3.4+, 3.5+
- NumPy 1.10+
- SciPy 0.16+

## Training
To train the system you can


## Average size and shape of pedestrian
Getting the system to converge on a 

# Caltech Pedestrian Dataset

```
$ bash shells/download.sh
$ python scripts/convert_annotations.py
$ python scripts/convert_seqs.py
```

Each `.seq` movie is separated into `.png` images. Each image's filename is consisted of `{set**}_{V***}_{frame_num}.png`. According to [the official site](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/), `set06`~`set10` are for test dataset, while the rest are for training dataset.

(Number of objects: 346621)

# Draw Bounding Boxes

```
$ python tests/test_plot_annotations.py
```

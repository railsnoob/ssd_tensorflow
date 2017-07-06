SSD Implementation 
==============

# Overview

This is an implementation of [Single Shot MultiBox Detector (SSD)](https://arxiv.org/abs/1512.02325)
paper for pedestrian detection. *This is still very much a work in progress.*

# Datasets

The code supports the VOC2012 dataset and the Caltech Pedestrian
dataset. 
The code has been written so you can add  support for your own
datasets as well and mix & match as necessary. 

## Requirements

- OpenCV 3.0+ (with Python binding)
- Python 3.4+, 3.5+
- NumPy 1.10+
- SciPy 0.16+

# Steps To Run Your Own Experiments

## 1. Create a directory to hold data & results
```
$ mkdir voc_vgg16_1000
```

## 2. Create configuration file
Create a yaml file containing all hyper-parameters and meta-data for the experiment you want to run. Use the template yaml file given in the home directory. You can specify nets, datasets and hyper-parameters.
```
$cp ssd_config.tmpl.yaml voc_vgg16_1000/
```

## 3. Figure out your feature map sizes
The core hyperparameter of SSD is the size of the feature maps used. To figure out feature map sizes run the following:
Make sure you add this to the yaml file. 
Once this is done, all other hyperparameters derived from this are automatically calculated. 

## 4. Create your own dataset
You can choose from the Pascal VOC2012 dataset or the Stanford Pedestrian detection dataset. Change the yaml file accordingly and run "python pre_process.py". This will create a sample dataset of whatever size you choose and pre_process all images and put them into the project directory. (eg: voc_vgg16_1000)

## 5. Training
After configuring the correct directory name run the following command.
```
$ python train.py
```
This command will use all the meta-data and hyper-parameters in <your_directory>/ssd_config.yaml and it will create a new directory containing loss and accuracy data every epoch and save the model every 5 epochs. 

## 6. Testing 


# Experimental Results So Far
I've trained the system with VGG16 using 3000 images from the Caltech Pedestrian Detection dataset. This took 2 days of running on AWS gpu.large instance. There are still a lot of false positives being created by the system. 
Training & Validation loss plots over **100 epochs** with batch_size=16 and 3000 images with at least 1 ground truth box.
![Loss After 100 EPOCHS](https://raw.githubusercontent.com/railsnoob/ssd_tensorflow/master/docs/ssd_loss.png)
The plan is to keep training for another 100 epochs.


## Sample Detection
![Sample Detection](https://raw.githubusercontent.com/railsnoob/ssd_tensorflow/master/docs/SampleDetections.png)

# Future Work
1. Further train the VGG16 model with the Stanford Pedestrian dataset for 100 more epochs.
2. Add accuracy plots for test set.
2. Add image augmentation. Based on empirical results this should increase the accuracy of the model substantially.
3. Find out the best default box sizes over both datasets.
4. Run forward net in real time.


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

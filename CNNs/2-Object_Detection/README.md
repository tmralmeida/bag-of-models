# bag-of-models-usage

# Table of Contents
- [Hardware](#hardware)
- [Image Classifcation](#image-classification)
- [Object Detection](#object-detection)
    * [Input Args](#input-args)
    * [Single GPU](#single-gpu-training)
    * [Multi GPU](#multi-gpu-training)
- [Acknowledgements](#acknowledgements)


# Hardware

The training/evaluation were performed on Nvidia RTX2080ti GPUs.

# Image Classification

This module is composed of several Jupyter notebooks with simple implementations of the  conventional image classifcation networks in Tensorflow.


# Object Detection

With regard to the Object Detection task, these implementations yield an aggregation of an organized and clean set of models under Pytorch framework. 
Those models can be trained in a single GPU or in a multi-GPU fashion depending on the input arguments. To do so, we use [Ignite](https://pytorch.org/ignite/)
and [torch.dist](#https://pytorch.org/tutorials/beginner/dist_overview.html) module.


## Input Args

| Argument Name       |       Type       |                Default               |                      Description                          |
|---------------------|:----------------:|:------------------------------------:|-----------------------------------------------------------|
| --dataset           |     `string`     | `"bdd100k"`                          |         Dataset                                           |
| --model             |     `string`     | `"faster"`                           | Model to train: faster, ssd512, yolov3, yolov3spp,yolov4  |
| --feature extractor |     `string`     | `"mobilenetv2"`                      | Feature extractors for models whose backbone is a conventional classification network        |



## Single GPU  

To train  the yolov4 model:

````
python scripts/train_yolo.py --model yolov4 --batch_size 8 --epochs 30 --imgs_rect False
````

## Multi GPU

To train the YOLOv4 model:

````

````


# Acknowledgements

Repos: [MobileNetV2 + Single Shot Multibox Detector](https://github.com/qfgaohao/pytorch-ssd) and [Ultralytics](https://github.com/ultralytics/yolov3)


Computational power: [lardemua](#https://github.com/lardemua)
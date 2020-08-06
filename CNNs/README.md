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
| --feature_extractor |     `string`     | `"mobilenetv2"`                      | Feature extractors for models whose backbone is a conventional classification network        |
| --pretrained        |     `bool`       | `True`                               | Pretrained backbones on ImageNet and COCO datasets for Faster R-CNN |
| --batch_size        |     `int`       | `4`                                   | Batch size |
| --epochs            |     `int`       | `10`                                  | Number of epochs |
| --learning_rate     |     `int`       | `1e-3`                                | Learning rate value |
| --weight_decay      |     `int`       | `1e-4`                                | Weight decay value |
| --workers           |     `int`       | `8`                                   | Number of subprocesses to use for data loading |
| --distributed       |      ---        |  ---                                  | For distributed training. True once used. |
| --state_dict        |     `string`    | `""`                               | State dict path for models evaluation |
| --imgs_rect         |     `bool`      | `True`                                   | False if mosaic augmentation on YOLOv4 |




## Single GPU  

To train  yolov4 model:

````
python scripts/train_yolo.py --model yolov4 --batch_size 8 --epochs 30 --imgs_rect False
````

To evaluate the same model:

````
python scripts/eval_yolo.py --model yolov4 --batch_size 32 --state_dict <model.pt>
````

## Multi GPU

To train the YOLOv4 model:

````

````


# Acknowledgements

Repos: [MobileNetV2 + Single Shot Multibox Detector](https://github.com/qfgaohao/pytorch-ssd) and [Ultralytics](https://github.com/ultralytics/yolov3)


Computational power: [lardemua](#https://github.com/lardemua)
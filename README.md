# bag-of-models
This is a guide for Deep Learning practitioners. It covers Tensorflow and Pytorch techniques to train the best-known models for a image classification and object detection fields.


# Table of Contents
- [Overview](#overview)
- [Note](#note)
- [Table of Contents](#table-of-contents)
- [CNNs](#setup)
  * [Image Classification](#image-classification) 
    * [AlexNet](#alexnet)
    * [ZFNet](#zfnet)
    * [VGG16](#vgg16)
    * [ResNet18](#resnet18)
    * [GoogLeNet](#googlenet)
    * [Xception](#xception)
    * [MobileNet](#mobilenet)
  * [Object Detection](#object-detection)
    * [Faster R-CNN](#faster-rcnn)
    * [SSD](#ssd)
    * [YoloV3](#yolov3)
- [Acknowledgements](#acknowledgements)

# Overview

This set of models was trained and implemented to help Deep Learning beginners starting their journey on Deep Learning in practice. This was developed by a practitioner who throughout this project has been developing more skills to improve the quality of de code as well as the usage of new and better tools. Thus, the first set of models, [Image Classification](#image-classification), was developed in [TensorFlow](https://github.com/tensorflow/tensorflow), and shows some simple ways to address the problem. The remaining models and the respective training/evaluations were deployed under [PyTorch](https://github.com/pytorch/pytorch) and show the usage of new libraries such as [Albumentations](https://github.com/albumentations-team/albumentations) and [Ignite](https://pytorch.org/ignite/).



# Image Classification

Image classifcation is the Computer Vision branch used to classify an image based on its content. On Deep Learning, we use a set of convolutional layers that reduce progressively the image size by applying filters. Those filters allow, throughout the layers, the construction of more complex image features, which on the final stage of each architecture are used to predict the class of the image. Here, I used the [CINIC10](https://github.com/BayesWatch/cinic-10) dataset in order to perform the study of the best-known models that address this problem. 

Thus, this set of models was developed through the [Jupyter](https://github.com/jupyter/notebook) environment. During the notebooks, I tried to use differents TensorFlow methods of how to deploy a model; however, on preprocessing data procedures I used always [tf.data](https://www.tensorflow.org/guide/data) TensorFlow API.

# Object Detection

Object detection is the task of locating an object in an image and then assingning it a class. From now on, all the methodologies are developing under PyTorch framework.
The problem that I address here is the detection of road objects. Therefore, the models are trained on the most diversified/complex Dataset for this task: [BDD100K](http://bair.berkeley.edu/blog/2018/05/30/bdd/). The training set is composed of 70K images and the validation set of 10k images. Those images are annotated taking into account 10 different classes: bus, light, sign, person, bike, truck, motor, car, train and rider. Finally, for this task we have a particular [README](https://github.com/tmralmeida/bag-of-models/tree/master/CNNs) in order to be clearer to you to see some of the interesting techniques that I applied.

# Note :warning:

I am learning Deep Learning as possibly you are and I believe that the best way to learn is to do it! So, please try not to use the git clone command, this project is for you to take ideas, see how certain things are done as I did from other resources. The initial models you can even copy just to see the results, but from there, develop your own methods through the study of these notebooks/scripts presented here.  This is not a model library, this is a library of methodologies to deploy, train and evaluate models. If you are looking for a library of models, GitHub is plenty of them out there. 

# Acknowledgements

Repos: [MobileNetV2 + Single Shot Multibox Detector](https://github.com/qfgaohao/pytorch-ssd) and [Ultralytics](https://github.com/ultralytics/yolov3)


Computational power: [Lardemua](https://github.com/lardemua)

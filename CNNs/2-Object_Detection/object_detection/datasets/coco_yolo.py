import torch
import cv2 
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pickle



PATH_IMG_INFO_TRAIN = "/srv/datasets/coco/annotations/train_imgs.txt"
PATH_ANNS_INFO_TRAIN = "/srv/datasets/coco/annotations/anns_info_train.json"
PATH_IMAGES_TRAIN = "/srv/datasets/coco/images/train2017"

PATH_IMG_INFO_VAL = "/srv/datasets/coco/annotations/test_imgs.txt"
PATH_ANNS_INFO_VAL = "/srv/datasets/coco/annotations/anns_info_val.json"
PATH_IMAGES_VAL = "/srv/datasets/coco/images/val2017"


class COCODetection(object):  # for training/testing
    """COCO dataset for YOLO 
    Keyword arguments:
    - transforms: transformations to be applied based on
    - target transforms: for the SSD model
    albumentations library
    - mode (train or val)     
    """
    def __init__(self, transforms = None, target_transform = None, mode = "train"):
        self.transforms = transforms 
        self.target_transform = target_transform
        self.mode = mode
        
        if self.mode == "train" or self.mode == "val":
            if self.mode=="train":
                img_info_file = PATH_IMG_INFO_TRAIN
                anns_info_file = PATH_ANNS_INFO_TRAIN
                imgs_path = PATH_IMAGES_TRAIN
            else:
                img_info_file = PATH_IMG_INFO_VAL
                anns_info_file = PATH_ANNS_INFO_VAL
                imgs_path = PATH_IMAGES_VAL

            with open(img_info_file, "rb") as fp:   # Unpickling
                imgs_data = pickle.load(fp)
            with open(anns_info_file) as json_file:
                anns_data = json.load(json_file)
                
            self.imgs_data = imgs_data
            self.anns_data = anns_data
            self.imgs_path = imgs_path
        else:
            raise Exception("Oops. There are only two modes: train and val!")



    
import cv2
import os
import json
import numpy as np
import torch


PATH_ANNS_INFO_TRAIN = "/srv/datasets/coco/annotations/instances_train2017.json"
PATH_IMAGES_TRAIN = "/srv/datasets/coco/images/train2017"

PATH_ANNS_INFO_VAL = "/srv/datasets/coco/annotations/instances_val2017.json"
PATH_IMAGES_VAL = "/srv/datasets/coco/images/val2017"

class COCODetection(object):
    """COCO dataset for object detection 
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
                anns_info_file = PATH_ANNS_INFO_TRAIN
                imgs_path = PATH_IMAGES_TRAIN
            else:
                anns_info_file = PATH_ANNS_INFO_VAL
                imgs_path = PATH_IMAGES_VAL

            with open(anns_info_file) as json_file:
                anns_data = json.load(json_file)
                
            self.anns_data = anns_data
            self.imgs_path = imgs_path

        else:
            raise Exception("Oops. There are only two modes: train and val!")
        
    
    def __getitem__(self, idx):
        image_info = self.anns_data["images"][idx]
        filename = image_info["file_name"]
        image_id = image_info["id"]
        anns_img = [ann for ann in anns_data["annotations"] if ann["image_id"] == image_id]

        img = cv2.imread(os.path.join(self.imgs_path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width, height = img.shape[1], img.shape[0]
        img = np.array(img)
        
        ann_boxes = [ann["bbox"] for ann in anns_img]  #[x, y, w, h]
        labels = [ann["category_id"] for ann in anns_img]
        areas = [ann["area"] for ann in anns_img]
        iscrowd = [ann["iscrowd"] for ann in anns_img]
        
        boxes_xywh = np.array(ann_boxes)
        # boxes to [xmin, y_min, x_max, y_max]
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:,0] = boxes_xywh[:,0]
        boxes_xyxy[:,1] = boxes_xywh[:,1]
        boxes_xyxy[:,2] = boxes_xywh[:,0] +  boxes_xywh[:,2]
        boxes_xyxy[:,3] = boxes_xywh[:,1] + boxes_xywh[:,3]
        
        boxes_ssd = np.array(boxes_xyxy, dtype = np.float32)
        labels_ssd = np.array(labels, dtype = np.int64)
        boxes = torch.as_tensor(boxes_xyxy, dtype = torch.float32)
        labels = torch.as_tensor(labels, dtype = torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor(image_id)
        target["area"] = torch.as_tensor(areas,dtype = torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype = torch.int64)
        
        sample = {
            "image":img,
            "bboxes":boxes,
            "labels":labels
        }
        
        if (self.transforms is not None) and (self.target_transform is None):
            augmented = self.transforms(**sample)
            img = augmented['image']
            target['boxes'] = torch.as_tensor(augmented['bboxes'],  dtype = torch.float32)
            target['labels'] = torch.as_tensor(augmented['labels'], dtype = torch.int64)
            
        elif (self.transforms is not None) and (self.target_transform is not None): #SSD
            img, boxes, labels = self.transforms(img, boxes_ssd, labels_ssd)
            target['boxes'] = boxes
            target['labels'] = labels
            boxes, labels = self.target_transform(target['boxes'], target['labels'])
            return img, boxes, labels 
        
        return img, target
    
        
    def __len__(self):
        return len(self.imgs_data)
    
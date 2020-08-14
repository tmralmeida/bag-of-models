import cv2
import matplotlib.pyplot as plt
import numpy as np


class VisualizationImg:
    """Visualization of img data and the respectives bounding boxes 
    Keyword arguments:
    - tuple (img, target). Both are torch tensors and the target has to have two fields: boxes
    (bounding boxes in the format xmin, ymin, xmax, ymax) adn the respective labels
    Example of usage for the Faster RCNN data loader:
    from datasets.bdd_dataset import BDD100kDataset
    from utils.visualization import VisualizationImg
    ds = COCODetection()
    img, targets = ds[1]
    vis = VisualizationImg(np.transpose(img, (2,0,1)), targets)
    img = vis.get_img(ds = "coco")
    plt.imshow(img);  
    """
    def __init__(self, img,target, eval_frames = False, thresh = 0.5):
        self.img = img
        self.target = target
        self.eval_frames = eval_frames
        self.threshold = thresh

    def visualize_bbox(self, img, bbox, label, ds, model=None, thickness=4):
        if ds == "bdd":
            if model == "yolo":
                class_name = LBLS_MAP_BDD_YOLO[label]
            else:
                class_name = LBLS_MAP_BDD_CONV[label]
            colormap = COLORMAP_BDD[class_name]
        elif ds == "coco":
            class_name = LBLS_MAP_COCO[label - 1] 
            colormap = 255
            
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), 255, thickness) #top left and bottom right
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.rectangle(img, (int(x_min), int(y_min) - int(1.3 * text_height)), (int(x_min) + text_width, int(y_min)), colormap, -1)
        cv2.putText(img, class_name, (int(x_min), int(y_min) - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,TEXT_COLOR, lineType=cv2.LINE_AA)
        return img

    def get_img(self, ds = "bdd"):
        img_final = np.array(self.img, dtype = np.uint8).transpose(1,2,0)
        target = self.target
        boxes = []
        labels = []
        for i in range(len(target['boxes'])):
            if self.eval_frames == True and target['scores'][i]>self.threshold:
                boxes.append(list(np.array(target['boxes'][i])))
                labels.append(int(target['labels'][i]))
            elif self.eval_frames == False:
                boxes.append(list(np.array(target['boxes'][i])))
                labels.append(int(target['labels'][i]))
        for i in range(len(boxes)):
            img_final = self.visualize_bbox(img_final, boxes[i], labels[i], ds)
        return img_final


# Faster and SSD
LBLS_MAP_BDD_CONV = {	
    0 : 'BACKGROUND',	
    1 : 'bus',	
    2 : 'traffic light',	
    3 : 'traffic sign',	
    4 : 'person',	
    5 : 'bike',	
    6 : 'truck',	
    7 : 'motor',	
    8 : 'car',	
    9 : 'train',	
    10 : 'rider'    	
}   


# Yolo:	
LBLS_MAP_BDD_YOLO =  {
    0 : 'bus',	   
    1 : 'traffic light',	  
    2 : 'traffic sign',	
    3 : 'person',	
    4 : 'bike',	
    5 : 'truck',	
    6 : 'motor',	
    7 : 'car',	
    8 : 'train',	
    9 : 'rider'    	
} 


COLORMAP_BDD = {	
    'BACKGROUND' : (30,30,30),	   
    'bus' : (128,128,128),	    
    'traffic light': (0,0,0),	
    'traffic sign': (0,255,255),	
    'person': (255,0,0),	
    'bike':(200,0,127),	
    'truck':(100,125,125),	
    'motor':(10,10,10),	
    'car':(0,0,255),	
    'train':(200,200,200),	
    'rider':(120,0,120)    	
}


LBLS_MAP_COCO = {
    0 : "person",
    1  : "bicycle",
    2  : "car",
    3  : "motorcycle",
    4  : "airplane",
    5  : "bus",
    6  : "train",
    7  : "truck",
    8  : "boat",
    9  : "traffic light",
    10 : "fire hydrant",
    12 : "stop sign",
    13 : "parking meter",
    14 : "bench",
    15 : "bird",
    16 : "cat",
    17 : "dog",
    18 : "horse",
    19 : "sheep",
    20 : "cow",
    21 : "elephant",
    22 : "bear",
    23 : "zebra",
    24 : "giraffe",
    26 : "backpack",
    27 : "umbrella",
    30 : "handbag",
    31 : "tie",
    32 : "suitcase",
    33 : "frisbee",
    34 : "skis",
    35 : "snowboard",
    36 : "sports ball",
    37 : "kite",
    38 : "baseball bat",
    39 : "baseball glove",
    40 : "skateboard",
    41 : "surfboard",
    42 : "tennis racket",
    43 : "bottle",
    45 : "wine glass",
    46 : "cup",
    47 : "fork",
    48 : "knife",
    49 : "spoon",
    50 : "bowl",
    51 : "banana",
    52 : "apple",
    53 : "sandwich",
    54 : "orange",
    55 : "broccoli",
    56 : "carrot",
    57 : "hot dog",
    58 : "pizza",
    59 : "donut",
    60 : "cake",
    61 : "chair",
    62 : "couch",
    63 : "potted plant",
    64 : "bed",
    66 : "dining table",
    69 : "toilet",
    71 : "tv",
    72 : "laptop",
    73 : "mouse",
    74 : "remote",
    75 : "keyboard",
    76 : "cell phone",
    77 : "microwave",
    78 : "oven",
    79 : "toaster",
    80 : "sink",
    81 : "refrigerator",
    83 : "book",
    84 : "clock",
    85 : "vase",
    86 : "scissors",
    87 : "teddy bear",
    88  : "hair drier",
    89 : "toothbrush"
}

TEXT_COLOR = (255, 255, 255)
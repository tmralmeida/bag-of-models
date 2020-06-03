import cv2
import matplotlib.pyplot as plt
import numpy as np

# LBLS_MAP = {
#     0 : 'BACKGROUND',
#     1 : 'bus',
#     2 : 'traffic light',
#     3 : 'traffic sign',
#     4 : 'person',
#     5 : 'bike',
#     6 : 'truck',
#     7 : 'motor',
#     8 : 'car',
#     9 : 'train',
#     10 : 'rider'    
# }   

# Yolo:
LBLS_MAP = {
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
        
COLORMAP = {
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

TEXT_COLOR = (255, 255, 255)


class VisualizationImg:
    """Visualization of img data and the respectives bounding boxes 
    Keyword arguments:
    - tuple (img, target). Both are torch tensors and the target has to have two fields: boxes
    (bounding boxes in the format xmin, ymin, xmax, ymax) adn the respective labels
    Example of usage for the Faster RCNN data loader:
    from datasets.bdd_dataset import BDD100kDataset
    from utils.visualization import VisualizationImg
    ds = BDD100kDataset()
    example = ds[1]
    vis = VisualizationImg(*example)
    img = vis.get_img()
    plt.imshow(img);  
    """
    def __init__(self, img,target, eval_frames = False, thresh = 0.5):
        self.img = img
        self.target = target
        self.eval_frames = eval_frames
        self.threshold = thresh

    def visualize_bbox(self, img, bbox, label, thickness=2):
        x_min, y_min, x_max, y_max = bbox
        class_name = LBLS_MAP[label]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORMAP[class_name], thickness) #top left and bottom right
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.rectangle(img, (int(x_min), int(y_min) - int(1.3 * text_height)), (int(x_min) + text_width, int(y_min)), COLORMAP[class_name], -1)
        cv2.putText(img, class_name, (int(x_min), int(y_min) - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,TEXT_COLOR, lineType=cv2.LINE_AA)
        return img

    def get_img(self):
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
            img_final = self.visualize_bbox(img_final, boxes[i], labels[i])
        return img_final

        
def log_results(num_images, loader_size, print_freq, loss_dict, losses, lr):
    """Prints the results
    Keyword arguments:
    - number of images already passed through the DNN
    - size of the loader
    - step of iterations to print the results
    - loss_dict: dictionary with all the losses
    - losses: total loss
    - lr: last learning rate
    """
    print('[{}/{}]'.format(str(num_images),str(loader_size)),
        'total_loss:', round(losses.item(),4),
        'loss_classifier:', round(loss_dict['loss_classifier'].item(),4), 
        'loss_box_reg:', round(loss_dict['loss_box_reg'].item(),4),
        'loss_objectness:', round(loss_dict['loss_objectness'].item(),4),
        'loss_rpn_box_reg:', round(loss_dict['loss_rpn_box_reg'].item(),4),
        'lr:', lr)

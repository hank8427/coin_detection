import torch
from xml.dom.minidom import parse
import os
import numpy as np
from PIL import Image

class P_dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.img = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.bbox = list(sorted(os.listdir(os.path.join(root, 'Annotations'))))
        
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.img[idx])
        img = Image.open(img_path).convert("RGB")
        bbox_path = os.path.join(self.root, 'Annotations', self.bbox[idx] )
        
        dom = parse(bbox_path)
        data = dom.documentElement
        objs = data.getElementsByTagName('object')
        
        classes = {'1 dollar':1, '5 dollar':2, '10 dollar':3, '50 dollar':4}
        boxes = []
        labels = []
        for obj in objs:
            name = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
            label = classes[name]
            bbox = obj.getElementsByTagName( 'bndbox' )[ 0 ]
            xmin = np.float64(bbox.getElementsByTagName( 'xmin' )[ 0 ].childNodes[ 0 ].nodeValue)
            ymin = np.float64(bbox.getElementsByTagName( 'ymin' )[ 0 ].childNodes[ 0 ].nodeValue)
            xmax = np.float64(bbox.getElementsByTagName( 'xmax' )[ 0 ].childNodes[ 0 ].nodeValue)
            ymax = np.float64(bbox.getElementsByTagName( 'ymax' )[ 0 ].childNodes[ 0 ].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        iscrowd = torch.zeros((len(objs),), dtype = torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["iscrowd"] = iscrowd
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms != None:
            img, target = self.transforms(img, target)
        return img, target

import utils
import transforms as T

def get_transform(train):
    transforms=[]
    transforms.append(T.PILToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        
    return T.Compose(transforms)

from engine import train_one_epoch, evaluate
import utils

root = "C:/Users/user/Desktop/coin_detection"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = P_dataset(root, get_transform(train=True))
dataset_test = P_dataset(root, get_transform(train=False))
# train_data, test_data = train_test_split(dataset, random_state=1, train_size=0.8)
indices = torch.randperm(len(dataset)).tolist()
train_data = torch.utils.data.Subset(dataset, indices[:-14])
test_data = torch.utils.data.Subset(dataset_test, indices[-14:])

data_loader = torch.utils.data.DataLoader(train_data, batch_size= 2, shuffle= True, collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(test_data, batch_size= 2, shuffle= False, collate_fn=utils.collate_fn)

import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load model

# model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# num_class = 5
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class)
# model.to(device)
model = torch.load('D:/Py_file/PennFudanPed/model.pkl')

# Set model to eval mode
model.eval()

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr= 0.0001 ,
                            momentum= 0.9 , weight_decay= 0.0002 )

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0= 1 , T_mult= 2 )
num_epochs = 10

for epoch in  range (num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)
    
    print('')
    print('==================================================')
    print('')
    
print("That's it!")

torch.save(model, 'D:/Py_file/PennFudanPed/model_v3.pkl')
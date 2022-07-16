import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

model = torch.load('D:/Py_file/PennFudanPed/model.pkl')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def showbbox(model, img):
    # 输入的img是0-1范围的tensor        
    model.eval()
    with torch.no_grad():
        '''
        prediction形如：
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        prediction = model([img.to(device)])
        
    print(prediction)
        
    img = img.permute(1,2,0)  # C,H,W → H,W,C，用来画图
    img = (img * 255).byte().data.cpu()  # * 255，float转0-255
    img = np.array(img)  # tensor → ndarray
    img = np.ascontiguousarray(img)
    
    t = prediction[0]['scores'] >= 0.5
    index = [i for i, x in enumerate(t) if x]
    
    for i in index:
        xmin = int(round(prediction[0]['boxes'][i][0].item()))
        ymin = int(round(prediction[0]['boxes'][i][1].item()))
        xmax = int(round(prediction[0]['boxes'][i][2].item()))
        ymax = int(round(prediction[0]['boxes'][i][3].item()))
        
        label = prediction[0]['labels'][i].item()
        
        if label == 1:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
            cv2.putText(img, '1 dollar', (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                               thickness=2)
        elif label == 2:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
            cv2.putText(img, '5 dollar', (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                               thickness=2)
        elif label == 3:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            cv2.putText(img, '10 dollar', (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                               thickness=2)
        elif label == 4:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), thickness=2)
            cv2.putText(img, '50 dollar', (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                               thickness=2)
            
    plt.figure(figsize=(15,8))
    plt.imshow(img)

img_path = "C:/Users/user/Desktop/coin_detection/images/IMAG0093.jpg"
img = Image.open(img_path).convert("RGB")
converter = transforms.ToTensor()
img = converter(img)
showbbox(model, img)
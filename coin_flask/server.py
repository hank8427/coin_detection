from io import BytesIO
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import base64
import torch.nn.functional as F
from torchvision import models
import flask
from argparse import ArgumentParser
import cv2
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
app = flask.Flask(__name__)


model = torch.load('D:/Py_file/coin_flask/model/model_v3.pkl')


global classes
classes = ['1 dollar', '5 dollar', '10 dollar', '50 dollar']




@app.route("/predict/", methods=["POST"])

def predict():
    output_dict = {}
    if flask.request.method == "POST":
        response = flask.request.get_json()
        
        data_str = response["image"]
        points = data_str.find(',')
        base64str = data_str[points:]
        image = base64.b64decode(base64str)

        image = Image.open(BytesIO(image))
        image = np.array(image)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255
        img = np.rollaxis(img, 2)
        image_tenor = torch.FloatTensor(img).unsqueeze(0)

        with torch.no_grad():
            results = model(image_tenor)
        '''
        results形如：
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        t = results[0]['scores'] >= 0.5
        index = [i for i, x in enumerate(t) if x]

        labels = []
        scores = []
        boxes = []
        for i in index:
            labels.append(classes[np.array(results[0]['labels'][i]).tolist()])
            scores.append(np.array(results[0]['scores'][i]).tolist())
            boxes.append(np.array(results[0]['boxes'][i]).tolist())

        print(labels)
        print(classes)
        output_dict['name'] = labels[0]
        output_dict['scores'] = scores[0]
        output_dict['boxes'] = boxes[0]
        output_dict = [output_dict]
        print(output_dict)

    return flask.jsonify(output_dict), 200

@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--port', help='port', type=int, default=5000)
    args = parser.parse_args()

    app.run(host="127.0.0.1", debug=True, port=args.port)
    

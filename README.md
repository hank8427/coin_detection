# NT dollar Detection Practice
目前可檢測新台幣的1元、5元及10元

## 方法
- [PyTorch](https://pytorch.org/)

- [pycocotools](https://pypi.org/project/pycocotools/)

- 使用Faster-RCNN ResNet-50 FPN進行transfer learning

- Dataset: 70 images(720x720)


## Test

![image](https://github.com/hank8427/coin_detection/blob/main/coin_detection_test03.PNG)

## Future Work

Add the pictures of the reverse side of the coin and the 50 yuan to improve the recognition rate!

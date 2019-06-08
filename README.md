# README
### this repo is a YOLO v3 Simple Implementation by Pytorch (still debuging have some bug to fix)
* backbones will support: `resnet18`, `resnet50`, `densenet121` and other backbones
* finished: `backbone with resnet18`, `FPN module`, `YOLO layer`, `Dataloader`, `Loss`, `train`
* visual by `visdom`
* still debuging, need helps to debug together

### bugs
* 1. can't train on big datasets, because of bad performance on big datasets.
* 2. on little datasets like 1~8 picture could have expected result(but not every time can success), and every time trained results is different, need to find reasons.
* 3. multi-GPU will hang! I have not find the reason but I guess maybe some modules not registered success, but single GPU could run.

##### run train
* change Hyperparameters and data path in `configs/resnet18_yolo_style_fpn_yolov3.py`
* prepare data: same like origin darknet yolo do
* run `nohup python -m visdom.server &` to start visdom server
* in `modules/YOLOv3.py` add `sys.path.insert(0, '/path/to/yolov3/modules')`
* `python train.py`
* if you wanna try little datasets, `yoloDataset(xxx,... little_train=8)` mean only use first 8 images to train

##### some visual results
* FCN32-resnet18
  * ![detect_results](readme/yolo0.png)
  * ![detect_results](readme/yolo1.png)
  * ![detect_results](readme/yolo2.png)
  * ![detect_results](readme/yolo3.png)
  * ![detect_results](readme/yolo4.png)
  * ![detect_results](readme/yolo5.png)


##### TODO
* fix bugs 
* add backbones: `resnet50`, `densenet121` and so on 
* add eval scripts
* add test mAP(maybe use cocoapi)
* find better train params
* improve performance

##### Requirements
* python3
* pytorch 0.4.0
* visdom
* torchsummary
* torchvision
* cv2
* PIL
* numpy
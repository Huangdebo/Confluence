#  Confluence: A Robust Non-IoU Alternative to Non-Maxima Suppression in Object Detection 

## 1. Introduction

在 yolov4 的基础上添加车道线检测分支，实现同时进行普通物体检测和车道线检测

## 2. Requirements

- pytorch >= 1.2.0 (lower versions may work too)
- opencv-python


## 3. Running demos

###（1）pytorch

pytorch 和 caffe 模型链接：

链接：https://pan.baidu.com/s/10UXd7DAr7QWSs8OfZhRGTA 
提取码：yolo 

视频测试:

```shell
python detect_video.py --video-path=./data/samples/video.avi
```

效果 demo 在 ./data/samples 中

###（2）caffe

测试用 caffe 的 yolov3 即可，lane 的解析参考 detect_video.py 

## 4. Code 

该版本的 yolov4 是使用 cfg 文件来构建网络，所以比较方便添加自己的分支，只需要修改 .cfg 和 添加自定义模块的定义即可：

###（1）cfg
```shell
# lane

[route]
layers = 48

.....

[lane]
num_lanes=4
cls_dim=101, 10, 4
use_aux=0
```

其中的 ......是车道线检测的分支网络，[lane] 便是我们定义地的模块，有三个参数，num_lanes 是最多能检测车道线的个数

###（2）cfg 文件解析

```shell
# Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'num_lanes', 'cls_dim', 'use_aux']
```

需要把新定义[lane]模块中的三个参数添加进 supported 字段中

###（3）自定义模块的实现

```shell
class LaneLayer(nn.Module):
    def __init__(self, num_lanes=4, cls_dim=(100+1, 10, 4), use_aux=False):
        super(LaneLayer, self).__init__()
        self.num_lanes = num_lanes
        self.cls_dim = cls_dim
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)
        
        self.pool = torch.nn.Conv2d(256,4,1)
        
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, self.total_dim),
        )    
        
        initialize_weights(self.cls)
        
    def forward(self, x):             
        fea = self.pool(x).view(-1, 1024)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        
        return group_cls
```

###（4）模型构建

#### 在 create_modules() 函数中添加对新定义模块的解析

```shell
elif mdef['type'] == 'lane':
	modules = LaneLayer(num_lanes=mdef['num_lanes'],
						cls_dim=mdef['cls_dim'],
						use_aux=mdef['use_aux'])
```

#### 在 Darknet中的 forward_once() 中添加新定义模块

```shell
elif name == 'LaneLayer':
    lane_detect = module(x)
```

至此网络就搭建完毕


## 5. Train 

如果很多小伙伴对训练感兴趣，后期方便的话就再开源训练代码。对于多任务网络有好意见的小伙伴，可以加我qq（702864842）交流哦。

## Credits:

Yolov4 是参考 WongKinYiu 大神的版本:

https://github.com/WongKinYiu/PyTorch_YOLOv4





 
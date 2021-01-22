#  Confluence: A Robust Non-IoU Alternative to Non-Maxima Suppression in Object Detection 

## 1. 介绍

![image](https://github.com/Huangdebo/Confluence/blob/master/imges/1.png)

用以替代 NMS，在所有 bbox 中挑选出最优的集合。 NMS 仅考虑了 bbox 的得分，然后根据 IOU 来去除重叠的 bbox。而 Confluence 则是利用曼哈顿距离作为 bbox 之间的重合度，并根据置信度加权的曼哈顿距离还作为最优 bbox 的选择依据。

## 2. 算法原理

#### 2.1 曼哈顿距离

两点的曼哈顿距离就是坐标值插的 L1 范数：

![image](https://github.com/Huangdebo/Confluence/blob/master/imges/2.png)

推广到两个 bbox 对的哈曼顿距离则为：

![image](https://github.com/Huangdebo/Confluence/blob/master/imges/3.png)

该算法便是以曼哈顿距离作为两个 bbox 的重合度，曼哈顿距离小于一定值的的 bbox 则被认为是一个 cluster。

#### 2.2 归一化

因为 bbox 有个各样的 size 和 position，所以直接计算曼哈顿距离就没有可比性，没有标准的度量。所以需要对其进行归一化：

![image](https://github.com/Huangdebo/Confluence/blob/master/imges/4.png)

#### 2.3 置信度加权曼哈顿距离

NMS在去除重合 bbox 是仅考虑其置信度的高低，Condluence 则同时考虑了曼哈顿距离和置信度，构成一个置信度加权曼哈顿距离：

![image](https://github.com/Huangdebo/Confluence/blob/master/imges/5.png)

## 3. 算法实现

![image](https://github.com/Huangdebo/Confluence/blob/master/imges/6.png)

算法：

（1）针对每个类别挑出属于该类别的 bbox 集合 B

（2）遍历 B 中所有的 bbox bi，并计算 bi 和其他 boox的 曼哈顿距离 p，并归一化

    2.1 选出 p < 2 的集合，作为一个 cluster，并计算加权曼哈顿距离 wp。 
    
    2.2 在该 cluster 中挑选出最小的 wp 作为 bi 的 wp。 
    
（3）遍历完毕后，挑出 wp 最小的 bi 作为最优 bbox，添加进最终结果集合中，并将其从 B 去除

（4）把与最优 bbox 的曼哈顿距离小于阈值 MD 的的 bbox 从 B 中去除

（5）不断重复 （2） - （4），每次都选出一个最优 bbox，知道 B 为空

注意： 

（1）原文伪代码第 5 行：optimalConfuence 初始化成一个比较大的值就可以，不一定要是 Ip

（2）原文伪代码第 12 行：应该是 Proximity / si


## 4. 实验结果

![image](https://github.com/Huangdebo/Confluence/blob/master/imges/7.png)

## 5. 代码解析

#### 5.1 YOLOv3/4 的后处理
这个接口可以直接处理 YOLOv3/4 的 yolo 层的输出进行后处理
```python
confluence_process(prediction, conf_thres=0.1, wp_thres=0.6)
```


支持多标签和单标签，并把数据重组后进行 confluence/NMS 处理
```python
# Detections matrix nx6 (xyxy, conf, cls)
if multi_label:
    i, j = (x[:, 5:] > conf_thres).nonzero().t()
    x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
else:  # best class only
    conf, j = x[:, 5:].max(1, keepdim=True)
    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
```


#### 5.2  Confluence 算法

```python
confluence(prediction, class_num, wp_thres=0.6)
```

给所有目标添加上序号

```python
index = np.arange(0, len(prediction), 1).reshape(-1,1)
infos = np.concatenate((prediction, index), 1)
```

不同类别单独处理，并遍历所有的剩余目标集合 B，直到集合为空，对应上面伪代码的(1)-(2)

```python
for c in range(class_num):       
    pcs = infos[infos[:, 5] == c]             
    while (len(pcs)):                      
        n = len(pcs)       
        xs = pcs[:, [0, 2]]
        ys = pcs[:, [1, 3]]             
        ps = []        
        # 遍历 pcs，计算每一个box 和其余 box 的 p 值，然后聚类成簇，再根据 wp 挑出 best
        confluence_min = 10000
        best = None
        for i, pc in enumerate(pcs):
```

计算所有目标与其他目标的曼和顿距离 p 和加权曼哈顿距离 wp，p < 2 的目标作为一个 cluster，其中最小的 wp 作为该 cluster 的 wp

```python
index_other = [j for j in range(n) if j!= i]
x_t = xs[i]
x_t = np.tile(x_t, (n-1, 1))
x_other = xs[index_other]
x_all = np.concatenate((x_t, x_other), 1)
.
.
.
# wp
wp = p / pc[4]
wp = wp[p < 2]

if (len(wp) == 0):
    value = 0
else:
    value = wp.min()
```

选出最小的 wp，确定目标

```python
# select the bbox which has the smallest wp as the best bbox
if (value < confluence_min):
   confluence_min = value
   best = i  
```

然后把与目标的曼哈顿距离小于阈值的目标和本身都从集合 B 中去除

```python
keep.append(int(pcs[best][6])) 
if (len(ps) > 0):               
    p = ps[best]
    index_ = np.where(p < wp_thres)[0]
    index_ = [i if i < best else i +1 for i in index_]
else:
    index_ = []
    
# delect the bboxes whose Manhattan Distance is below the predefined MD
index_eff = [j for j in range(n) if (j != best and j not in index_)]            
pcs = pcs[index_eff]
```

最后继续重复遍历集合 B，直到集合为空。

**仓库里我放了一张测试照片和原始检测结果，大家可以直接用来调试 confluence 函数。**

## Credits:

https://arxiv.org/pdf/2012.00257.pdf



 

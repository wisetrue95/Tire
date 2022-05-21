# Efficient Tire Wear and Defect Detection Algorithm Based on Deep Learning
paper : https://www.koreascience.or.kr/article/JAKO202125761197586.pdf \
project : https://scienceon.kisti.re.kr/commons/util/originalView.do


## environment


## environment setting
```


```

# detectDefect
This project is tire defect detection model using [@matterport Mask-RCNN balloon.py](https://github.com/matterport/Mask_RCNN.git). \
and start by reading this [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46).


### 1. Create dataset


### 2. Download a preatrained model
You can download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).

### 3. Train
```
--command train --weights coco --dataset [path/to/dataset] --image [path/to/image]
```

### 4. Visualize
Visualize bounding box and defect masks
```
--command splash --weights [path/to/checkpoint] --dataset [path/to/dataset] --image [path/to/image]
```

![augmentations](defect.png "CAM result")


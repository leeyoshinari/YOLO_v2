# YOLO_v2

This is implementation of [YOLO v2](https://arxiv.org/pdf/1612.08242.pdf) with TensorFlow.

## Demo
![](https://github.com/leeyoshinari/YOLO_v2/blob/master/test/yolo%20v2%20demo.gif)

## Installation
1. Clone YOLO_v2 repository
	```Shell
	$ git clone https://github.com/leeyoshinari/YOLO_v2.git
    $ cd YOLO_v2
	```

2. Download Pascal VOC2007 dataset, and put the dataset into `data/Pascal_voc`.

   If you download other dataset, you also need to modify file paths.

3. Download weights file [yolo_weights](https://drive.google.com/drive/folders/13TWYuNY-XcX9EyoU87dH9XsBKuWcPHHw?usp=sharing) for COCO, and put weight file into `data/output`.

   Or you can also download my training weights file [YOLO_v2](https://drive.google.com/drive/folders/14w9JL74VZivk0iD00I3eQYL67bvNyq0N?usp=sharing) for VOC.

4. Modify configuration into `yolo/config.py`.

5. Training
	```Shell
	$ python train_val.py
	```

6. Test
	```Shell
	$ python test_val.py
	```
7. For more information to [wiki](https://github.com/leeyoshinari/YOLO_v2/wiki/YOLO_v2). 

## Darknet-19
Darknet-19 has 19 convolutional layers, it's faster than yolo_v2. If you use darknet-19, you need some modifications. It's easy to modify.

 Please download Darknet-19 weights file for VOC from [darknet-19](https://drive.google.com/open?id=1XWWecDpekQ1t2DjhizF-virWyQCTSUeF).

## Training on Your Own Dataset
To train the model on your own dataset, you should need to modify:

1. Put all the images into the `Images` folder, put all the labels into the `Labels` folder. Select a part of the image for training, write this part of the image filenames into `train.txt`, the remaining part of the image filenames written in `test.txt`. Then put the `Images`, `Labels`, `train.txt` and `test.txt` into `data/dataset`. Put weight file in `data/output`.

2. `config.py:` modify the CLASSES.

3. `train.py:` replace`from pascal_voc import Pascal_voc` with `from preprocess import Data_preprocess`, and replace `pre_data = Pascal_voc()` with `pre_data = Data_preprocess()`.

## Requirements
1. Tensorflow
2. OpenCV

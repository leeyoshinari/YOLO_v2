# YOLO_v2

This implementation of [YOLO v2](https://arxiv.org/pdf/1612.08242.pdf) with TensorFlow.

## Installation
1. Clone YOLO_tiny repository
	```Shell
	$ git clone https://github.com/leeyoshinari/YOLO_v2.git
    $ cd YOLO_v2
	```

2. Download Pascal VOC2007 dataset, and put the dataset into `data/Pascal_voc`.

   if you download other dataset, you also need to modify file paths.

3. Download weights file [yolo_weights](https://pan.baidu.com/s/1A4a2pIEGG_ERBwTN3F-jcw) for COCO, and put weight file into `data/output`.

   Or you can also download my training weights file [YOLO_v2](https://pan.baidu.com/s/1Xf-YEAHj2PJ35ImDR-Tthw) for VOC.

4. Modify configuration into `yolo/config.py`.

5. Training
	```Shell
	$ python train_val.py
	```

6. Test
	```Shell
	$ python test_val.py
	```

## Training on Your Own Dataset
To train the model on your own dataset, you should need to modefy:

1. Put all the images into the `Images` folder, put all the labels into the `Labels` folder. Select a part of the image for training, write this part of the image filename into `train.txt`, the remaining part of the image filename written in `test.txt`. Then put the `Images`, `Labels`, `train.txt` and `test.txt` into `data/dataset`. Put weight file in `data/output`.

2. `config.py` modify the CLASSES.

3. `train.py` replace`from pascal_voc import Pascal_voc` with `from preprocess import Data_preprocess`, and replace `pre_data = Pascal_voc()` with `pre_data = Data_preprocess()`.

## Requirements
1. Tensorflow
2. OpenCV

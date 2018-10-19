#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:leeyoshinari

DATA_DIR = 'data'
DATA_SET = 'data_set'
WEIGHTS_FILE = 'yolo_weights.ckpt'

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

#ANCHOR = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHOR = [0.57273, 1.87446, 3.33843, 7.88282, 9.77052, 0.677385, 2.06253, 5.47434, 3.52778, 9.16828]

GPU = ''

IMAGE_SIZE = 416    #The size of the input images

LEARN_RATE = 0.0001   #The learn_rate of training
MAX_ITER = 20000    #The max step
SUMMARY_ITER = 5    #Every 'summary_iter' step output a summary
SAVER_ITER = 50    #Every 'saver_iter' step save a weights

BOX_PRE_CELL = 5    #The number of BoundingBoxs predicted by each grid
CELL_SIZE = 13      #The size of the last layer  #(batch_size, 13, 13, ?)
BATCH_SIZE = 32     #The batch size of each training
ALPHA = 0.1

THRESHOLD = 0.3    #The threshold of the probability of the classes

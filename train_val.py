#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:leeyoshinari
#-----------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import argparse
import datetime
import time
import os
import yolo.config as cfg

from pascal_voc import Pascal_voc
from six.moves import xrange
from yolo.yolo_v2 import yolo_v2
# from yolo.darknet19 import Darknet19

class Train(object):
    def __init__(self, yolo, data):
        self.yolo = yolo
        self.data = data
        self.num_class = len(cfg.CLASSES)
        self.max_step = cfg.MAX_ITER
        self.saver_iter = cfg.SAVER_ITER
        self.summary_iter = cfg.SUMMARY_ITER
        self.initial_learn_rate = cfg.LEARN_RATE
        self.output_dir = os.path.join(cfg.DATA_DIR, 'output')
        weight_file = os.path.join(self.output_dir, cfg.WEIGHTS_FILE)

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore)
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir)

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learn_rate = tf.train.exponential_decay(self.initial_learn_rate, self.global_step, 20000, 0.1, name='learn_rate')
        # self.global_step = tf.Variable(0, trainable = False)
        # self.learn_rate = tf.train.piecewise_constant(self.global_step, [100, 190, 10000, 15500], [1e-3, 5e-3, 1e-2, 1e-3, 1e-4])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.yolo.total_loss, global_step=self.global_step)

        self.average_op = tf.train.ExponentialMovingAverage(0.999).apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.average_op)

        config = tf.ConfigProto(gpu_options=tf.GPUOptions())
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        print('Restore weights from:', weight_file)
        self.saver.restore(self.sess, weight_file)
        self.writer.add_graph(self.sess.graph)

    def train(self):
        labels_train = self.data.load_labels('train')
        labels_test = self.data.load_labels('test')

        num = 5
        initial_time = time.time()

        for step in xrange(0, self.max_step + 1):
            images, labels = self.data.next_batches(labels_train)
            feed_dict = {self.yolo.images: images, self.yolo.labels: labels}

            if step % self.summary_iter == 0:
                if step % 50 == 0:
                    summary_, loss, _ = self.sess.run([self.summary_op, self.yolo.total_loss, self.train_op], feed_dict = feed_dict)
                    sum_loss = 0

                    for i in range(num):
                        images_t, labels_t = self.data.next_batches_test(labels_test)
                        feed_dict_t = {self.yolo.images: images_t, self.yolo.labels: labels_t}
                        loss_t = self.sess.run(self.yolo.total_loss, feed_dict=feed_dict_t)
                        sum_loss += loss_t

                    log_str = ('{} Epoch: {}, Step: {}, train_Loss: {:.4f}, test_Loss: {:.4f}, Remain: {}').format(
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.data.epoch, int(step), loss, sum_loss/num, self.remain(step, initial_time))
                    print(log_str)

                    if loss < 1e4:
                        pass
                    else:
                        print('loss > 1e04')
                        break

                else:
                    summary_, _ = self.sess.run([self.summary_op, self.train_op], feed_dict = feed_dict)

                self.writer.add_summary(summary_, step)

            else:
                self.sess.run(self.train_op, feed_dict = feed_dict)

            if step % self.saver_iter == 0:
                self.saver.save(self.sess, self.output_dir + '/yolo_v2.ckpt', global_step = step)

    def remain(self, i, start):
        if i == 0:
            remain_time = 0
        else:
            remain_time = (time.time() - start) * (self.max_step - i) / i
        return str(datetime.timedelta(seconds = int(remain_time)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'yolo_v2.ckpt', type = str)  # darknet-19.ckpt
    parser.add_argument('--gpu', default = '', type = str)  # which gpu to be selected
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.weights is not None:
        cfg.WEIGHTS_FILE = args.weights

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    yolo = yolo_v2()
    # yolo = Darknet19()
    pre_data = Pascal_voc()

    train = Train(yolo, pre_data)

    print('start training ...')
    train.train()
    print('successful training.')


if __name__ == '__main__':
    main()

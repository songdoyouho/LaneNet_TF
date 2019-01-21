#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 1/15 19
# @Author  : Arthur Lai
# @File    : test_uff.py
# @IDE: Visual Studio Code

import tensorflow as tf
import uff

from lanenet_model import lanenet_merge_model

if __name__ == '__main__':
    
    sess = tf.Session()

    with sess.as_default():
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        phase_tensor = tf.constant('test', tf.string)
        net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
        binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')
        with open('convert_before.txt', 'w') as f:
            for op in tf.get_default_graph().get_operations():
                f.write(op.name + '\n')
        saver = tf.train.Saver()
        saver.restore(sess, 'Tusimple_Lanenet_Model_Weights/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000')
        print('stored!')
        g_def = tf.get_default_graph().as_graph_def()
        
        g_def_freezed = tf.graph_util.convert_variables_to_constants(sess, g_def, ['lanenet_model/pix_embedding_relu', 'lanenet_model/Softmax'])
        g_uff = uff.from_tensorflow(g_def_freezed, ['lanenet_model/pix_embedding_relu', 'lanenet_model/Softmax'],
                output_filename='convert_after.uff', text=True)
        print('done!')
        

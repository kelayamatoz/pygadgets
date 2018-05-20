import tensorflow as tf
import numpy as np

"""
Expect: Console prints a set of information indicating that the GPU on DMA is detected by tf
Should see something like: 
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8236
pciBusID 0000:00:04.0
Total memory: 11.17GiB
Free memory: 11.11GiB
2018-05-20 06:46:25.118092: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0
2018-05-20 06:46:25.118098: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   Y
2018-05-20 06:46:25.118129: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
"""

ROWS=3
COLS=3
TF_VER=tf.__version__

print("Testing Installation of Tensorflow ver ", TF_VER)

a = tf.constant(np.ones((ROWS,COLS)))
b = tf.constant(np.ones((ROWS,COLS)))

with tf.Session() as sess:
	op = tf.matmul(a, b)
	op_re = sess.run(op)


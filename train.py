# https://cuijiahua.com/blog/2018/01/dl_3.html

import tensorflow as tf
import numpy
import load

batch_size = 100
train_step = 10

ck1 = tf.Variable(tf.ones([5, 5, 3, 6]))
ck3 = tf.Variable(tf.ones([5, 5, 6, 16]))
ck5 = tf.Variable(tf.ones([5, 5, 16, 120]))

x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
x_ = tf.placeholder(dtype=tf.float32, shape=[None])

c1 = tf.nn.conv2d(input=x, filter=ck1, strides=[1, 1, 1, 1], padding='VALID')
r1 = tf.nn.relu(c1)
s2 = tf.nn.max_pool(value=r1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
c3 = tf.nn.conv2d(input=s2, filter=ck3, strides=[1, 1, 1, 1], padding='VALID')
r3 = tf.nn.relu(c3)
s4 = tf.nn.max_pool(value=r3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
c5 = tf.nn.conv2d(input=s4, filter=ck5, strides=[1, 1, 1, 1], padding='VALID')
r5 = tf.nn.relu(c5)
r5 = tf.reshape(r5, [-1, 18*18*120])
# r5 = numpy.reshape(r5, [-1, 3240])
f6 = tf.contrib.layers.fully_connected(inputs=r5, num_outputs=84)
f7 = tf.contrib.layers.fully_connected(inputs=f6, num_outputs=10)
f7 = tf.nn.softmax(f7)


xt = tf.one_hot(indices=tf.to_int32(x_), depth=10)
loss = tf.reduce_mean(tf.square(f7-xt))
train = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = load.get_train_data()
    labels = load.get_train_labels()
    for i in range(train_step):
        big = i*batch_size
        end = big+batch_size
        data_i = data[big:end, ...]
        labels_i = labels[big:end]
        c = sess.run([train, loss], feed_dict={x: data_i, x_: labels_i})
        print(c[1])

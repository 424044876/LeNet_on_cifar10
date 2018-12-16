import pickle
import numpy
import tensorflow as tf


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def reshape_to_photo(line):
    assert len(line) == 3072

    r = line[0:1024]
    r = numpy.reshape(r, [32, 32, 1])
    b = line[1024:2048]
    b = numpy.reshape(b, [32, 32, 1])
    g = line[2048:]
    g = numpy.reshape(g, [32, 32, 1])

    photo = numpy.concatenate([r, b, g], -1)
    # (32, 32, 3)
    return photo


def get_train_data():

    data = []

    for i in range(1, 6):
        batch_i = unpickle("cifar-10-batches-py/data_batch_%d" % i)[b'data']
        # numpy.shape(batch_i) = (10000, 3072)
        data.append(batch_i)

    data = numpy.concatenate(data, 0)
    final_data = numpy.ndarray(shape=[len(data), 32, 32, 3], dtype=numpy.float32)

    for i in range(len(data)):
        final_data[i] = reshape_to_photo(data[i])

    # NHWC (10000, 32, 32, 3)
    return final_data


def get_train_labels():

    labels = []

    for i in range(1, 6):
        labels_i = unpickle("cifar-10-batches-py/data_batch_%d" % i)[b'labels']
        # numpy.shape(batch_i) = (10000, 3072)
        labels += labels_i

    # labels = numpy.repeat(labels, 10, -1)
    return labels


def get_test_data():
    return unpickle('cifar-10-batches-py/test_batch')[b'data']


def get_test_labels():
    return unpickle('cifar-10-batches-py/test_batch')[b'labels']


# test

# s = numpy.shape(get_train_data())
# print(s)
# s2 = numpy.shape(get_train_labels())
# print(s2)
# t = numpy.shape(get_test_data())
# print(t)
# t2 = numpy.shape(get_test_labels())
# print(t2)

# print(numpy.shape(get_train_labels()))


"""
These classes provide datasets for training and testing of neural networks
in the context of continual learning.
"""

import abc
import numpy as np
import tensorflow as tf

class dataset:
    """
    Base class for dataset.
    """
    __metaclass__ = abc.ABCMeta

    # Constructor
    def __init__(self):
        print("Constructor of dataset class...")

    def get_train_samples(self, idx = None):
        # Return training samples based on a index
        if (idx is None):
            return self.train
        else:
            return (self.train[0][idx], self.train[1][idx])

    def get_eval_samples(self, idx = None):
        # Return training samples based on a index
        if (idx is None):
            return self.eval
        else:
            return (self.eval[0][idx], self.eval[1][idx])

    def filter(self, x, l):
        tmp = np.zeros(x.shape, dtype=np.bool)
        for i in range(x.shape[0]):
            if (x[i] in l):
                tmp[i] = True
        return tmp

    @abc.abstractmethod
    def get_dataset_name(self):
        pass

class mnist(dataset):
    """
    Class for the mnist dataset.
    """

    def __init__(self):
        # Load MNIST data
        [self.train, self.eval] = tf.keras.datasets.mnist.load_data()

    def get_dataset_name(self):
        return "MNIST"

class permutated_mnist(mnist):
    """
    Class for the mnist dataset with permutated pixels.
    """

    def __init__(self, np_seed = 0):
        # Load MNIST data
        [self.train, self.eval] = tf.keras.datasets.mnist.load_data()
        # Set seed of numpy
        np.random.seed(np_seed)
        # Create permutation order
        self.idx = np.arange(784)
        np.random.shuffle(self.idx)
        # Permutate dataset
        self.train = self.permutate(self.train)
        self.eval = self.permutate(self.eval)

    def get_dataset_name(self):
        return "Permutated MNIST"

    def permutate(self, x):
        # Permutate pixels in a order given by self.idx
        for i in range(x[0].shape[0]):
            x[0][i] = np.reshape(np.reshape(x[0][i], 784)[self.idx], (28, 28))
        return x

class split_mnist(mnist):
    """
    Class for the mnist dataset with task to classify two digits.
    """

    def __init__(self, class_a = [0], class_b = [1]):
        # Load MNIST data
        [train, eva] = tf.keras.datasets.mnist.load_data()
        # Get samples and labels of both classes
        train_a = (train[0][self.filter(train[1], class_a)], train[1][self.filter(train[1], class_a)])
        train_b = (train[0][self.filter(train[1], class_b)], train[1][self.filter(train[1], class_b)])
        eval_a = (eva[0][self.filter(eva[1], class_a)], eva[1][self.filter(eva[1], class_a)])
        eval_b = (eva[0][self.filter(eva[1], class_b)], eva[1][self.filter(eva[1], class_b)])
        # Stack both classes ontop of each other
        self.train = (np.concatenate((train_a[0], train_b[0]), axis=0), np.concatenate((train_a[1], train_b[1]), axis=0))
        self.eval = (np.concatenate((eval_a[0], eval_b[0]), axis=0), np.concatenate((eval_a[1], eval_b[1]), axis=0))
        # Shuffle data
        train_idx = np.arange(self.train[0].shape[0])
        eval_idx = np.arange(self.eval[0].shape[0])
        np.random.shuffle(train_idx)
        np.random.shuffle(eval_idx)
        self.train = (self.train[0][train_idx], self.train[1][train_idx])
        self.eval = (self.eval[0][eval_idx], self.eval[1][eval_idx])

    def get_dataset_name(self):
        return "Split MNIST"

class fashion_mnist(mnist):
    """
    Class for the fashion mnist dataset.
    """
    def __init__(self):
        # Load fashion mnist data
        [self.train, self.eval] = tf.keras.datasets.fashion_mnist.load_data()
    def get_dataset_name(self):
        return "Fashion MNIST"

class permutated_fashion_mnist(fashion_mnist):
    """
    Class for the fashion mnist dataset with permutated pixels.
    """

    def __init__(self, np_seed = 0):
        # Load fashion mnist data
        [self.train, self.eval] = tf.keras.datasets.fashion_mnist.load_data()
        # Change permission for the data to be writeable
        self.train[0].flags.writeable = True
        self.eval[0].flags.writeable = True
        # Set seed of numpy
        np.random.seed(np_seed)
        # Create permutation order
        self.idx = np.arange(784)
        np.random.shuffle(self.idx)
        # Permutate dataset
        self.train = self.permutate(self.train)
        self.eval = self.permutate(self.eval)

    def get_dataset_name(self):
        return "Permutated Fashion MNIST"

    def permutate(self, x):
        # Permutate pixels in a order given by self.idx
        for i in range(x[0].shape[0]):
            x[0][i] = np.reshape(np.reshape(x[0][i], 784)[self.idx], (28, 28))
        return x

class split_fashion_mnist(fashion_mnist):
    """
    Class for the fashion mnist dataset with task to classify two classes.
    """

    def __init__(self, class_a = [0], class_b = [1]):
        # Load fashion mnist data
        [train, eva] = tf.keras.datasets.fashion_mnist.load_data()
        # Get samples and labels of both classes
        train_a = (train[0][self.filter(train[1], class_a)], train[1][self.filter(train[1], class_a)])
        train_b = (train[0][self.filter(train[1], class_b)], train[1][self.filter(train[1], class_b)])
        eval_a = (eva[0][self.filter(eva[1], class_a)], eva[1][self.filter(eva[1], class_a)])
        eval_b = (eva[0][self.filter(eva[1], class_b)], eva[1][self.filter(eva[1], class_b)])
        # Stack both classes ontop of each other
        self.train = (np.concatenate((train_a[0], train_b[0]), axis=0), np.concatenate((train_a[1], train_b[1]), axis=0))
        self.eval = (np.concatenate((eval_a[0], eval_b[0]), axis=0), np.concatenate((eval_a[1], eval_b[1]), axis=0))
        # Shuffle data
        train_idx = np.arange(self.train[0].shape[0])
        eval_idx = np.arange(self.eval[0].shape[0])
        np.random.shuffle(train_idx)
        np.random.shuffle(eval_idx)
        self.train = (self.train[0][train_idx], self.train[1][train_idx])
        self.eval = (self.eval[0][eval_idx], self.eval[1][eval_idx])

    def get_dataset_name(self):
        return "Split MNIST"

class cifar10(dataset):
    """
    Class for the CIFAR-10 dataset.
    """

    def __init__(self):
        # Load CIFAR-10 data
        [train, eva] = tf.keras.datasets.cifar10.load_data()
        self.train = (train[0], np.squeeze(train[1]))
        self.eval = (eva[0], np.squeeze(eva[1].astype(np.uint8)))

    def get_dataset_name(self):
        return "CIFAR-10"

class permutated_cifar10(cifar10):
    """
    Class for the CIFAR-10 dataset with permutated pixels.
    """

    def __init__(self, np_seed = 0):
        # Load CIFAR-10 data
        [train, eva] = tf.keras.datasets.cifar10.load_data()
        self.train = (train[0], np.squeeze(train[1]))
        self.eval = (eva[0], np.squeeze(eva[1].astype(np.uint8)))
        # Set seed of numpy
        np.random.seed(np_seed)
        # Create permutation order
        self.idx = np.arange(1024)
        np.random.shuffle(self.idx)
        # Permutate dataset
        self.train = self.permutate(self.train)
        self.eval = self.permutate(self.eval)

    def get_dataset_name(self):
        return "Permutated CIFAR-10"

    def permutate(self, x):
        # Permutate pixels in a order given by self.idx
        for i in range(x[0].shape[0]):
            x[0][i, :, :, 0] = np.reshape(np.reshape(x[0][i, :, :, 0], 1024)[self.idx], (32, 32))
            x[0][i, :, :, 1] = np.reshape(np.reshape(x[0][i, :, :, 1], 1024)[self.idx], (32, 32))
            x[0][i, :, :, 2] = np.reshape(np.reshape(x[0][i, :, :, 2], 1024)[self.idx], (32, 32))
        return x

class split_cifar10(cifar10):
    """
    Class for the CIFAR-10 dataset with task to classify two classes.
    """

    def __init__(self, class_a = [0], class_b = [1]):
        # Load CIFAR-10 data
        [train_tmp, eva_tmp] = tf.keras.datasets.cifar10.load_data()
        train = [train_tmp[0], np.squeeze(train_tmp[1])]
        eva = [eva_tmp[0], np.squeeze(eva_tmp[1].astype(np.uint8))]
        # Get samples and labels of both classes
        train_a = (train[0][self.filter(train[1], class_a)], train[1][self.filter(train[1], class_a)])
        train_b = (train[0][self.filter(train[1], class_b)], train[1][self.filter(train[1], class_b)])
        eval_a = (eva[0][self.filter(eva[1], class_a)], eva[1][self.filter(eva[1], class_a)])
        eval_b = (eva[0][self.filter(eva[1], class_b)], eva[1][self.filter(eva[1], class_b)])
        # Stack both classes ontop of each other
        self.train = (np.concatenate((train_a[0], train_b[0]), axis=0), np.concatenate((train_a[1], train_b[1]), axis=0))
        self.eval = (np.concatenate((eval_a[0], eval_b[0]), axis=0), np.concatenate((eval_a[1], eval_b[1]), axis=0))
        # Shuffle data
        train_idx = np.arange(self.train[0].shape[0])
        eval_idx = np.arange(self.eval[0].shape[0])
        np.random.shuffle(train_idx)
        np.random.shuffle(eval_idx)
        self.train = (self.train[0][train_idx], self.train[1][train_idx])
        self.eval = (self.eval[0][eval_idx], self.eval[1][eval_idx])

    def get_dataset_name(self):
        return "Split CIFAR-10"

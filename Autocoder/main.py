import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import Autocoder as au


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test

#从样本集中随机取连续的batch size个样本


def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index + batch_size)]

mnist=input_data.read_data_sets("G:/TensorFlow/", one_hot=True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 3
batch_size = 128
display_step = 1


autocoder = au.AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200, optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                 scale=0.01)


for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autocoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples * batch_size
    if epoch % display_step==0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

print("Total cost:" + str(autocoder.calc_total_cost(X_test)))

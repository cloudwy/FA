#!/usr/bin/python3.6

import os
import matplotlib
matplotlib.use('Agg')
import datasets
import models
import utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.manifold import TSNE

os.environ["CUDA_VISIBLE_DEVICES"]="3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Constants
cont_latent_size = 10
n_clusters = 16
img_size = [28, 28, 1]

##encoder
enc_neurons = [500,500,2000]
enc_filters = []
##decoder
dec_neurons = [2000, 500, 500, 784]
dec_im_shape = [784]
dec_filters = []

epochs = 100
batch_size = 256
learning_rate = 0.001
num_classes = 10

N_plot = 16
log_path = "./log"

# Create dataset and batching using tensorflow
data_ph = tf.placeholder(tf.float32)
labels_ph = tf.placeholder(tf.float32)
batch_size_ph = tf.placeholder(tf.int64)
shufflebuffer_ph = tf.placeholder(tf.int64)
epochs_ph = tf.placeholder(tf.int64)
repl_data_ph = tf.placeholder(tf.float32)
repl_batch_size_ph = tf.placeholder(tf.int64)
#repl_shufflebuffer_ph = tf.placeholder(tf.int64)

#original dataset
data_ds = tf.data.Dataset.from_tensor_slices(data_ph)
labels_ds = tf.data.Dataset.from_tensor_slices(labels_ph)
dataset = tf.data.Dataset.zip((data_ds, labels_ds)).repeat(epochs_ph).shuffle(shufflebuffer_ph).batch(batch_size_ph)
iterator = dataset.make_initializable_iterator()
[batch_data, batch_labels] = iterator.get_next()
#replay dataset
repl_dataset = tf.data.Dataset.from_tensor_slices(repl_data_ph).repeat().shuffle(shufflebuffer_ph).batch(repl_batch_size_ph)
repl_iterator = repl_dataset.make_initializable_iterator()
repl_batch_data = repl_iterator.get_next()

# Create AE_l2_GR
model = models.AE_l2_GR(batch_data, batch_labels, repl_batch_data, cont_latent_size, img_size, enc_neurons, enc_filters, dec_neurons, dec_im_shape, dec_filters)

# Start tf session
sess = tf.Session(config=config)

# Initialize variables
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

# Create hyperparameter dict
hp_dict = {
"cont_latent_size": cont_latent_size,
"n_clusters": n_clusters,
"img_size": img_size,
"enc_neurons": enc_neurons,
"enc_filters": enc_filters,
"dec_neurons": dec_neurons,
"dec_im_shape": dec_im_shape,
"dec_filters": dec_filters,
"epochs": epochs,
"batch_size": batch_size,
"learning_rate": learning_rate,
"num_classes": num_classes,
"N_plot": N_plot,
"log_path": log_path}

#Create Tensorboard Filewriter & Save Results
[fw,log_path_dt]=utils.create_tensorboard(sess,log_path)

#Start Sequential Learining
acc_summary = []
repl_data = []
for task in range(5):
    #Reinitialize optimizers
    sess.run(tf.variables_initializer(model.opt_recon.variables()))
    #Load data for training
    data = datasets.split_mnist([2 * task], [2 * task + 1])
    [train_data, train_labels] = data.get_train_samples()
    train_data = train_data / 255.0
    if task > 0:
        b_replay = True
        curr_batch_size = np.floor(batch_size / (task + 1.0))
        repl_batch_size = batch_size - curr_batch_size
        sess.run(repl_iterator.initializer, feed_dict={repl_data_ph: repl_data, repl_batch_size_ph: repl_batch_size,
                                                       shufflebuffer_ph: train_data.shape[0]})
    else:
        b_replay = False
        curr_batch_size = batch_size
        repl_batch_size = batch_size
        sess.run(repl_iterator.initializer, feed_dict={repl_data_ph: np.random.randn(1, cont_latent_size), repl_batch_size_ph: repl_batch_size,
                                                       shufflebuffer_ph: train_data.shape[0]})

    sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size,
                                              shufflebuffer_ph: train_data.shape[0], epochs_ph: epochs})


    #Train model
    i = 0
    while True:
        try:
            [_, loss, summaries] = sess.run([model.update_recon, model.loss, model.summaries],
                                            feed_dict={model.learning_rate: learning_rate, model.b_replay: b_replay})
            i += 1
            fw.add_summary(summaries, i)
            if (i % 100 == 0):
                print("Iteration: {}\tloss: {:.5}".format(i, loss))
        except tf.errors.OutOfRangeError:
            break

    # Copy weights from trainable to non-trainable generator
    model.update_gen_weights(sess)
    print("End Training model for task{}".format(task))

    # Get compressed data for replay
    sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size,
                                              shufflebuffer_ph: train_data.shape[0], epochs_ph: 1})
    repl_z_enc = []
    while True:
        try:
            z_enc = sess.run(model.z_enc, feed_dict={model.b_replay: False})
            repl_z_enc.append(z_enc)
        except tf.errors.OutOfRangeError:
            break
    repl_z_enc = np.concatenate(repl_z_enc, axis=0)
    #repl_data = repl_data.append(repl_z_enc)
    if task>0:
        repl_data = np.concatenate((repl_data, repl_z_enc), axis=0)
    else:
        repl_data = repl_z_enc
    print("Collection of compressed replay data complete")

    #Generate and save generative images
    sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size,
                                              shufflebuffer_ph: train_data.shape[0], epochs_ph: 1})
    sess.run(repl_iterator.initializer, feed_dict={repl_data_ph: repl_data, repl_batch_size_ph: batch_size,
                                                   shufflebuffer_ph: train_data.shape[0]})

    org_imgs = sess.run(model.input, feed_dict={model.batch_size: N_plot * N_plot, model.b_replay: False})
    gen_imgs = sess.run(model.static_dec_out,feed_dict={model.batch_size: N_plot * N_plot, model.b_replay: False})
    org_img = np.zeros((28 * N_plot, 28 * N_plot), dtype=np.float32)
    for j in range(N_plot):
        for k in range(N_plot):
            org_img[j * 28:(j + 1) * 28, k * 28:(k + 1) * 28] = np.reshape(org_imgs[j * N_plot + k, :], [28, 28])
    plt.imshow(org_img)
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M")
    fname = log_path_dt + "/" + "org_imgs_AE"+str(task)
    plt.savefig(fname, format="png")
    print("End save original images{}".format(task))
    gen_img = np.zeros((28 * N_plot, 28 * N_plot), dtype=np.float32)
    for j in range(N_plot):
        for k in range(N_plot):
            gen_img[j * 28:(j + 1) * 28, k * 28:(k + 1) * 28] = np.reshape(gen_imgs[j * N_plot + k, :], [28, 28])
    plt.imshow(gen_img)
    fname = log_path_dt + "/" + "gen_imgs_AE"+str(task)
    plt.savefig(fname, format="png")
    plt.close()
    print("End save generated images{}".format(task))

    #Compute accuracy
    #data = datasets.split_mnist(np.arange(2*(task+1)))
    data = datasets.split_mnist(np.arange(2*(task+1)),[])
    [train_data, train_labels] = data.get_train_samples()
    train_data = train_data / 255.0
    acc = utils.acc_AE_GR(train_data, train_labels, sess, model, batch_size, learning_rate, data_ph, labels_ph,
                             batch_size_ph, shufflebuffer_ph,epochs_ph, iterator, n_clusters, num_classes)
    print("Accuracy on Task{} for all previous data:{}".format(task,acc))
    data = datasets.split_mnist([2*task],[2*task+1])
    [train_data,train_labels] = data.get_train_samples()
    acc_current = utils.acc_AE_GR(train_data, train_labels, sess, model, batch_size, learning_rate, data_ph, labels_ph,
                             batch_size_ph, shufflebuffer_ph,epochs_ph, iterator, n_clusters, num_classes)
    print("Accuracy on Task{} for current data:{}".format(task, acc_current))
    acc_summary.append(acc)

# Save results
utils.result_saver1(acc_summary, hp_dict, log_path_dt)





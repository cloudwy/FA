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

os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Constants
cat_latent_size = 16
cont_latent_size = 5
img_size = [28, 28, 1]
##encoder
enc_neurons = [200, 100] #FCL
enc_filters = [32, 32] #CL
##decoder
dec_neurons = [100, 200, 7*7*32] #FCL,7*7*32 is the reshape layer, 32 channels
dec_im_shape = [7, 7, 32]
dec_filters = [32, 32, 1] #Transpose_CL
##discriminator network
disc_neurons = [400, 300, 1]
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

data_ds = tf.data.Dataset.from_tensor_slices(data_ph)
labels_ds = tf.data.Dataset.from_tensor_slices(labels_ph)
dataset = tf.data.Dataset.zip((data_ds, labels_ds)).repeat(epochs_ph).shuffle(shufflebuffer_ph).batch(batch_size_ph)
iterator = dataset.make_initializable_iterator()
[batch_data, batch_labels] = iterator.get_next()

# Create AAE
model = models.AAE(batch_data, batch_labels,cont_latent_size, cat_latent_size, img_size, enc_neurons, enc_filters, dec_neurons, dec_im_shape, dec_filters, disc_neurons)

# Start tf session
sess = tf.Session(config=config)

# Initialize variables
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

# Create hyperparameter dict
hp_dict = {"cat_latent_size": cat_latent_size,
"cont_latent_size": cont_latent_size,
"img_size": img_size,
"enc_neurons": enc_neurons,
"enc_filters": enc_filters,
"dec_neurons": dec_neurons,
"dec_im_shape": dec_im_shape,
"dec_filters": dec_filters,
"disc_neurons": disc_neurons,
"epochs": epochs,
"batch_size": batch_size,
"learning_rate": learning_rate,
"num_classes": num_classes,
"N_plot": N_plot,
"log_path": log_path}

# Create Tensorboard Filewriter & Save Results
[fw,log_path_dt]=utils.create_tensorboard(sess,log_path)

# Start Sequential Learning
acc_summary = []
for task in range(5):
    # Reinitialize optimizers
    sess.run(tf.variables_initializer(model.opt_disc.variables() + model.opt_recon.variables() + model.opt_fool.variables()))
    # Load data for training
    #data = datasets.split_mnist([2*task], [2*task+1])
    data = datasets.split_fashion_mnist([2 * task], [2 * task + 1])

    [train_data, train_labels] = data.get_train_samples()
    train_data = train_data / 255.0
    if task > 0:
        b_replay = True
        curr_batch_size = np.floor(batch_size / (task + 1.0))
        repl_batch_size = batch_size - curr_batch_size
    else:
        b_replay = False
        curr_batch_size = batch_size
        repl_batch_size = batch_size
    sess.run(iterator.initializer,feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: curr_batch_size,shufflebuffer_ph: train_data.shape[0],epochs_ph: epochs})

    # Train model
    i=0
    while True:
        try:
            [_, _, _, loss, summaries] = sess.run([model.update_disc, model.update_fool, model.update_recon, model.loss, model.summaries],
                feed_dict={model.batch_size: batch_size, model.learning_rate: learning_rate,model.b_replay: b_replay,model.repl_batch_size: repl_batch_size})
            i += 1
            fw.add_summary(summaries, i)
            if (i%100 == 0):
                print("Task{}\tIteration: {}\tloss: {:.5}".format(task,i,loss))
        except tf.errors.OutOfRangeError:
            break

    # Copy weights from trainable to non-trainable generator
    model.update_gen_weights(sess)
    print("End Training model for task{}".format(task))

    # Generate and save generative images
    utils.plot_gen_imgs(sess, model, N_plot, log_path_dt, task)

    # Compute accuracy
    # Evaluate the AAE on all tasks individually
    acc_list = []
    for task_acc in range(5):
        if task_acc > task:
            acc=0
        else:
            # Load data for evaluation on one task
            #data = datasets.split_mnist([2 * task_acc], [2 * task_acc + 1])
            data = datasets.split_fashion_mnist([2 * task_acc], [2 * task_acc + 1])
            [train_data, train_labels] = data.get_train_samples()
            train_data = train_data / 255.0

            # Get all cluster assignments for training data
            sess.run(iterator.initializer,
                     feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size,
                                shufflebuffer_ph: train_data.shape[0], epochs_ph: 1})

            label = []
            cluster = []
            while True:
                try:
                    [label_tmp, cluster_tmp] = sess.run([model.label, model.z_enc_cat],
                                                        feed_dict={model.batch_size: batch_size,
                                                                   model.learning_rate: learning_rate,
                                                                   model.b_replay: False,
                                                                   model.repl_batch_size: batch_size})
                    label.append(label_tmp)
                    cluster.append(cluster_tmp)
                except tf.errors.OutOfRangeError:
                    break

            label = np.concatenate(label, axis=0)
            cluster = np.concatenate(cluster, axis=0)

            # Compute cluster accuracy
            counts = np.zeros((cat_latent_size, num_classes), dtype=np.float32)

            for i in range(label.shape[0]):
                j = np.argmax(cluster[i, :], axis=0)
                k = np.int32(label[i])
                counts[j, k] += 1
            cluster_label = np.argmax(counts, axis=1)

            acc = 0
            for i in range(label.shape[0]):
                idx = np.argmax(cluster[i, :], axis=0)
                if (label[i] == cluster_label[idx]):
                    acc += 1
            acc /= np.float32(label.shape[0])
            print("Accuracy on task {}/{}: {}".format(2 * task_acc, 2 * task_acc + 1,acc))
            #print("Cluster accuracy: {}".format(acc))
        acc_list.append(acc)
        #tf.summary.scalar('acc',acc)
        #fw.add_summary(acc,task_acc)
    acc_summary.append(acc_list)

# Save results
utils.result_saver(acc_summary, hp_dict, log_path_dt)









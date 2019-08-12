#!/usr/bin/python3.6

"""
Model:AAE1(only one discriminators)

"""
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

os.environ["CUDA_VISIBLE_DEVICES"]="2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# Constants
cat_latent_size = 16
cont_latent_size = 5
img_size = [28, 28, 1]
##encoder
enc_neurons = [200, 100]
enc_filters = [32, 32]
##decoder
dec_neurons = [100, 200, 7*7*32]
dec_im_shape = [7, 7, 32]
dec_filters = [32, 32, 1]
##discriminator network
disc_neurons = [400, 300, 1]
epochs = 100
batch_size = 256
learning_rate = 0.001

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
model = models.AAE1(batch_data, batch_labels, cont_latent_size, cat_latent_size, img_size, enc_neurons, enc_filters, dec_neurons, dec_im_shape, dec_filters, disc_neurons)

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

#Create Tensorboard Filewriter & Save Results
[fw,log_path_dt]=utils.create_tensorboard(sess,log_path)

# Load data for training
data = datasets.mnist()
#data = datasets.split_mnist([0], [1])
#data = datasets.fashion_mnist()
#data = datasets.split_fashion_mnist([0], [1])

[train_data, train_labels] = data.get_train_samples()
train_data = train_data / 255.0
sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size, shufflebuffer_ph: train_data.shape[0], epochs_ph: epochs})

# Train model
i = 0
while True:
    try:
        [_, _, _, loss,summaries] = sess.run([model.update_disc, model.update_fool, model.update_recon, model.loss,model.summaries],
                                             feed_dict={model.batch_size: batch_size, model.learning_rate: learning_rate, model.b_replay: False, model.repl_batch_size: batch_size})
        i += 1
        fw.add_summary(summaries,i)
        if (i%100 == 0):
            print("Iteration: {}\tloss: {:.5}".format(i, loss))
    except tf.errors.OutOfRangeError:
        break
print("End Training model")


# Generate and save some images
model.update_gen_weights(sess) #Copy weights from trainable to non trainable generator
gen_imgs = sess.run(model.static_dec_out, feed_dict={model.repl_batch_size: N_plot*N_plot})
img = np.zeros((28*N_plot, 28*N_plot), dtype=np.float32)
for j in range(N_plot):
    for k in range(N_plot):
        img[j*28:(j+1)*28, k*28:(k+1)*28] = np.reshape(gen_imgs[j*N_plot+k, :], [28, 28])
plt.imshow(img)
dt = datetime.now().strftime("%Y_%m_%d_%H_%M")
#fname = "./pics/gen_imgs_AAE_"+dt
fname = log_path_dt+"/"+"gen_imgs_AAE"
plt.savefig(fname, format="png")
plt.close()
print("End generate and save images")

# Get all cluster assignments for training data
sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size, shufflebuffer_ph: train_data.shape[0], epochs_ph: 1})

label = []
cluster = []
while True:
    try:
        [label_tmp, cluster_tmp] = sess.run([model.label, model.z_enc_cat], feed_dict={model.batch_size: batch_size, model.learning_rate: learning_rate, model.b_replay: False, model.repl_batch_size: batch_size})
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
print("Cluster accuracy: {}".format(acc))

# Save results
utils.result_saver1(acc, hp_dict, log_path_dt)








#!/usr/bin/python3.6

import os
import matplotlib
#matplotlib.use('Agg')
import datasets
import models
import utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import umap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"]="6"
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
epochs = 1
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
model = models.AAE(batch_data, batch_labels, cont_latent_size, cat_latent_size, img_size, enc_neurons, enc_filters, dec_neurons, dec_im_shape, dec_filters, disc_neurons)

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

model.update_gen_weights(sess)
# UMAP
sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size, shufflebuffer_ph: train_data.shape[0], epochs_ph: 1})
org_img = []
org_label =[]
cont_z = []
cat_z=[]
while True:
    try:
        [img_tmp,label_tmp,z_cont_tmp,z_cat_tmp] = sess.run([model.input,model.label, model.z_enc_cont,model.z_enc_cat], feed_dict={model.batch_size: batch_size,
                                                                                       model.learning_rate: learning_rate,
                                                                                       model.b_replay: False,
                                                                                       model.repl_batch_size: batch_size})
        org_img.append(img_tmp)
        org_label.append(label_tmp)
        cont_z.append(z_cont_tmp)
        cat_z.append(z_cat_tmp)
    except tf.errors.OutOfRangeError:
        break
org_img = np.concatenate(org_img, axis=0)
org_label = np.concatenate(org_label, axis=0)
cont_z = np.concatenate(cont_z, axis=0)
cat_z = np.concatenate(cat_z, axis=0)
print('Data for umap')

'''
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(org_img)
sns.set(context="paper", style="white")
fig, ax = plt.subplots(figsize=(12, 10))
color = org_label.astype(int)
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
plt.show()
'''

# image for original data
reducer = umap.UMAP(random_state=42)
reducer.fit(org_img)
embedding = reducer.transform(org_img)
assert(np.all(embedding == reducer.embedding_))
embedding.shape
plt.scatter(embedding[:, 0], embedding[:, 1], c=org_label, cmap='Spectral', s=10)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24)
#plt.show()
plt.savefig('org.PNG')

# image for latent data
cat_z = np.argmax(cat_z, axis=1)
reducer = umap.UMAP(random_state=42)
reducer.fit(cont_z)
embedding = reducer.transform(cont_z)
assert(np.all(embedding == reducer.embedding_))
embedding.shape
plt.scatter(embedding[:, 0], embedding[:, 1], c=cat_z, cmap='Spectral', s=16)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the latent code of Digits dataset', fontsize=24)
#plt.show()
plt.savefig('cat_z.PNG')






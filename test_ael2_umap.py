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
#for ump
import umap
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"]="3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Constants
cont_latent_size = 10 #in paper:10
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

data_ds = tf.data.Dataset.from_tensor_slices(data_ph)
labels_ds = tf.data.Dataset.from_tensor_slices(labels_ph)
dataset = tf.data.Dataset.zip((data_ds, labels_ds)).repeat(epochs_ph).shuffle(shufflebuffer_ph).batch(batch_size_ph)
iterator = dataset.make_initializable_iterator()
[batch_data, batch_labels] = iterator.get_next()

# Create AAE
model = models.AE_l2(batch_data, batch_labels, cont_latent_size, img_size, enc_neurons, enc_filters, dec_neurons, dec_im_shape, dec_filters)

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

# Load data for training
data = datasets.mnist()
#data = datasets.split_mnist([1], [0])
#data = datasets.fashion_mnist()

[train_data, train_labels] = data.get_train_samples()
train_data = train_data / 255.0
sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size, shufflebuffer_ph: train_data.shape[0], epochs_ph: epochs})

# Train model
i = 0
while True:
    try:
        [_, loss,summaries] = sess.run([model.update_recon, model.loss,model.summaries],
                                             feed_dict={model.batch_size: batch_size, model.learning_rate: learning_rate})
        i += 1
        fw.add_summary(summaries,i)
        if (i%100 == 0):
            print("Iteration: {}\tloss: {:.5}".format(i, loss))
    except tf.errors.OutOfRangeError:
        break
print("End Training model")

# Generate and save some images
sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size, shufflebuffer_ph: train_data.shape[0], epochs_ph: epochs})
[org_imgs,gen_imgs]= sess.run([model.input,model.dec_out], feed_dict={model.batch_size: N_plot*N_plot})
org_img = np.zeros((28*N_plot, 28*N_plot), dtype=np.float32)
for j in range(N_plot):
    for k in range(N_plot):
        org_img[j*28:(j+1)*28, k*28:(k+1)*28] = np.reshape(org_imgs[j*N_plot+k, :], [28, 28])
plt.imshow(org_img)
dt = datetime.now().strftime("%Y_%m_%d_%H_%M")
fname = log_path_dt+"/"+"org_imgs_AE"
plt.savefig(fname, format="png")
print("End save original images")
gen_img = np.zeros((28*N_plot, 28*N_plot), dtype=np.float32)
for j in range(N_plot):
    for k in range(N_plot):
        gen_img[j*28:(j+1)*28, k*28:(k+1)*28] = np.reshape(gen_imgs[j*N_plot+k, :], [28, 28])
plt.imshow(gen_img)
fname = log_path_dt+"/"+"gen_imgs_AE"
plt.savefig(fname, format="png")
plt.close()
print("End save generated images")

# UMAP
[org_img_umap,org_label_umap,cont_z_umap,cluster_umap,cat_z_umap]=utils.AE_Vdata(train_data,train_labels,sess,model,
                                                                                 batch_size,learning_rate,data_ph,
                                                                                 labels_ph,batch_size_ph,shufflebuffer_ph,
                                                                                 epochs_ph,iterator,n_clusters,num_classes)
##umap for original data
reducer = umap.UMAP(random_state = 42)
reducer.fit(org_img_umap)
embedding = reducer.transform(org_img_umap)
assert(np.all(embedding == reducer.embedding_))
embedding.shape
plt.scatter(embedding[:, 0], embedding[:, 1], c=org_label_umap, cmap='Spectral', s=0.1)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(num_classes+1)-0.5).set_ticks(np.arange(num_classes))
plt.title('Original Code', fontsize=12)
fname = log_path_dt+"/"+"org_imgs_AE_umap"
plt.savefig(fname, format="png")
plt.close()
print("End save original images of umap1")
##umap for latent code(cluster)
reducer = umap.UMAP(random_state = 42)
reducer.fit(cont_z_umap)
embedding = reducer.transform(cont_z_umap)
assert(np.all(embedding == reducer.embedding_))
embedding.shape
plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_umap, cmap='Spectral', s=0.1)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(n_clusters+1)-0.5).set_ticks(np.arange(n_clusters))
plt.title('Latent Code to Cluster_Label', fontsize=12)
fname = log_path_dt+"/"+"latent_imgs_AE_umap"
plt.savefig(fname, format="png")
plt.close()
print("End save generated images of umap1")
##umap for latent code(cluster->class)
reducer = umap.UMAP(random_state = 42)
reducer.fit(cont_z_umap)
embedding = reducer.transform(cont_z_umap)
assert(np.all(embedding == reducer.embedding_))
embedding.shape
plt.scatter(embedding[:, 0], embedding[:, 1], c=cat_z_umap, cmap='Spectral', s=0.1)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(num_classes+1)-0.5).set_ticks(np.arange(num_classes))
plt.title('Latent Code to Class_Label', fontsize=12)
fname = log_path_dt+"/"+"gen_imgs_AE_umap"
plt.savefig(fname, format="png")
plt.close()
print("End save generated images of umap2")

# Computate acc_train and acc_test
#acc_train
acc_train = utils.acc_AE(train_data,train_labels,sess,model,batch_size,learning_rate,data_ph,labels_ph,batch_size_ph,shufflebuffer_ph,
                   epochs_ph,iterator,n_clusters,num_classes)
print("Cluster accuracy for Train_data: {}".format(acc_train))
#acc_test
[test_data, test_labels] = data.get_eval_samples()
test_data = test_data / 255.0
acc_test = utils.acc_AE(test_data,test_labels,sess,model,batch_size,learning_rate,data_ph,labels_ph,batch_size_ph,shufflebuffer_ph,
                   epochs_ph,iterator,n_clusters,num_classes)
print("Cluster accuracy for Test_data: {}".format(acc_test))

# Save results
utils.result_saver(acc_train,acc_test, hp_dict, log_path_dt)








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

#for ump
import umap
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"]="7"
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
epochs = 200
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
acc_pre = []
acc_curr = []
for task in range(5):
    # Reinitialize optimizers
    sess.run(tf.variables_initializer(model.opt_disc.variables() + model.opt_recon.variables() + model.opt_fool.variables()))
    # Load data for training
    data = datasets.split_mnist([2*task], [2*task+1])
    #data = datasets.split_fashion_mnist([2 * task], [2 * task + 1])
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
    data = datasets.split_mnist(np.arange(2 * (task + 1)), [])
    # data = datasets.split_fashion_mnist(np.arange(2 * (task + 1)), [])
    [train_data, train_labels] = data.get_train_samples()
    train_data = train_data / 255.0
    sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size,
                                              shufflebuffer_ph: train_data.shape[0], epochs_ph: epochs})
    [org_imgs, gen_imgs] = sess.run([model.original, model.static_dec_out],
                                    feed_dict={model.batch_size: N_plot * N_plot,
                                               model.repl_batch_size: N_plot * N_plot, model.b_replay: False})
    org_img = np.zeros((28 * N_plot, 28 * N_plot), dtype=np.float32)
    for j in range(N_plot):
        for k in range(N_plot):
            org_img[j * 28:(j + 1) * 28, k * 28:(k + 1) * 28] = np.reshape(org_imgs[j * N_plot + k, :], [28, 28])
    plt.title('Task ' + str(task + 1), fontsize=12)
    plt.imshow(org_img)
    fname = log_path_dt + "/" + "org_imgs_AAE_GR" + str(task) + ".png"
    plt.savefig(fname)
    print("End save original images")
    gen_img = np.zeros((28 * N_plot, 28 * N_plot), dtype=np.float32)
    for j in range(N_plot):
        for k in range(N_plot):
            gen_img[j * 28:(j + 1) * 28, k * 28:(k + 1) * 28] = np.reshape(gen_imgs[j * N_plot + k, :], [28, 28])
    plt.title('Task ' + str(task + 1), fontsize=12)
    plt.imshow(gen_img)
    fname = log_path_dt + "/" + "gen_imgs_AAE_GR" + str(task) + ".png"
    plt.savefig(fname)
    plt.close()
    print("End generate and save images for task %d" % task)
    '''
    # UMAP
    data = datasets.split_mnist(np.arange(2 * (task + 1)), [])
    # data = datasets.split_fashion_mnist(np.arange(2 * (task + 1)), [])
    [train_data, train_labels] = data.get_train_samples()
    train_data = train_data / 255.0
    [org_img_umap, org_label_umap, cont_z_umap, cluster_umap, cat_z_umap] = utils.AAE_Vdata(train_data, train_labels,
                                                                                            sess, model, batch_size,
                                                                                            learning_rate, data_ph,
                                                                                            labels_ph, batch_size_ph,
                                                                                            shufflebuffer_ph, epochs_ph,
                                                                                            iterator, num_classes,
                                                                                            cat_latent_size)
    ##umap for original data
    reducer = umap.UMAP(random_state=42)
    reducer.fit(org_img_umap)
    embedding = reducer.transform(org_img_umap)
    assert (np.all(embedding == reducer.embedding_))
    embedding.shape
    plt.scatter(embedding[:, 0], embedding[:, 1], c=org_label_umap, cmap='Spectral', s=0.1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(2 * (task + 1) + 1) - 0.5).set_ticks(np.arange(2 * (task + 1)))
    plt.title('Task ' + str(task + 1), fontsize=12)
    fname = log_path_dt + "/" + "org_imgs_AAE_GR_umap" + str(task) + ".png"
    plt.savefig(fname)
    plt.close()
    print("End save original images of umap{}".format(task))
    ##umap for latent code(cluster)
    reducer = umap.UMAP(random_state=42)
    reducer.fit(cont_z_umap)
    embedding = reducer.transform(cont_z_umap)
    assert (np.all(embedding == reducer.embedding_))
    embedding.shape
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_umap, cmap='Spectral', s=0.1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(cat_latent_size + 1) - 0.5).set_ticks(np.arange(cat_latent_size))
    plt.title('Task ' + str(task + 1), fontsize=12)
    fname = log_path_dt + "/" + "latent_imgs_AAE_GR_umap" + str(task) + ".png"
    plt.savefig(fname)
    plt.close()
    print("End save generated images of umap{}".format(task))
    ##umap for latent code(cluster->class)
    reducer = umap.UMAP(random_state=42)
    reducer.fit(cont_z_umap)
    embedding = reducer.transform(cont_z_umap)
    assert (np.all(embedding == reducer.embedding_))
    embedding.shape
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cat_z_umap, cmap='Spectral', s=0.1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(2 * (task + 1) + 1) - 0.5).set_ticks(np.arange(2 * (task + 1)))
    plt.title('Task ' + str(task + 1), fontsize=12)
    fname = log_path_dt + "/" + "gen_imgs_AAE_GR_umap" + str(task) + ".png"
    plt.savefig(fname)
    plt.close()
    print("End save generated images of umap{}".format(task))
    '''
    # Compute accuracy
    acc_list = []
    # Load all previous data
    data = datasets.split_mnist(np.arange(2 * (task + 1)), [])
    [train_data, train_labels] = data.get_train_samples()
    train_data = train_data / 255.0
    acc_train_pre = utils.acc_AAE(train_data, train_labels, sess, model, batch_size, learning_rate, data_ph, labels_ph,
                              batch_size_ph, shufflebuffer_ph, epochs_ph, iterator, num_classes, cat_latent_size)
    print("Accuracy on Task{} for all previous data:{}".format(task, acc_train_pre))
    acc_pre.append(acc_train_pre)
    # Load current data
    data = datasets.split_mnist([2 * task], [2 * task + 1])
    [train_data, train_labels] = data.get_train_samples()
    train_data = train_data / 255.0
    acc_train_curr = utils.acc_AAE(train_data, train_labels, sess, model, batch_size, learning_rate, data_ph, labels_ph,
                                   batch_size_ph, shufflebuffer_ph, epochs_ph, iterator, num_classes, cat_latent_size)
    print("Accuracy on Task{} for current data:{}".format(task, acc_train_curr))
    acc_curr.append(acc_train_curr)

# Save results
utils.result_saver1(acc_pre,acc_curr,hp_dict, log_path_dt)








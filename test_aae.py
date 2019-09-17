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

os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

# Create Tensorboard Filewriter & Save Results
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

model.update_gen_weights(sess) #Copy weights from trainable to non trainable generator

# Generate and save some images
img = utils.plot_gen_imgs(sess, model, N_plot)
plt.imshow(img)
fname = log_path_dt+"/"+"gen_imgs_AAE"
plt.savefig(fname, format="png")
plt.close()
print("End generate and save images")

#Computate acc_train and acc_test
#acc_train
acc_train = utils.acc_AAE(train_data,train_labels,sess,model,batch_size,learning_rate,data_ph,labels_ph,batch_size_ph,shufflebuffer_ph,epochs_ph,
           iterator,num_classes,cat_latent_size)
print("Cluster accuracy for Train_data: {}".format(acc_train))
#acc_test
[test_data, test_labels] = data.get_eval_samples()
test_data = test_data / 255.0
acc_test = utils.acc_AAE(test_data,test_labels,sess,model,batch_size,learning_rate,data_ph,labels_ph,batch_size_ph,shufflebuffer_ph,epochs_ph,
           iterator,num_classes,cat_latent_size)
print("Cluster accuracy for Test_data: {}".format(acc_test))

# Save results
utils.result_saver(acc_train, acc_test, hp_dict, log_path_dt)








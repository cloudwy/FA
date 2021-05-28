"""
This file contains utility functions.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.cluster import KMeans
from collections import Counter

def initialize_train_data(sess, task, iterator, data, labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph):
    curr_batch_size = np.floor(batch_size/(task+1.0))
    repl_batch_size = batch_size - curr_batch_size
    sess.run(iterator.initializer, feed_dict={data_ph: data, labels_ph: labels, batch_size_ph: curr_batch_size, shufflebuffer_ph: data.shape[0]})

def initialize_eval_data(sess, iterator, data, labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph):
    sess.run(iterator.initializer, feed_dict={data_ph: data, labels_ph: labels, batch_size_ph: batch_size, shufflebuffer_ph: data.shape[0]})

def train(sess, task, model, epochs, iterator, data, labels, batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph):
    curr_batch_size = np.floor(batch_size/(task+1.0))
    repl_batch_size = batch_size - curr_batch_size
    if (task > 0):
        b_replay = True
    else:
        b_replay = False
    for i in range(epochs):
        initialize_train_data(sess, task, iterator, data, labels, curr_batch_size, data_ph, labels_ph, batch_size_ph, shufflebuffer_ph)
        while True:
            try:
                for i in range(5):
                    sess.run(model.update_disc, feed_dict={model.gen_sample_batch_size: repl_batch_size, model.b_replay: b_replay})
                    sess.run(model.clip)
                [_, loss] = sess.run([model.update_gen, model.loss], feed_dict={model.gen_sample_batch_size: repl_batch_size, model.b_replay: b_replay})
            except tf.errors.OutOfRangeError:
                break
        print("Epoch: {}\tloss: {:.5}".format(i, loss))
    model.update_gen_weights(sess)

"""
def plot_gen_imgs(sess, model, gen_sample_batch_size, log_path_dt, i):
    img_len = np.int32(np.sqrt(gen_sample_batch_size))
    gen_imgs = sess.run(model.gen.output, feed_dict={model.gen_sample_batch_size: gen_sample_batch_size, model.b_replay: False})
    img = np.zeros((28*img_len, 28*img_len), dtype=np.float32)
    for j in range(img_len):
        for k in range(img_len):
            img[j*28:(j+1)*28, k*28:(k+1)*28] = np.reshape(gen_imgs[j*img_len+k, :], [28, 28])
    plt.imshow(img)
    fname = log_path_dt+"/gen_imgs_"+str(i)
    plt.savefig(fname, format="png")
    plt.close()
"""
'''
def plot_gen_imgs1(sess,model,N_plot,log_path_dt,i):
    gen_imgs = sess.run(model.static_dec_out,feed_dict={model.repl_batch_size: N_plot*N_plot})
    img = np.zeros((28 * N_plot, 28 * N_plot), dtype=np.float32)
    for j in range(N_plot):
        for k in range(N_plot):
            img[j * 28:(j + 1) * 28, k * 28:(k + 1) * 28] = np.reshape(gen_imgs[j * N_plot + k, :], [28, 28])
    plt.imshow(img)
    fname = log_path_dt+"/gen_imgs_AAE"+str(i)
    plt.savefig(fname, format="png")
    plt.close()
    print("End generate and save images for task %d" %i)
'''

def plot_gen_imgs(sess,model,N_plot):
    gen_imgs = sess.run(model.static_dec_out, feed_dict={model.repl_batch_size: N_plot * N_plot})
    img = np.zeros((28 * N_plot, 28 * N_plot), dtype=np.float32)
    for j in range(N_plot):
        for k in range(N_plot):
            img[j * 28:(j + 1) * 28, k * 28:(k + 1) * 28] = np.reshape(gen_imgs[j * N_plot + k, :], [28, 28])
    return img

def create_tensorboard(sess, log_path):
    # Create log path if it does not exist
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M")
    log_path_dt = log_path+"/"+dt
    os.makedirs(log_path_dt, exist_ok=True)
    # Create tensorboard FileWriter
    fw = tf.summary.FileWriter(log_path_dt, sess.graph, flush_secs=1)
    return fw, log_path_dt

def result_saver(acc_train,acc_test, hp_dict, log_path):
    # for all data
    # Create log path if it does not exist
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M")
    log_path_dt = log_path + "/" + dt
    filename = log_path+"/"+"log.txt"
    os.makedirs(log_path, exist_ok=True)
    # Save hyperparameters and metrics in a log folder
    with open(filename, "w") as f:
        f.write("+---------------------------+\n")
        f.write("| Accuracy_Train            |\n")
        f.write("+---------------------------+\n")
        f.write(str(acc_train)+"\n")
        f.write("+---------------------------+\n")
        f.write("| Accuracy_Test             |\n")
        f.write("+---------------------------+\n")
        f.write(str(acc_test)+"\n")
        f.write("+---------------------------+\n")
        f.write("| Hyperparameters           |\n")
        f.write("+---------------------------+\n")
        for key, val in hp_dict.items():
            f.write(key.ljust(25)+": "+str(val)+"\n")

def result_saver1(acc_pre,acc_curr, hp_dict, log_path):
    # for generative replay
    # Create log path if it does not exist
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M")
    log_path_dt = log_path + "/" + dt
    filename = log_path+"/"+"log.txt"
    os.makedirs(log_path, exist_ok=True)
    # Save hyperparameters and metrics in a log folder
    with open(filename, "w") as f:
        f.write("+---------------------------+\n")
        f.write("| Accuracy_Pre            |\n")
        f.write("+---------------------------+\n")
        f.write(str(acc_pre)+"\n")
        f.write("+---------------------------+\n")
        f.write("| Accuracy_Curr             |\n")
        f.write("+---------------------------+\n")
        f.write(str(acc_curr) + "\n")
        f.write("+---------------------------+\n")
        f.write("| Hyperparameters           |\n")
        f.write("+---------------------------+\n")
        for key, val in hp_dict.items():
            f.write(key.ljust(25)+": "+str(val)+"\n")

def acc_AAE(train_data,train_labels,sess,model,batch_size,learning_rate,data_ph,labels_ph,batch_size_ph,shufflebuffer_ph,epochs_ph,
           iterator,num_classes,cat_latent_size):
    # Get all cluster assignments for training data
    sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size,
                                              shufflebuffer_ph: train_data.shape[0], epochs_ph: 1})
    label = []
    cluster = []
    while True:
        try:
            [label_tmp, cluster_tmp] = sess.run([model.label, model.z_enc_cat], feed_dict={model.batch_size: batch_size,
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
    return acc

def acc_AAE2(train_data,train_labels,sess,model,batch_size,learning_rate,data_ph,labels_ph,batch_size_ph,shufflebuffer_ph,epochs_ph,
           iterator,num_classes,cat_latent_size,momentum_recon,momentum):
    # for the model AAE2: change optimizer and add momentum
    # Get all cluster assignments for training data
    sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size,
                                              shufflebuffer_ph: train_data.shape[0], epochs_ph: 1})
    label = []
    cluster = []
    while True:
        try:
            [label_tmp, cluster_tmp] = sess.run([model.label, model.z_enc_cat], feed_dict={model.batch_size: batch_size,
                                                                                           model.learning_rate: learning_rate,
                                                                                           model.b_replay: False,
                                                                                           model.repl_batch_size: batch_size,
                                                                                           model.momentum_recon:momentum_recon,
                                                                                           model.momentum:momentum})
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
    return acc

def acc_AE(data,labels,sess,model,batch_size,learning_rate,data_ph,labels_ph,batch_size_ph,shufflebuffer_ph,epochs_ph,
           iterator,n_clusters,num_classes):
    # Get all cluster assignments for training data
    sess.run(iterator.initializer, feed_dict={data_ph: data, labels_ph: labels, batch_size_ph: batch_size,
                                              shufflebuffer_ph: data.shape[0], epochs_ph: 1})
    label = []
    feature = []
    while True:
        try:
            [label_tmp, feature_tmp] = sess.run([model.label, model.z_enc], feed_dict={model.batch_size: batch_size,
                                                                                       model.learning_rate: learning_rate})
            label.append(label_tmp)
            feature.append(feature_tmp)
        except tf.errors.OutOfRangeError:
            break
    label = np.concatenate(label, axis=0)
    feature = np.concatenate(feature, axis=0)
    # K-Means
    cluster = KMeans(n_clusters, random_state=9).fit_predict(feature)
    # Determine the label for each cluster
    counts = np.zeros((n_clusters, num_classes), dtype=np.float32)
    for i in range(label.shape[0]):
        j = np.int32(cluster[i])
        k = np.int32(label[i])
        counts[j, k] += 1

    cluster_label = np.argmax(counts, axis=1)
    # Compute cluster accuracy
    acc = 0
    for i in range(label.shape[0]):
        idx = cluster[i]
        if (label[i] == cluster_label[idx]):
            acc += 1
    print("Total number of data：{} True classification: {}".format(data.shape[0],acc))
    acc /= np.float32(label.shape[0])
    return acc

def acc_AE_GR(data,labels,sess,model,batch_size,learning_rate,data_ph,labels_ph,batch_size_ph,shufflebuffer_ph,epochs_ph,
           iterator,n_clusters,num_classes):
    # Get all cluster assignments for training data
    sess.run(iterator.initializer, feed_dict={data_ph: data, labels_ph: labels, batch_size_ph: batch_size,
                                              shufflebuffer_ph: data.shape[0], epochs_ph: 1})
    label = []
    feature = []
    while True:
        try:
            [label_tmp, feature_tmp] = sess.run([model.label, model.z_enc], feed_dict={model.batch_size: batch_size,
                                                model.learning_rate: learning_rate,model.b_replay:False})
            label.append(label_tmp)
            feature.append(feature_tmp)
        except tf.errors.OutOfRangeError:
            break
    label = np.concatenate(label, axis=0)
    feature = np.concatenate(feature, axis=0)
    # K-Means
    cluster = KMeans(n_clusters, random_state=9).fit_predict(feature)
    # Determine the label for each cluster
    counts = np.zeros((n_clusters, num_classes), dtype=np.float32)
    for i in range(label.shape[0]):
        j = np.int32(cluster[i])
        k = np.int32(label[i])
        counts[j, k] += 1

    cluster_label = np.argmax(counts, axis=1)
    # Compute cluster accuracy
    acc = 0
    for i in range(label.shape[0]):
        idx = cluster[i]
        if (label[i] == cluster_label[idx]):
            acc += 1
    print("Total number of data：{} True classification: {}".format(data.shape[0],acc))
    acc /= np.float32(label.shape[0])
    return acc

def AE_Vdata(data,labels,sess,model,batch_size,learning_rate,data_ph,labels_ph,batch_size_ph,shufflebuffer_ph,epochs_ph,
           iterator,n_clusters,num_classes):
    # Get all cluster addignments for training data
    sess.run(iterator.initializer, feed_dict={data_ph: data, labels_ph: labels, batch_size_ph: batch_size,
                                              shufflebuffer_ph: data.shape[0], epochs_ph: 1})
    org_img = []
    org_label = []
    cont_z = []
    while True:
        try:
            [img_tmp,label_tmp, z_cont_tmp] = sess.run([model.input, model.label, model.z_enc], feed_dict={model.batch_size: batch_size,
                                                                                       model.learning_rate: learning_rate})
            org_img.append(img_tmp)
            org_label.append(label_tmp)
            cont_z.append(z_cont_tmp)
        except tf.errors.OutOfRangeError:
            break
    org_img = np.concatenate(org_img, axis=0)
    org_label = np.concatenate(org_label, axis=0)
    cont_z = np.concatenate(cont_z, axis=0)
    # K-Means
    cluster = KMeans(n_clusters,random_state=9).fit_predict(cont_z)
    # Determine the label for each cluster
    counts = np.zeros((n_clusters, num_classes), dtype=np.float32)
    for i in range(org_label.shape[0]):
        j = np.int32(cluster[i])
        k = np.int32(org_label[i])
        counts[j, k] += 1
    cluster_label = np.argmax(counts, axis=1)
    #Determine the cat_z
    cat_z = []
    for i in range(org_label.shape[0]):
        idx = cluster[i]
        cat_z.append(cluster_label[idx])
    cat_z = np.array(cat_z)
    return org_img,org_label,cont_z,cluster,cat_z

def AE_GR_Vdata(data,labels,sess,model,batch_size,learning_rate,data_ph,labels_ph,batch_size_ph,shufflebuffer_ph,epochs_ph,
           iterator,n_clusters,num_classes):
    # Get all cluster assignments for training data
    sess.run(iterator.initializer, feed_dict={data_ph: data, labels_ph: labels, batch_size_ph: batch_size,
                                              shufflebuffer_ph: data.shape[0], epochs_ph: 1})
    org_img = []
    org_label = []
    cont_z = []
    while True:
        try:
            [org_img_tmp, org_label_tmp, cont_z_tmp] = sess.run([model.input, model.label, model.z_enc], feed_dict={model.batch_size: batch_size,
                                                model.learning_rate: learning_rate,model.b_replay:False})
            org_img.append(org_img_tmp)
            org_label.append(org_label_tmp)
            cont_z.append(cont_z_tmp)
        except tf.errors.OutOfRangeError:
            break
    org_img = np.concatenate(org_img, axis=0)
    org_label = np.concatenate(org_label, axis=0)
    cont_z = np.concatenate(cont_z, axis=0)
    # K-Means
    cluster = KMeans(n_clusters, random_state=9).fit_predict(cont_z)
    # Determine the label for each cluster
    counts = np.zeros((n_clusters, num_classes), dtype=np.float32)
    for i in range(org_label.shape[0]):
        j = np.int32(cluster[i])
        k = np.int32(org_label[i])
        counts[j, k] += 1
    cluster_label = np.argmax(counts, axis=1)
    # Determine the cat_z
    cat_z = []
    for i in range(org_label.shape[0]):
        idx = cluster[i]
        cat_z.append(cluster_label[idx])
    cat_z = np.array(cat_z)
    return org_img, org_label, cont_z, cluster, cat_z

def AAE_Vdata(train_data,train_labels,sess,model,batch_size,learning_rate,data_ph,labels_ph,batch_size_ph,shufflebuffer_ph,epochs_ph,
           iterator,num_classes,cat_latent_size):
    # Get all cluster assignments for training data
    sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size,
                                              shufflebuffer_ph: train_data.shape[0], epochs_ph: 1})
    org_img = []
    org_label =[]
    cont_z = []
    cat_z_matrix = []
    while True:
        try:
            [img_tmp, label_tmp, z_cont_tmp, z_cat_tmp] = sess.run([model.original, model.label, model.z_enc_cont,model.z_enc_cat], feed_dict={model.batch_size: batch_size,
                                                                                           model.learning_rate: learning_rate,
                                                                                           model.b_replay: False,
                                                                                           model.repl_batch_size: batch_size})
            org_img.append(img_tmp)
            org_label.append(label_tmp)
            cont_z.append(z_cont_tmp)
            cat_z_matrix.append(z_cat_tmp)
        except tf.errors.OutOfRangeError:
            break
    org_img = np.concatenate(org_img,axis=0)
    org_img = np.reshape(org_img, [-1, 784])
    org_label = np.concatenate(org_label,axis=0)
    cont_z = np.concatenate(cont_z,axis=0)
    cat_z_matrix= np.concatenate(cat_z_matrix,axis=0)
    #determine the label for each cluster
    counts = np.zeros((cat_latent_size, num_classes), dtype=np.float32)
    for i in range(org_label.shape[0]):
        j = np.argmax(cat_z_matrix[i, :], axis=0)
        k = np.int32(org_label[i])
        counts[j, k] += 1
    cluster_label = np.argmax(counts, axis=1)
    #determine cluster
    cluster = np.argmax(cat_z_matrix, axis=1)
    #determine cat_z
    cat_z = []
    for i in range(org_label.shape[0]):
        idx = cluster[i]
        cat_z.append(cluster_label[idx])
    cat_z = np.array(cat_z)
    return org_img,org_label,cont_z, cluster,cat_z

def AAE_Vdata1(train_data,train_labels,sess,model,batch_size,learning_rate,data_ph,labels_ph,batch_size_ph,shufflebuffer_ph,epochs_ph,
           iterator,num_classes,cat_latent_size):
    #''' latent code = continuous+categorical'''
    # Get all cluster assignments for training data
    sess.run(iterator.initializer, feed_dict={data_ph: train_data, labels_ph: train_labels, batch_size_ph: batch_size,
                                              shufflebuffer_ph: train_data.shape[0], epochs_ph: 1})
    org_img = []
    org_label =[]
    cont_z = []
    cat_z_matrix = []
    while True:
        try:
            [img_tmp, label_tmp, z_cont_tmp, z_cat_tmp] = sess.run([model.original, model.label, model.z_enc_cont,model.z_enc_cat], feed_dict={model.batch_size: batch_size,
                                                                                           model.learning_rate: learning_rate,
                                                                                           model.b_replay: False,
                                                                                           model.repl_batch_size: batch_size})
            org_img.append(img_tmp)
            org_label.append(label_tmp)
            cont_z.append(z_cont_tmp)
            cat_z_matrix.append(z_cat_tmp)
        except tf.errors.OutOfRangeError:
            break
    org_img = np.concatenate(org_img,axis=0)
    org_img = np.reshape(org_img, [-1, 784])
    org_label = np.concatenate(org_label,axis=0)
    cont_z = np.concatenate(cont_z,axis=0)
    cat_z_matrix= np.concatenate(cat_z_matrix,axis=0)
    #determine the label for each cluster
    counts = np.zeros((cat_latent_size, num_classes), dtype=np.float32)
    for i in range(org_label.shape[0]):
        j = np.argmax(cat_z_matrix[i, :], axis=0)
        k = np.int32(org_label[i])
        counts[j, k] += 1
    cluster_label = np.argmax(counts, axis=1)
    #determine cluster
    cluster = np.argmax(cat_z_matrix, axis=1)
    #determine cat_z
    cat_z = []
    for i in range(org_label.shape[0]):
        idx = cluster[i]
        cat_z.append(cluster_label[idx])
    cat_z = np.array(cat_z)
    return org_img,org_label,cont_z, cat_z_matrix, cluster,cat_z

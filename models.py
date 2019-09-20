"""
These classes provide models for training.
"""

import abc
import numpy as np
import tensorflow as tf

class model:
    """
    Base class for model.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

class AAE(model):
    """
    Adversarial Autoencoder
    """

    def __init__(self, x, y, cont_latent_size, cat_latent_size, img_size, enc_neurons, enc_filters, dec_neurons, dec_im_shape, dec_filters, disc_neurons):
        # Make variables available to class
        self.cont_latent_size = cont_latent_size
        self.cat_latent_size = cat_latent_size
        self.img_size = img_size
        self.enc_neurons = enc_neurons
        self.enc_filters = enc_filters
        self.dec_neurons = dec_neurons
        self.dec_im_shape = dec_im_shape
        self.dec_filters = dec_filters
        self.disc_neurons = disc_neurons
        self.label = y

        # Define constants
        self.enc_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)
        self.dec_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)
        self.disc_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)

        # Create placeholders
        self.batch_size = tf.placeholder(tf.int32)
        self.repl_batch_size = tf.placeholder(tf.int32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.b_replay = tf.placeholder(tf.bool)

        # Create input based on generative replay
        x = tf.reshape(x, [-1, np.prod(img_size)]) #x: (?,784)matrix
        self.cont_prior = tf.distributions.Normal(tf.zeros(self.cont_latent_size), tf.ones(self.cont_latent_size))
        self.cat_prior = tf.distributions.Categorical(probs=tf.constant(np.ones(self.cat_latent_size)/self.cat_latent_size))
        self.z_cont_static = self.cont_prior.sample(self.repl_batch_size)
        self.z_cat_static = tf.one_hot(self.cat_prior.sample(self.repl_batch_size), self.cat_latent_size)
        self.static_dec_out = self.build_static_dec(tf.concat([self.z_cont_static, self.z_cat_static], axis=1))
        self.input = tf.cond(self.b_replay, lambda: tf.concat([x, self.static_dec_out], axis=0), lambda: x)

        # Build structure
        [self.z_enc_cont, self.z_enc_cat] = self.build_enc(self.input)
        self.z_cont = self.cont_prior.sample(tf.shape(self.input)[0]) #[?,15]
        self.z_cat = tf.one_hot(self.cat_prior.sample(tf.shape(self.input)[0]), self.cat_latent_size) #[?,16]
        self.dec_out = self.build_dec(tf.concat([self.z_enc_cont, self.z_enc_cat], axis=1)) #tf.concat: ?*31, dec_out:?*784
        [disc_cont_real_out, disc_cont_real_feature] = self.build_disc_cont(self.z_enc_cont)
        [disc_cont_fake_out, disc_cont_fake_feature] = self.build_disc_cont(self.z_cont)
        [disc_cat_real_out, disc_cat_real_feature] = self.build_disc_cat(self.z_enc_cat)
        [disc_cat_fake_out, disc_cat_fake_feature] = self.build_disc_cat(self.z_cat)
        # 1->fake 0->real
        loss_disc_cont = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_cont_fake_out), logits=disc_cont_fake_out)),\
                            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_cont_real_out), logits=disc_cont_real_out)))
        loss_disc_cont_fool = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_cont_real_out), logits=disc_cont_real_out))
        loss_disc_cat = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_cat_fake_out), logits=disc_cat_fake_out)),\
                            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_cat_real_out), logits=disc_cat_real_out)))
        loss_disc_cat_fool = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_cat_real_out), logits=disc_cat_real_out))
        loss_disc = tf.add(loss_disc_cont, loss_disc_cat)
        loss_disc_fool = tf.add(loss_disc_cont_fool, loss_disc_cat_fool)
        loss_recon = tf.reduce_mean(tf.square(tf.subtract(self.input, self.dec_out))) #l2 loss
        self.loss = loss_recon
        tf.summary.scalar('loss_recon',loss_recon)
        tf.summary.scalar('loss_disc',loss_disc)
        tf.summary.scalar('loss_disc_fool',loss_disc_fool)
        self.summaries = tf.summary.merge_all()


        # Create optimizer
        self.opt_disc = tf.train.RMSPropOptimizer(self.learning_rate)
        self.opt_recon = tf.train.RMSPropOptimizer(self.learning_rate)
        self.opt_fool = tf.train.RMSPropOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.update_disc = self.opt_disc.minimize(loss_disc, var_list=tf.trainable_variables("AAE_Disc_Cont")+tf.trainable_variables("AAE_Disc_Cat"))
            self.update_recon = self.opt_recon.minimize(loss_recon, var_list=tf.trainable_variables("AAE_Dec")+tf.trainable_variables("AAE_Enc"))
            self.update_fool = self.opt_fool.minimize(loss_disc_fool, var_list=tf.trainable_variables("AAE_Enc"))

    def build_dec(self, z):
        # Build generator
        output = z #[?,31]
        output = tf.reshape(output, [-1, self.cont_latent_size + self.cat_latent_size]) #[?,31]
        with tf.name_scope("AAE_Dec"):
            with tf.variable_scope("AAE_Dec_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.dec_neurons):
                    output = tf.layers.Dense(n, self.dec_activation)(output)
                output = tf.reshape(output, [-1]+self.dec_im_shape)
                for i, n in enumerate(self.dec_filters):
                    if (i == len(self.dec_filters)-1):
                        output = tf.layers.Conv2DTranspose(n, 3, 1, activation=tf.nn.sigmoid, padding="SAME")(output)
                    else:
                        output = tf.layers.Conv2DTranspose(n, 3, 2, activation=self.dec_activation, padding="SAME")(output)
                output = tf.layers.Flatten()(output)
        return output

    def build_static_dec(self, z):
        # Build generator(using for the generative replay)
        output = tf.reshape(z, [-1, self.cont_latent_size + self.cat_latent_size])
        with tf.name_scope("AAE_Static_Dec"):
            with tf.variable_scope("AAE_Static_Dec_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.dec_neurons):
                    output = tf.layers.Dense(n, self.dec_activation, trainable=False)(output)
                output = tf.reshape(output, [-1]+self.dec_im_shape)
                for i, n in enumerate(self.dec_filters):
                    if (i == len(self.dec_filters)-1):
                        output = tf.layers.Conv2DTranspose(n, 3, 1, activation=tf.nn.sigmoid, padding="SAME", trainable=False)(output)
                    else:
                        output = tf.layers.Conv2DTranspose(n, 3, 2, activation=self.dec_activation, padding="SAME", trainable=False)(output)
                output = tf.layers.Flatten()(output)
        return output

    def build_disc_cont(self, z):
        # Build discriminator
        output = z
        with tf.name_scope("AAE_Disc_Cont"):
            with tf.variable_scope("AAE_Disc_Cont_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.disc_neurons):
                    if (i == len(self.disc_neurons)-1):
                        feature = output
                        output = tf.layers.Dense(n)(output)
                    else:
                        output = tf.layers.Dense(n, self.disc_activation)(output)
                        output = tf.layers.Dropout(0.2)(output, training=True)
        return [output, feature]

    def build_disc_cat(self, z):
        # Build discriminator
        output = z
        with tf.name_scope("AAE_Disc_Cat"):
            with tf.variable_scope("AAE_Disc_Cat_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.disc_neurons):
                    if (i == len(self.disc_neurons)-1):
                        feature = output
                        output = tf.layers.Dense(n)(output)
                    else:
                        output = tf.layers.Dense(n, self.disc_activation)(output)
                        output = tf.layers.Dropout(0.2)(output, training=True)
        return [output, feature]


    def build_enc(self, x):
        # Build encoder
        output = tf.reshape(x, [-1]+self.img_size)#output:[?,28,28,1]
        with tf.name_scope("AAE_Enc"):
            with tf.variable_scope("AAE_Enc_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.enc_filters):
                    output = tf.layers.Conv2D(n, 3, 1, activation=self.enc_activation, padding="SAME")(output) #[?,28,28,32]--->[?,27,27,32]
                    output = tf.layers.MaxPooling2D(2, 1)(output) #[?,27,27,32]--->[?,26,26,32]
                output = tf.layers.Flatten()(output)
                for i, n in enumerate(self.enc_neurons):
                    output = tf.layers.Dense(n, self.enc_activation)(output) #n=200:[?,200];n=100:[?,100]
                cont_output = tf.layers.Dense(self.cont_latent_size)(output) #[?,15]
                cat_output = tf.layers.Dense(self.cat_latent_size, tf.nn.softmax)(output) #[?,16]
        return [cont_output, cat_output]

    def update_gen_weights(self, sess):
        # Copy weights from trainable to non trainable generator
        trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AAE_Dec")
        non_trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AAE_Static_Dec")
        for a, b in zip(trainable_vars, non_trainable_vars):
            sess.run(tf.assign(b, a))

class AAE1(model):
    """
    Adversarial Autoencoder, only one discriminator
    """

    def __init__(self, x, y, cont_latent_size, cat_latent_size, img_size, enc_neurons, enc_filters, dec_neurons, dec_im_shape, dec_filters, disc_neurons):
        # Make variables available to class
        self.cont_latent_size = cont_latent_size
        self.cat_latent_size = cat_latent_size
        self.img_size = img_size
        self.enc_neurons = enc_neurons
        self.enc_filters = enc_filters
        self.dec_neurons = dec_neurons
        self.dec_im_shape = dec_im_shape
        self.dec_filters = dec_filters
        self.disc_neurons = disc_neurons
        self.label = y

        # Define constants
        self.enc_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)
        self.dec_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)
        self.disc_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)

        # Create placeholders
        self.batch_size = tf.placeholder(tf.int32)
        self.repl_batch_size = tf.placeholder(tf.int32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.b_replay = tf.placeholder(tf.bool)

        # Create input based on generative replay
        x = tf.reshape(x, [-1, np.prod(img_size)]) #x: (?,784)matrix
        self.cont_prior = tf.distributions.Normal(tf.zeros(self.cont_latent_size), tf.ones(self.cont_latent_size))
        self.cat_prior = tf.distributions.Categorical(probs=tf.constant(np.ones(self.cat_latent_size)/self.cat_latent_size))
        self.z_cont_static = self.cont_prior.sample(self.repl_batch_size)
        self.z_cat_static = tf.one_hot(self.cat_prior.sample(self.repl_batch_size), self.cat_latent_size)
        self.static_dec_out = self.build_static_dec(tf.concat([self.z_cont_static, self.z_cat_static], axis=1))
        self.input = tf.cond(self.b_replay, lambda: tf.concat([x, self.static_dec_out], axis=0), lambda: x)

        # Build structure
        [self.z_enc_cont, self.z_enc_cat] = self.build_enc(self.input)
        self.z_cont = self.cont_prior.sample(tf.shape(self.input)[0]) #[?,15]
        self.z_cat = tf.one_hot(self.cat_prior.sample(tf.shape(self.input)[0]), self.cat_latent_size)
        self.dec_out = self.build_dec(tf.concat([self.z_enc_cont, self.z_enc_cat], axis=1))

        [disc_real_out, disc_real_feature] = self.build_disc_cont(tf.concat((self.z_enc_cont, self.z_enc_cat), axis=-1))
        [disc_fake_out, disc_fake_feature] = self.build_disc_cont(tf.concat((self.z_cont, self.z_cat), axis=-1))

        loss_disc = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_fake_out), logits=disc_fake_out)),
                            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_real_out), logits=disc_real_out)))
        loss_disc_fool = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_fake_out), logits=disc_fake_out)),
                                tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_real_out), logits=disc_real_out)))
        loss_recon = tf.reduce_mean(tf.square(tf.subtract(self.input, self.dec_out))) #l2 loss
        self.loss = loss_recon
        tf.summary.scalar('loss_recon',loss_recon)
        tf.summary.scalar('loss_disc',loss_disc)
        tf.summary.scalar('loss_disc_fool',loss_disc_fool)
        self.summaries = tf.summary.merge_all()

        # Create optimizer
        self.opt_disc = tf.train.RMSPropOptimizer(self.learning_rate)
        self.opt_recon = tf.train.RMSPropOptimizer(self.learning_rate)
        self.opt_fool = tf.train.RMSPropOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.update_disc = self.opt_disc.minimize(loss_disc, var_list=tf.trainable_variables("AAE_Disc_Cont"))
            self.update_recon = self.opt_recon.minimize(loss_recon, var_list=tf.trainable_variables("AAE_Dec")+tf.trainable_variables("AAE_Enc"))
            self.update_fool = self.opt_fool.minimize(loss_disc_fool, var_list=tf.trainable_variables("AAE_Enc"))

    def build_dec(self, z):
        # Build generator
        output = z
        output = tf.reshape(output, [-1, self.cont_latent_size + self.cat_latent_size]) #[?,31]
        with tf.name_scope("AAE_Dec"):
            with tf.variable_scope("AAE_Dec_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.dec_neurons):
                    output = tf.layers.Dense(n, self.dec_activation)(output)
                output = tf.reshape(output, [-1]+self.dec_im_shape)
                for i, n in enumerate(self.dec_filters):
                    if (i == len(self.dec_filters)-1):
                        output = tf.layers.Conv2DTranspose(n, 3, 1, activation=tf.nn.sigmoid, padding="SAME")(output)
                    else:
                        output = tf.layers.Conv2DTranspose(n, 3, 2, activation=self.dec_activation, padding="SAME")(output)
                output = tf.layers.Flatten()(output)
        return output

    def build_static_dec(self, z):
        # Build generator(using for the generative replay)
        output = tf.reshape(z, [-1, self.cont_latent_size + self.cat_latent_size])
        with tf.name_scope("AAE_Static_Dec"):
            with tf.variable_scope("AAE_Static_Dec_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.dec_neurons):
                    output = tf.layers.Dense(n, self.dec_activation, trainable=False)(output)
                output = tf.reshape(output, [-1]+self.dec_im_shape)
                for i, n in enumerate(self.dec_filters):
                    if (i == len(self.dec_filters)-1):
                        output = tf.layers.Conv2DTranspose(n, 3, 1, activation=tf.nn.sigmoid, padding="SAME", trainable=False)(output)
                    else:
                        output = tf.layers.Conv2DTranspose(n, 3, 2, activation=self.dec_activation, padding="SAME", trainable=False)(output)
                output = tf.layers.Flatten()(output)
        return output

    def build_disc_cont(self, z):
        # Build discriminator
        output = z
        with tf.name_scope("AAE_Disc_Cont"):
            with tf.variable_scope("AAE_Disc_Cont_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.disc_neurons):
                    if (i == len(self.disc_neurons)-1):
                        feature = output
                        output = tf.layers.Dense(n)(output)
                    else:
                        output = tf.layers.Dense(n, self.disc_activation)(output)
                        output = tf.layers.Dropout(0.2)(output, training=True)
        return [output, feature]

    """
    def build_disc_cat(self, z):
        # Build discriminator
        output = z
        with tf.name_scope("AAE_Disc_Cat"):
            with tf.variable_scope("AAE_Disc_Cat_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.disc_neurons):
                    if (i == len(self.disc_neurons)-1):
                        feature = output
                        output = tf.layers.Dense(n)(output)
                    else:
                        output = tf.layers.Dense(n, self.disc_activation)(output)
                        output = tf.layers.Dropout(0.2)(output, training=True)
        return [output, feature]
    """

    def build_enc(self, x):
        # Build encoder
        output = tf.reshape(x, [-1]+self.img_size)#output:[?,28,28,1]
        with tf.name_scope("AAE_Enc"):
            with tf.variable_scope("AAE_Enc_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.enc_filters):
                    output = tf.layers.Conv2D(n, 3, 1, activation=self.enc_activation, padding="SAME")(output) #[?,28,28,32]--->[?,27,27,32]
                    output = tf.layers.MaxPooling2D(2, 1)(output) #[?,27,27,32]--->[?,26,26,32]
                output = tf.layers.Flatten()(output)
                for i, n in enumerate(self.enc_neurons):
                    output = tf.layers.Dense(n, self.enc_activation)(output) #n=200:[?,200];n=100:[?,100]
                cont_output = tf.layers.Dense(self.cont_latent_size)(output) #[?,15]
                cat_output = tf.layers.Dense(self.cat_latent_size, tf.nn.softmax)(output) #[?,16]
                # l2 normalization
                #cont_output = tf.math.l2_normalize(cont_output, axis=1, epsilon=1e-12, name=None, dim=None)
        return [cont_output, cat_output]

    def update_gen_weights(self, sess):
        # Copy weights from trainable to non trainable generator
        trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AAE_Dec")
        non_trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AAE_Static_Dec")
        for a, b in zip(trainable_vars, non_trainable_vars):
            sess.run(tf.assign(b, a))

class AAE2(model):
    """
    Adversarial Autoencoder, only one discriminator
    change optimizer: from RMS to gradient descent with momentum
    """
    def __init__(self, x, y, cont_latent_size, cat_latent_size, img_size, enc_neurons, enc_filters, dec_neurons, dec_im_shape, dec_filters, disc_neurons):
        # Make variables available to class
        self.cont_latent_size = cont_latent_size
        self.cat_latent_size = cat_latent_size
        self.img_size = img_size
        self.enc_neurons = enc_neurons
        self.enc_filters = enc_filters
        self.dec_neurons = dec_neurons
        self.dec_im_shape = dec_im_shape
        self.dec_filters = dec_filters
        self.disc_neurons = disc_neurons
        self.label = y


        # Define constants
        self.enc_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)
        self.dec_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)
        self.disc_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)

        # Create placeholders
        self.batch_size = tf.placeholder(tf.int32)
        self.repl_batch_size = tf.placeholder(tf.int32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.b_replay = tf.placeholder(tf.bool)
        self.momentum = tf.placeholder(tf.float32) #new
        self.momentum_recon = tf.placeholder(tf.float32) #new

        # Create input based on generative replay
        x = tf.reshape(x, [-1, np.prod(img_size)]) #x: (?,784)matrix
        self.cont_prior = tf.distributions.Normal(tf.zeros(self.cont_latent_size), tf.ones(self.cont_latent_size))
        self.cat_prior = tf.distributions.Categorical(probs=tf.constant(np.ones(self.cat_latent_size)/self.cat_latent_size))
        self.z_cont_static = self.cont_prior.sample(self.repl_batch_size)
        self.z_cat_static = tf.one_hot(self.cat_prior.sample(self.repl_batch_size), self.cat_latent_size)
        self.static_dec_out = self.build_static_dec(tf.concat([self.z_cont_static, self.z_cat_static], axis=1))
        self.input = tf.cond(self.b_replay, lambda: tf.concat([x, self.static_dec_out], axis=0), lambda: x)

        # Build structure
        [self.z_enc_cont, self.z_enc_cat] = self.build_enc(self.input)
        self.z_cont = self.cont_prior.sample(tf.shape(self.input)[0]) #[?,15]
        self.z_cat = tf.one_hot(self.cat_prior.sample(tf.shape(self.input)[0]), self.cat_latent_size)
        self.dec_out = self.build_dec(tf.concat([self.z_enc_cont, self.z_enc_cat], axis=1))

        [disc_real_out, disc_real_feature] = self.build_disc_cont(tf.concat((self.z_enc_cont, self.z_enc_cat), axis=-1))
        [disc_fake_out, disc_fake_feature] = self.build_disc_cont(tf.concat((self.z_cont, self.z_cat), axis=-1))

        loss_disc = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_fake_out), logits=disc_fake_out)),
                            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_real_out), logits=disc_real_out)))
        loss_disc_fool = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_fake_out), logits=disc_fake_out)),
                                tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_real_out), logits=disc_real_out)))
        loss_recon = tf.reduce_mean(tf.square(tf.subtract(self.input, self.dec_out))) #l2 loss
        self.loss = loss_recon
        tf.summary.scalar('loss_recon',loss_recon)
        tf.summary.scalar('loss_disc',loss_disc)
        tf.summary.scalar('loss_disc_fool',loss_disc_fool)
        self.summaries = tf.summary.merge_all()

        # Create optimizer
        """
        self.opt_disc = tf.train.RMSPropOptimizer(self.learning_rate)
        self.opt_recon = tf.train.RMSPropOptimizer(self.learning_rate)
        self.opt_fool = tf.train.RMSPropOptimizer(self.learning_rate)
        """
        self.opt_disc = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        self.opt_recon = tf.train.MomentumOptimizer(self.learning_rate, self.momentum_recon)
        self.opt_fool = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.update_disc = self.opt_disc.minimize(loss_disc, var_list=tf.trainable_variables("AAE_Disc_Cont"))
            self.update_recon = self.opt_recon.minimize(loss_recon, var_list=tf.trainable_variables("AAE_Dec")+tf.trainable_variables("AAE_Enc"))
            self.update_fool = self.opt_fool.minimize(loss_disc_fool, var_list=tf.trainable_variables("AAE_Enc"))

    def build_dec(self, z):
        # Build generator
        output = z
        output = tf.reshape(output, [-1, self.cont_latent_size + self.cat_latent_size]) #[?,31]
        with tf.name_scope("AAE_Dec"):
            with tf.variable_scope("AAE_Dec_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.dec_neurons):
                    output = tf.layers.Dense(n, self.dec_activation)(output)
                output = tf.reshape(output, [-1]+self.dec_im_shape)
                for i, n in enumerate(self.dec_filters):
                    if (i == len(self.dec_filters)-1):
                        output = tf.layers.Conv2DTranspose(n, 3, 1, activation=tf.nn.sigmoid, padding="SAME")(output)
                    else:
                        output = tf.layers.Conv2DTranspose(n, 3, 2, activation=self.dec_activation, padding="SAME")(output)
                output = tf.layers.Flatten()(output)
        return output

    def build_static_dec(self, z):
        # Build generator(using for the generative replay)
        output = tf.reshape(z, [-1, self.cont_latent_size + self.cat_latent_size])
        with tf.name_scope("AAE_Static_Dec"):
            with tf.variable_scope("AAE_Static_Dec_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.dec_neurons):
                    output = tf.layers.Dense(n, self.dec_activation, trainable=False)(output)
                output = tf.reshape(output, [-1]+self.dec_im_shape)
                for i, n in enumerate(self.dec_filters):
                    if (i == len(self.dec_filters)-1):
                        output = tf.layers.Conv2DTranspose(n, 3, 1, activation=tf.nn.sigmoid, padding="SAME", trainable=False)(output)
                    else:
                        output = tf.layers.Conv2DTranspose(n, 3, 2, activation=self.dec_activation, padding="SAME", trainable=False)(output)
                output = tf.layers.Flatten()(output)
        return output

    def build_disc_cont(self, z):
        # Build discriminator
        output = z
        with tf.name_scope("AAE_Disc_Cont"):
            with tf.variable_scope("AAE_Disc_Cont_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.disc_neurons):
                    if (i == len(self.disc_neurons)-1):
                        feature = output
                        output = tf.layers.Dense(n)(output)
                    else:
                        output = tf.layers.Dense(n, self.disc_activation)(output)
                        output = tf.layers.Dropout(0.2)(output, training=True)
        return [output, feature]

    """
    def build_disc_cat(self, z):
        # Build discriminator
        output = z
        with tf.name_scope("AAE_Disc_Cat"):
            with tf.variable_scope("AAE_Disc_Cat_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.disc_neurons):
                    if (i == len(self.disc_neurons)-1):
                        feature = output
                        output = tf.layers.Dense(n)(output)
                    else:
                        output = tf.layers.Dense(n, self.disc_activation)(output)
                        output = tf.layers.Dropout(0.2)(output, training=True)
        return [output, feature]
    """

    def build_enc(self, x):
        # Build encoder
        output = tf.reshape(x, [-1]+self.img_size)#output:[?,28,28,1]
        with tf.name_scope("AAE_Enc"):
            with tf.variable_scope("AAE_Enc_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.enc_filters):
                    output = tf.layers.Conv2D(n, 3, 1, activation=self.enc_activation, padding="SAME")(output) #[?,28,28,32]--->[?,27,27,32]
                    output = tf.layers.MaxPooling2D(2, 1)(output) #[?,27,27,32]--->[?,26,26,32]
                output = tf.layers.Flatten()(output)
                for i, n in enumerate(self.enc_neurons):
                    output = tf.layers.Dense(n, self.enc_activation)(output) #n=200:[?,200];n=100:[?,100]
                cont_output = tf.layers.Dense(self.cont_latent_size)(output) #[?,15]
                cat_output = tf.layers.Dense(self.cat_latent_size, tf.nn.softmax)(output) #[?,16]
                # l2 normalization
                #cont_output = tf.math.l2_normalize(cont_output, axis=1, epsilon=1e-12, name=None, dim=None)
        return [cont_output, cat_output]

    def update_gen_weights(self, sess):
        # Copy weights from trainable to non trainable generator
        trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AAE_Dec")
        non_trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AAE_Static_Dec")
        for a, b in zip(trainable_vars, non_trainable_vars):
            sess.run(tf.assign(b, a))

class AAE3(model):
    """
    Adversarial Autoencoder
    based on the model in the paper
    AAE with two discriminators,optimizer is GD with momentum
    """

    def __init__(self, x, y, cont_latent_size, cat_latent_size, img_size, enc_neurons, enc_filters, dec_neurons, dec_im_shape, dec_filters, disc_neurons):
        # Make variables available to class
        self.cont_latent_size = cont_latent_size
        self.cat_latent_size = cat_latent_size
        self.img_size = img_size
        self.enc_neurons = enc_neurons
        self.enc_filters = enc_filters
        self.dec_neurons = dec_neurons
        self.dec_im_shape = dec_im_shape
        self.dec_filters = dec_filters
        self.disc_neurons = disc_neurons
        self.label = y

        # Define constants
        self.enc_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)
        self.dec_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)
        self.disc_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)

        # Create placeholders
        self.batch_size = tf.placeholder(tf.int32)
        self.repl_batch_size = tf.placeholder(tf.int32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.b_replay = tf.placeholder(tf.bool)
        self.momentum_recon = tf.placeholder(tf.float32)
        self.momentum = tf.placeholder(tf.float32)

        # Create input based on generative replay
        x = tf.reshape(x, [-1, np.prod(img_size)]) #x: (?,784)matrix
        self.cont_prior = tf.distributions.Normal(tf.zeros(self.cont_latent_size), tf.ones(self.cont_latent_size))
        self.cat_prior = tf.distributions.Categorical(probs=tf.constant(np.ones(self.cat_latent_size)/self.cat_latent_size))
        self.z_cont_static = self.cont_prior.sample(self.repl_batch_size)
        self.z_cat_static = tf.one_hot(self.cat_prior.sample(self.repl_batch_size), self.cat_latent_size)
        self.static_dec_out = self.build_static_dec(tf.concat([self.z_cont_static, self.z_cat_static], axis=1))
        self.input = tf.cond(self.b_replay, lambda: tf.concat([x, self.static_dec_out], axis=0), lambda: x)

        # Build structure
        [self.z_enc_cont, self.z_enc_cat] = self.build_enc(self.input)
        self.z_cont = self.cont_prior.sample(tf.shape(self.input)[0]) #[?,15]
        self.z_cat = tf.one_hot(self.cat_prior.sample(tf.shape(self.input)[0]), self.cat_latent_size) #[?,16]
        self.dec_out = self.build_dec(tf.concat([self.z_enc_cont, self.z_enc_cat], axis=1)) #tf.concat: ?*31, dec_out:?*784
        [disc_cont_real_out, disc_cont_real_feature] = self.build_disc_cont(self.z_enc_cont)
        [disc_cont_fake_out, disc_cont_fake_feature] = self.build_disc_cont(self.z_cont)
        [disc_cat_real_out, disc_cat_real_feature] = self.build_disc_cat(self.z_enc_cat)
        [disc_cat_fake_out, disc_cat_fake_feature] = self.build_disc_cat(self.z_cat)
        # 1->fake 0->real
        loss_disc_cont = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_cont_fake_out), logits=disc_cont_fake_out)),\
                            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_cont_real_out), logits=disc_cont_real_out)))
        loss_disc_cont_fool = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_cont_real_out), logits=disc_cont_real_out))
        loss_disc_cat = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_cat_fake_out), logits=disc_cat_fake_out)),\
                            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_cat_real_out), logits=disc_cat_real_out)))
        loss_disc_cat_fool = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_cat_real_out), logits=disc_cat_real_out))
        loss_disc = tf.add(loss_disc_cont, loss_disc_cat)
        loss_disc_fool = tf.add(loss_disc_cont_fool, loss_disc_cat_fool)
        loss_recon = tf.reduce_mean(tf.square(tf.subtract(self.input, self.dec_out))) #l2 loss
        self.loss = loss_recon
        tf.summary.scalar('loss_recon',loss_recon)
        tf.summary.scalar('loss_disc',loss_disc)
        tf.summary.scalar('loss_disc_fool',loss_disc_fool)
        self.summaries = tf.summary.merge_all()


        # Create optimizer
        '''
        self.opt_disc = tf.train.RMSPropOptimizer(self.learning_rate)
        self.opt_recon = tf.train.RMSPropOptimizer(self.learning_rate)
        self.opt_fool = tf.train.RMSPropOptimizer(self.learning_rate)
        '''
        self.opt_disc = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        self.opt_recon = tf.train.MomentumOptimizer(self.learning_rate, self.momentum_recon)
        self.opt_fool = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.update_disc = self.opt_disc.minimize(loss_disc, var_list=tf.trainable_variables("AAE_Disc_Cont")+tf.trainable_variables("AAE_Disc_Cat"))
            self.update_recon = self.opt_recon.minimize(loss_recon, var_list=tf.trainable_variables("AAE_Dec")+tf.trainable_variables("AAE_Enc"))
            self.update_fool = self.opt_fool.minimize(loss_disc_fool, var_list=tf.trainable_variables("AAE_Enc"))

    def build_dec(self, z):
        # Build generator
        output = z #[?,31]
        output = tf.reshape(output, [-1, self.cont_latent_size + self.cat_latent_size]) #[?,31]
        with tf.name_scope("AAE_Dec"):
            with tf.variable_scope("AAE_Dec_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.dec_neurons):
                    output = tf.layers.Dense(n, self.dec_activation)(output)
                output = tf.reshape(output, [-1]+self.dec_im_shape)
                for i, n in enumerate(self.dec_filters):
                    if (i == len(self.dec_filters)-1):
                        output = tf.layers.Conv2DTranspose(n, 3, 1, activation=tf.nn.sigmoid, padding="SAME")(output)
                    else:
                        output = tf.layers.Conv2DTranspose(n, 3, 2, activation=self.dec_activation, padding="SAME")(output)
                output = tf.layers.Flatten()(output)
        return output

    def build_static_dec(self, z):
        # Build generator(using for the generative replay)
        output = tf.reshape(z, [-1, self.cont_latent_size + self.cat_latent_size])
        with tf.name_scope("AAE_Static_Dec"):
            with tf.variable_scope("AAE_Static_Dec_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.dec_neurons):
                    output = tf.layers.Dense(n, self.dec_activation, trainable=False)(output)
                output = tf.reshape(output, [-1]+self.dec_im_shape)
                for i, n in enumerate(self.dec_filters):
                    if (i == len(self.dec_filters)-1):
                        output = tf.layers.Conv2DTranspose(n, 3, 1, activation=tf.nn.sigmoid, padding="SAME", trainable=False)(output)
                    else:
                        output = tf.layers.Conv2DTranspose(n, 3, 2, activation=self.dec_activation, padding="SAME", trainable=False)(output)
                output = tf.layers.Flatten()(output)
        return output

    def build_disc_cont(self, z):
        # Build discriminator
        output = z
        with tf.name_scope("AAE_Disc_Cont"):
            with tf.variable_scope("AAE_Disc_Cont_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.disc_neurons):
                    if (i == len(self.disc_neurons)-1):
                        feature = output
                        output = tf.layers.Dense(n)(output)
                    else:
                        output = tf.layers.Dense(n, self.disc_activation)(output)
                        output = tf.layers.Dropout(0.2)(output, training=True)
        return [output, feature]

    def build_disc_cat(self, z):
        # Build discriminator
        output = z
        with tf.name_scope("AAE_Disc_Cat"):
            with tf.variable_scope("AAE_Disc_Cat_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.disc_neurons):
                    if (i == len(self.disc_neurons)-1):
                        feature = output
                        output = tf.layers.Dense(n)(output)
                    else:
                        output = tf.layers.Dense(n, self.disc_activation)(output)
                        output = tf.layers.Dropout(0.2)(output, training=True)
        return [output, feature]


    def build_enc(self, x):
        # Build encoder
        output = tf.reshape(x, [-1]+self.img_size)#output:[?,28,28,1]
        with tf.name_scope("AAE_Enc"):
            with tf.variable_scope("AAE_Enc_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.enc_filters):
                    output = tf.layers.Conv2D(n, 3, 1, activation=self.enc_activation, padding="SAME")(output) #[?,28,28,32]--->[?,27,27,32]
                    output = tf.layers.MaxPooling2D(2, 1)(output) #[?,27,27,32]--->[?,26,26,32]
                output = tf.layers.Flatten()(output)
                for i, n in enumerate(self.enc_neurons):
                    output = tf.layers.Dense(n, self.enc_activation)(output) #n=200:[?,200];n=100:[?,100]
                cont_output = tf.layers.Dense(self.cont_latent_size)(output) #[?,15]
                cat_output = tf.layers.Dense(self.cat_latent_size, tf.nn.softmax)(output) #[?,16]
        return [cont_output, cat_output]

    def update_gen_weights(self, sess):
        # Copy weights from trainable to non trainable generator
        trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AAE_Dec")
        non_trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AAE_Static_Dec")
        for a, b in zip(trainable_vars, non_trainable_vars):
            sess.run(tf.assign(b, a))

class AE_l2(model):
    """
    Autoencoder with fully connected netzwerk
    """

    def __init__(self, x, y, cont_latent_size, img_size, enc_neurons, enc_filters, dec_neurons, dec_im_shape, dec_filters):
        # Make variables available to class
        self.cont_latent_size = cont_latent_size
        self.img_size = img_size
        self.enc_neurons = enc_neurons
        self.enc_filters = enc_filters
        self.dec_neurons = dec_neurons
        self.dec_im_shape = dec_im_shape
        self.dec_filters = dec_filters
        self.label = y

        # Define constants
        self.enc_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)
        self.dec_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)

        # Create placeholders
        self.batch_size = tf.placeholder(tf.int32)
        self.learning_rate = tf.placeholder(tf.float32)

        # Create input based on generative replay
        x = tf.reshape(x, [-1, np.prod(img_size)])
        self.input = x

        # Build structure
        self.z_enc = self.build_enc(self.input)
        self.dec_out = self.build_dec(self.z_enc)
        loss_recon = tf.reduce_mean(tf.square(tf.subtract(self.input, self.dec_out)))
        self.loss = loss_recon
        tf.summary.scalar('loss_recon', loss_recon) #new
        self.summaries = tf.summary.merge_all() #new

        # Create optimizer
        self.opt_recon = tf.train.RMSPropOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.update_recon = self.opt_recon.minimize(loss_recon, var_list=tf.trainable_variables("AE_Dec")+tf.trainable_variables("AE_Enc"))

    def build_dec(self, z):
        # Build decoder
        output = z
        output = tf.reshape(output, [-1, self.cont_latent_size])
        with tf.name_scope("AE_Dec"):
            with tf.variable_scope("AE_Dec_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.dec_neurons):
                    output = tf.layers.Dense(n, self.dec_activation)(output)
                output = tf.reshape(output, [-1]+self.dec_im_shape)
                for i, n in enumerate(self.dec_filters):
                    if (i == len(self.dec_filters)-1):
                        output = tf.layers.Conv2DTranspose(n, 3, 1, activation=tf.nn.sigmoid, padding="SAME")(output)
                    else:
                        output = tf.layers.Conv2DTranspose(n, 3, 2, activation=self.dec_activation, padding="SAME")(output)
                output = tf.layers.Flatten()(output)
        return output

    def build_enc(self, x):
        # Build encoder
        output = tf.reshape(x, [-1]+self.img_size)
        with tf.name_scope("AE_Enc"):
            with tf.variable_scope("AE_Enc_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.enc_filters):
                    output = tf.layers.Conv2D(n, 3, 1, activation=self.enc_activation, padding="SAME")(output)
                    output = tf.layers.MaxPooling2D(2, 1)(output)
                output = tf.layers.Flatten()(output)
                for i, n in enumerate(self.enc_neurons):
                    output = tf.layers.Dense(n, self.enc_activation)(output)
                cont_output = tf.layers.Dense(self.cont_latent_size)(output)
                #l2 normalization
                cont_output = tf.math.l2_normalize(cont_output,axis=1,epsilon=1e-12,name=None,dim=None)
        return cont_output

class AE_l2_GR(model):
    """
    Autoencoder with generative replay
    """

    def __init__(self, x, y, z, cont_latent_size, img_size, enc_neurons, enc_filters, dec_neurons, dec_im_shape, dec_filters):
        # Make variables available to class
        self.cont_latent_size = cont_latent_size
        self.img_size = img_size
        self.enc_neurons = enc_neurons
        self.enc_filters = enc_filters
        self.dec_neurons = dec_neurons
        self.dec_im_shape = dec_im_shape
        self.dec_filters = dec_filters
        self.label = y
        self.replay_z = z

        # Define constants
        self.enc_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)
        self.dec_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)

        # Create placeholders
        self.batch_size  = tf.placeholder(tf.int32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.b_replay = tf.placeholder(tf.bool)

        # Create input based on generative replay
        x = tf.reshape(x, [-1, np.prod(img_size)])
        self.static_dec_out = self.build_static_dec(self.replay_z)
        self.input = tf.cond(self.b_replay,lambda:tf.concat([x,self.static_dec_out],axis=0),lambda:x)

        # Build structure
        self.z_enc = self.build_enc(self.input)
        self.dec_out = self.build_dec(self.z_enc)
        loss_recon = tf.reduce_mean(tf.square(tf.subtract(self.input, self.dec_out)))
        self.loss = loss_recon
        tf.summary.scalar('loss_recon', loss_recon) #new
        self.summaries = tf.summary.merge_all() #new

        # Create optimizer
        self.opt_recon = tf.train.RMSPropOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.update_recon = self.opt_recon.minimize(loss_recon, var_list=tf.trainable_variables("AE_Dec")+tf.trainable_variables("AE_Enc"))

    def build_dec(self, z):
        # Build decoder
        output = z
        output = tf.reshape(output, [-1, self.cont_latent_size])
        with tf.name_scope("AE_Dec"):
            with tf.variable_scope("AE_Dec_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.dec_neurons):
                    output = tf.layers.Dense(n, self.dec_activation)(output)
                output = tf.reshape(output, [-1]+self.dec_im_shape)
                for i, n in enumerate(self.dec_filters):
                    if (i == len(self.dec_filters)-1):
                        output = tf.layers.Conv2DTranspose(n, 3, 1, activation=tf.nn.sigmoid, padding="SAME")(output)
                    else:
                        output = tf.layers.Conv2DTranspose(n, 3, 2, activation=self.dec_activation, padding="SAME")(output)
                output = tf.layers.Flatten()(output)
        return output

    def build_enc(self, x):
        # Build encoder
        output = tf.reshape(x, [-1]+self.img_size)
        with tf.name_scope("AE_Enc"):
            with tf.variable_scope("AE_Enc_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.enc_filters):
                    output = tf.layers.Conv2D(n, 3, 1, activation=self.enc_activation, padding="SAME")(output)
                    output = tf.layers.MaxPooling2D(2, 1)(output)
                output = tf.layers.Flatten()(output)
                for i, n in enumerate(self.enc_neurons):
                    output = tf.layers.Dense(n, self.enc_activation)(output)
                cont_output = tf.layers.Dense(self.cont_latent_size)(output)
                #l2 normalization
                cont_output = tf.math.l2_normalize(cont_output,axis=1,epsilon=1e-12,name=None,dim=None)
        return cont_output

    def build_static_dec(self, z):
        # Build generator(using for the generative replay)
        output = z
        output = tf.reshape(output, [-1, self.cont_latent_size])
        with tf.name_scope("AE_Static_Dec"):
            with tf.variable_scope("AE_Static_Dec_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.dec_neurons):
                    output = tf.layers.Dense(n, self.dec_activation, trainable=False)(output)
                output = tf.reshape(output, [-1]+self.dec_im_shape)
                for i, n in enumerate(self.dec_filters):
                    if (i == len(self.dec_filters)-1):
                        output = tf.layers.Conv2DTranspose(n, 3, 1, activation=tf.nn.sigmoid, padding="SAME", trainable=False)(output)
                    else:
                        output = tf.layers.Conv2DTranspose(n, 3, 2, activation=self.dec_activation, padding="SAME", trainable=False)(output)
                output = tf.layers.Flatten()(output)
        return output

    def update_gen_weights(self, sess):
        # Copy weights from trainable to non trainable generator
        trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AE_Dec")
        non_trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AE_Static_Dec")
        for a, b in zip(trainable_vars, non_trainable_vars):
            sess.run(tf.assign(b, a))

class CAE_l2(model):
    """
    Autoencoder with convolutional netzwerk
    """

    def __init__(self, x, y, cont_latent_size, img_size, enc_neurons, enc_filters, dec_neurons, dec_im_shape, dec_filters):
        # Make variables available to class
        self.cont_latent_size = cont_latent_size
        self.img_size = img_size
        self.enc_neurons = enc_neurons
        self.enc_filters = enc_filters
        self.dec_neurons = dec_neurons
        self.dec_im_shape = dec_im_shape
        self.dec_filters = dec_filters
        self.label = y

        # Define constants
        self.enc_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)
        self.dec_activation = lambda x: tf.nn.leaky_relu(tf.layers.BatchNormalization(center=True, scale=True)(x, training=True), alpha=0.1)

        # Create placeholders
        self.batch_size = tf.placeholder(tf.int32)
        self.learning_rate = tf.placeholder(tf.float32)

        # Create input based on generative replay
        x = tf.reshape(x, [-1, np.prod(img_size)])
        self.input = x

        # Build structure
        self.z_enc = self.build_enc(self.input)
        self.dec_out = self.build_dec(self.z_enc)
        loss_recon = tf.reduce_mean(tf.square(tf.subtract(self.input, self.dec_out)))
        self.loss = loss_recon
        tf.summary.scalar('loss_recon', loss_recon) #new
        self.summaries = tf.summary.merge_all() #new

        # Create optimizer
        opt_recon = tf.train.RMSPropOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.update_recon = opt_recon.minimize(loss_recon, var_list=tf.trainable_variables("AE_Dec")+tf.trainable_variables("AE_Enc"))

    def build_dec(self, z):
        # Build decoder
        output = z
        output = tf.reshape(output, [-1, self.cont_latent_size])
        with tf.name_scope("AE_Dec"):
            with tf.variable_scope("AE_Dec_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.dec_neurons):
                    output = tf.layers.Dense(n, self.dec_activation)(output)
                output = tf.reshape(output, [-1]+self.dec_im_shape)
                for i, n in enumerate(self.dec_filters):
                    if (i == len(self.dec_filters)-1):
                        output = tf.layers.Conv2DTranspose(n, 5, 2, activation=tf.nn.sigmoid, padding="SAME")(output)
                    else:
                        if (i == 0):
                            output = tf.layers.Conv2DTranspose(n, 3, 2, activation=self.dec_activation, padding="VALID")(output)
                        else:
                            output = tf.layers.Conv2DTranspose(n, 5, 2, activation=self.dec_activation, padding="SAME")(output)
                output = tf.layers.Flatten()(output)
        return output

    def build_enc(self, x):
        # Build encoder
        output = tf.reshape(x, [-1]+self.img_size)
        with tf.name_scope("AE_Enc"):
            with tf.variable_scope("AE_Enc_Vars", reuse=tf.AUTO_REUSE):
                for i, n in enumerate(self.enc_filters):
                    if (i == len(self.enc_filters) - 1):
                        output = tf.layers.Conv2D(n, 3, 2, activation=self.enc_activation, padding="VALID")(output)
                    else:
                        output = tf.layers.Conv2D(n, 5, 2, activation=self.enc_activation, padding="SAME")(output)
                output = tf.layers.Flatten()(output)
                for i, n in enumerate(self.enc_neurons):
                    output = tf.layers.Dense(n, self.enc_activation)(output)
                cont_output = tf.layers.Dense(self.cont_latent_size)(output)
                #l2 normalization
                cont_output = tf.math.l2_normalize(cont_output,axis=1,epsilon=1e-12,name=None,dim=None)
        return cont_output

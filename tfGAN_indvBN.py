import numpy as np
import tensorflow as tf
from time import time
from os import sep, getcwd
from os.path import join
from glob import glob
import re


BATCH_NORM_DECAY = 0.999
BATCH_RENORM = False


def batcher(n_data, batch_size):
    for ind in range(0, n_data - batch_size + 1, batch_size):
        yield ind, np.minimum(batch_size, n_data - ind)


def balanced_rand(labels, batchsize):
    # get info
    nlabels = labels.shape[1]
    full_sets = int(batchsize / nlabels)
    remainder = np.remainder(batchsize, nlabels)
    # sample
    index = np.empty(batchsize, dtype=int)
    for lab in range(nlabels):
        index[full_sets * lab:full_sets * (lab + 1)] = \
            np.random.choice(labels.shape[0], full_sets, p=labels[:, lab] / labels[:, lab].sum())
    if remainder:
        index[-remainder:] = np.random.randint(0, labels.shape[0], remainder)
    # return
    return index


def shuffle_data(data, label=None, nitems=None):
    if data.__class__ is list or label:
        if nitems is None:
            if data.__class__ is list:
                nitems = len(data[0])
                ind = np.random.permutation(nitems)
                data = [x[ind] for x in data]
            else:
                nitems = len(data)
                ind = np.random.permutation(nitems)
                data = data[ind]
        if label is not None:
            label = label[ind]
            return data, label
    else:
        np.random.shuffle(data)
    return data


def lrelu(tensor, name=None):
    return tf.maximum(tensor, 0.2 * tensor, name=name)


def init_normal():
    return tf.random_normal_initializer(0, 0.02, None, tf.float32)


def safe_log(logit, name='safe_log'):
    return tf.log(tf.where(tf.equal(logit, 0.), tf.ones_like(logit), logit), name=name)


class NanInfException (Exception):
    pass


class EarlyStoppingException (Exception):
    pass


class GANBase (object):

    name = 'GANBase'

    def __init__(self, n_extra_generator_layers=0, n_extra_discriminator_layers=0, mask=None, use_batch_norm_G=True,
                 use_batch_norm_D=False, name=None, log_and_save=True, seed=None, early_stopping=False):
        # parameters
        self.n_noise = 100
        self.n_pixel = 32
        self.n_channel = 1
        self.n_filtd = 128
        if mask.dtype is not bool:
            self.mask = np.ones(self.n_pixel * self.n_pixel, dtype=bool)
            self.mask[mask] = False
        else:
            self.mask = tf.constant(mask.reshape(), dtype=tf.float32, name='mask')
        self.mask = tf.constant(self.mask.reshape(1, self.n_pixel, self.n_pixel, 1), dtype=tf.float32, name='mask')
        self.batch_norm_G = use_batch_norm_G
        self.batch_norm_D = use_batch_norm_D
        if seed is None:
            seed = np.random.randint(int(1e8))
        self.seed = seed
        self.n_extra_generator_layers = n_extra_generator_layers
        self.n_extra_discriminator_layers = n_extra_discriminator_layers
        self.log_and_save = log_and_save
        self.early_stopping = early_stopping
        self.early_stopping_best = np.inf
        self.early_stopping_counter = 0
        self.filename = self.name
        if name is not None:
            self.name += '_' + name
        # if self.debug:
        #     self.name += '_debug'
        self.path = getcwd() + sep + 'output' + sep + self.filename + sep

        # network variables
        self.batch_ind = tf.placeholder(tf.int32, 0, 'batch_ind')
        self.batch_size = tf.placeholder(tf.int32, 0, 'batch_size')
        # self.training = tf.placeholder(tf.bool, 0, 'training')
        self.input_x = tf.placeholder(tf.float32, (None, self.n_pixel, self.n_pixel, self.n_channel), 'image')
        self.input_n = tf.placeholder(tf.float32, (None, self.n_noise), 'noise')
        # self.input_x = tf.Variable(self.input_x_ph, trainable=False, collections=[])
        # self.input_n = tf.Variable(self.input_n_ph, trainable=False, collections=[])

        # logging'
        self.saver = None
        self.writer_train = None
        self.writer_test = None

        # etc
        self.session = None

    def _build_generator(self, tensor=None, training=False, batch_norm=None):
        assert self.n_pixel % 16 == 0, "isize has to be a multiple of 16"
        nfilt = 2000
        csize = 4
        if tensor is None:
            tensor = self.input_n
        if batch_norm is None:
            batch_norm = self.batch_norm_G
        if batch_norm:
            def bn(x, name=None):
                return tf.contrib.layers.batch_norm(x, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        with tf.variable_scope('generator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # initial layer
            with tf.variable_scope('initial.{0}-{1}'.format(self.n_noise, nfilt)):
                tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tf.reshape(tensor, [-1, 1, 1, self.n_noise]),
                                                                  nfilt, 4, 2, 'valid', use_bias=not batch_norm,
                                                                  kernel_initializer=init_normal(),
                                                                  name='conv'), name='bn'))

            # upscaling layers
            while csize < self.n_pixel/2:
                with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt/2)):
                    tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt/2, 4, 2, 'same',
                                                                      use_bias=not batch_norm,
                                                                      kernel_initializer=init_normal(),
                                                                      name='conv'), name='bn'))
                csize *= 2
                nfilt /= 2

            # extra layers
            for it in range(self.n_extra_generator_layers):
                with tf.variable_scope('extra-{0}.{1}'.format(it, nfilt)):
                    tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                                      kernel_initializer=init_normal(),
                                                                      name='conv'), name='bn'))

            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(nfilt, self.n_channel)):
                tensor = tf.layers.conv2d_transpose(tensor, self.n_channel, 4, 2, 'same', activation=tf.tanh,
                                                    kernel_initializer=init_normal(),
                                                    name='conv')

            # mask layer
            return tf.identity(tensor * self.mask, name='generated_image')

    def _build_discriminator_base(self, tensor=None, training=False, batch_norm=None):
        nfilt = self.n_filtd
        if tensor is None:
            tensor = self.input_x
        if batch_norm is None:
            batch_norm = self.batch_norm_D
        if batch_norm:
            def bn(tensor, name=None):
                return tf.contrib.layers.batch_norm(tensor, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        # initial layer
        with tf.variable_scope('initial.{0}-{1}'.format(self.n_channel, nfilt)):
            tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 4, 2, 'same', use_bias=not batch_norm,
                                               kernel_initializer=init_normal(),
                                               name='conv'), name='bn'))
        # nfilt /= 2
        csize = self.n_pixel / 2

        # extra layers
        for it in range(self.n_extra_discriminator_layers):
            with tf.variable_scope('extra-{0}.{1}'.format(it, nfilt)):
                tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                   kernel_initializer=init_normal(),
                                                   name='conv'), name='bn'))

        # downscaling layers
        while csize > 4:
            with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt * 2)):
                tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt * 2, 4, 2, 'same', use_bias=not batch_norm,
                                                   kernel_initializer=init_normal(),
                                                   name='conv'), name='bn'))
            nfilt *= 2
            csize /= 2

        return tensor

    def _build_loss(self, label_strength=1.):
        raise NotImplementedError

    def _start_logging_and_saving(self, sess, log=True, save=True):
        if self.log_and_save and (log or save):
            # saver to save model
            if save:
                self.saver = tf.train.Saver(name='checkpoint', max_to_keep=2)
            # saver for early stopping
            if self.early_stopping:
                self.early_stopping_saver = tf.train.Saver(name='early_stopping', max_to_keep=2)
            # summary writer
            if log:
                self.writer_train = tf.summary.FileWriter(join(self.path, self.name, 'train'), sess.graph)
                self.writer_test = tf.summary.FileWriter(join(self.path, self.name, 'test'), sess.graph)

            print 'Saving to ' + self.path

    def _log(self, summary, counter=None, test=False):
        if self.log_and_save:
            if test:
                self.writer_test.add_summary(summary, counter)
            else:
                self.writer_train.add_summary(summary, counter)

    def _save(self, session, counter=None):
        if self.log_and_save:
            self.saver.save(session, join(self.path, self.name, self.name + '.ckpt'), counter)

    def _restore(self, session):
        if self.log_and_save:
            self.saver.restore(session, tf.train.latest_checkpoint(join(self.path, self.name)))

    def load(self, path=None):
        self._build_loss()
        self.session = tf.Session()
        self.session.as_default()
        self._start_logging_and_saving(None, log=False)
        if self.early_stopping:
            if path is None:
                max_ckpt = max([int(x[ind1 + 5:ind2])
                                for x in glob(join(self.path, self.name, self.name + '_early_stopping.ckpt*.meta'))
                                for ind1, ind2 in re.search('ckpt-\d+', x).regs])
                path = join(self.path, self.name, self.name + '_early_stopping.ckpt-' + str(max_ckpt))
            self.early_stopping_saver.restore(self.session, path)
        else:
            if path is None:
                max_ckpt = max([int(x[ind1 + 5:ind2])
                                for x in glob(join(self.path, self.name, self.name + '.ckpt*.meta'))
                                for ind1, ind2 in re.search('ckpt-\d+', x).regs])
                path = join(self.path, self.name, self.name + '.ckpt-' + str(max_ckpt))
            self.saver.restore(self.session, path)

    def _early_stopping(self, new_val, session, counter, feed_dict=None):
        if self.early_stopping:
            if self.early_stopping_best >= new_val:
                self.early_stopping_best = new_val
                self.early_stopping_counter = counter
                self.early_stopping_saver.save(session,
                                               join(self.path, self.name, self.name + '_early_stopping.ckpt'),
                                               counter)
            elif counter - self.early_stopping_counter >= 5000:
                self.inference_2_matfile()
                if feed_dict is not None:
                    self.make_test_vals(session, feed_dict)
                raise EarlyStoppingException

    def _train(self):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        try:
            self._train(*args, **kwargs)
        except EarlyStoppingException:
            print('Early stopping condition met. Stopping training. Use early stopping save as best model.')


class DCGAN (GANBase):

    name = 'DCGAN'

    def _build_discriminator(self, tensor=None, training=False):

        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_discriminator_base(tensor, training)

            # final layer
            d_out = 2
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], d_out)):
                out_logits = tf.reshape(tf.layers.conv2d(tensor, d_out, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, d_out])

        return tf.nn.softmax(out_logits), out_logits

    def _build_loss(self, label_strength=1., training=False):
        fake = self._build_generator(training=training) #tf.random_normal((self.batch_size, self.n_noise)))

        fake_label, fake_logits = self._build_discriminator(fake, training=training)
        real_label, real_logits = self._build_discriminator(training=training)
        label_goal = tf.concat((tf.ones((tf.shape(fake_logits)[0], 1)), tf.zeros((tf.shape(fake_logits)[0], 1))), 1)
        label_smooth = tf.concat((label_strength * tf.ones((tf.shape(fake_logits)[0], 1)),
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

        # generator
        # self.lossG =
        # -safe_log(1 - fake_label[:, -1]) or -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=1-label_goal))
        # -safe_log(fake_label[:, 0]) (better) or tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=label_goal) (best)
        lossG = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=label_goal))

        # discriminator
        lossD_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits, labels=label_smooth))
        lossD_g = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=1-label_goal))
        lossD = lossD_d + lossD_g

        # summaries
        if training:
            with tf.name_scope('training_metrics'):
                tf.summary.image('fake', fake)
                tf.summary.image('real', self.input_x)
                tf.summary.histogram('D_fake', fake_label[:, -1])
                tf.summary.histogram('D_real', real_label[:, -1])
                tf.summary.scalar('lossG', lossG)
                tf.summary.scalar('lossD_d', lossD_d)
                tf.summary.scalar('lossD_g', lossD_g)
                tf.summary.scalar('lossD', lossD)
                tf.summary.scalar('loss', lossG + lossD)
        else:
            with tf.name_scope('evaluation_metrics'):
                tf.summary.image('fake', fake)
                tf.summary.histogram('D_fake', fake_label[:, -1])
                tf.summary.scalar('lossG', lossG)
                tf.summary.scalar('lossD_d', lossD_d)
                tf.summary.scalar('lossD_g', lossD_g)
                tf.summary.scalar('lossD', lossD)
                tf.summary.scalar('loss', lossG + lossD)

        return lossG, lossD

    def train(self, trainx, testx, n_epochs=25, n_batch=128, learning_rate=2e-4, label_strength=1.):

        # handle data
        n_train = trainx.shape[0]
        n_test = testx.shape[0]
        # train = tf.constant(trainx, name='train')
        # test = tf.constant(testx, name='test')
        # dataset = tf.contrib.data.Dataset.from_tensor_slices(self.input_x)
        # iterator = dataset.make_initializable_iterator()
        # train = tf.contrib.data.Dataset.from_tensor_slices(trainx)
        # train = tf.contrib.data.Dataset.from_tensor_slices(testx)
        # iterator_train = train.make_initializable_iterator()

        # setup learning
        # train_batch = tf.train.shuffle_batch([train], n_batch, 50000, 10000, 2,
        #                                      enqueue_many=True, allow_smaller_final_batch=True, name='batch')
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD = self._build_loss(label_strength=label_strength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            adamG = tf.contrib.layers.optimize_loss(loss=lossG,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optG',
                                                    variables=tvarsG)
            adamD = tf.contrib.layers.optimize_loss(loss=lossD,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optD',
                                                    variables=tvarsD)

        # summary
        merged_summary = tf.summary.merge_all()

        # start session
        with tf.Session() as sess:
            # initialize variables
            sess.run(tf.global_variables_initializer())
            # sess.run(self.input_x.initializer, {self.input_x_ph: trainx})
            # sess.run(self.input_x.initializer, {self.input_x_ph: trainx})

            # train
            self._start_logging_and_saving(sess)
            for epoch in range(n_epochs):
                # train on epoch
                start = time()
                n, lg, ld = 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_train, n_batch):
                    n += n_batch_actual
                    # discriminator
                    temp = sess.run(adamD,
                                    {self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                     self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    ld += temp * n_batch_actual
                    # self._log(summary, step)
                    # print 'epoch {:d}/{:d} (part {:d}D):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds'\
                    #     .format(epoch, n_epochs, n, (lg + ld)/n, lg/n, ld/n, int(time() - start))
                    # generator
                    temp, summary, step = sess.run([adamG, merged_summary, global_step],
                                                   {self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                                    self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lg += temp * n_batch_actual
                    # self._log(summary, step)
                    # print 'epoch {:d}/{:d} (part {:d}G):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds'\
                    #     .format(epoch, n_epochs, n, (lg + ld)/n, lg/n, ld/n, int(time() - start))
                    self._log(summary, step)
                    print 'epoch {:d}/{:d} (part {:d}):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                        .format(epoch + 1, n_epochs, n, (lg + ld) / n, lg / n, ld / n, int(time() - start))
                    if n % (100 * n_batch) == 0:
                        # evaluate
                        ne, lge, lde = 0, 0, 0
                        for batch_index, n_batch_actual in batcher(n_test, n_batch):
                            ne += n_batch_actual
                            out = sess.run([evalG, evalD],
                                           {self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                            self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                            lge += out[0] * n_batch_actual
                            lde += out[1] * n_batch_actual
                print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                    .format(epoch + 1, n_epochs, (lge + lde) / ne, lge / ne, lde / ne, int(time() - start))
                # save after each epoch
                self._save(sess, step)

                # evaluate
                ne, lge, lde = 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                    ne += n_batch_actual
                    out = sess.run([evalG, evalD],
                                   {self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                    self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lge += out[0] * n_batch_actual
                    lde += out[1] * n_batch_actual
                print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                    .format(epoch + 1, n_epochs, (lge + lde) / ne, lge / ne, lde / ne, int(time() - start))


class NoisyDCGAN (DCGAN):

    name = 'NoisyDCGAN'

    def _build_discriminator(self, tensor=None, training=False):
        # get tensor
        if tensor is None:
            tensor = self.input_x

        with tf.variable_scope('input_noise') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # add noise
            noise_std = tf.identity(0.05 + tf.train.exponential_decay(0.45, tf.train.get_or_create_global_step(),
                                                                      25., 0.96, staircase=False), name='noise_std')
            tf.summary.scalar('noise_std', noise_std)
            tensor = tensor + tf.random_normal(tf.shape(tensor), stddev=noise_std)

        # build discriminator
        return super(NoisyDCGAN, self)._build_discriminator(tensor=tensor, training=training)


class SSGAN (GANBase):

    name = 'SSGAN'

    def __init__(self, n_y, n_extra_generator_layers=0, n_extra_discriminator_layers=0, mask=None,
                 use_batch_norm_G=True, use_batch_norm_D=False,
                 name=None, seed=np.random.randint(int(1e8)), log_and_save=True, debug=False):

        # label variables
        self.n_y = n_y
        self.input_y = tf.placeholder(tf.float32, (None, self.n_y), 'label')
        self.input_x_l = tf.placeholder(tf.float32, (None, 32, 32, 1), 'labeled_image')

        # init
        # if not hasattr(self, 'name'):
        #     self.name = 'SSGAN'
        super(SSGAN, self).__init__(n_extra_generator_layers=n_extra_generator_layers,
                                    n_extra_discriminator_layers=n_extra_discriminator_layers,
                                    mask=mask, use_batch_norm_G=use_batch_norm_G, use_batch_norm_D=use_batch_norm_D,
                                    name=name, log_and_save=log_and_save, seed=seed, debug=debug)

    def _build_discriminator(self, tensor=None, training=False):

        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_discriminator_base(tensor, training)

            # final layer
            d_out = 1 + self.n_y
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], d_out)):
                out_logits = tf.reshape(tf.layers.conv2d(tensor, d_out, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, d_out])

        return tf.nn.softmax(out_logits), out_logits

    def _build_loss(self, label_strength=1., training=False):

        fake = self._build_generator(training=training) #tf.random_normal((self.batch_size, self.n_noise)))

        fake_label, fake_logits = self._build_discriminator(fake, training=training)
        real_label_u, real_logits_u = self._build_discriminator(training=training)
        real_label_l, real_logits_l = self._build_discriminator(self.input_x_l, training=training)
        label_smooth = tf.concat((label_strength * self.input_y,
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

        # generator
        lossG = tf.reduce_mean(-safe_log(tf.reduce_sum(fake_label[:, :-1], 1)))

        # discriminator
        lossD_d_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits_l, labels=label_smooth))
        lossD_d_u = tf.reduce_mean(-safe_log(tf.reduce_sum(real_label_u[:, :-1], 1)))
        lossD_g = tf.reduce_mean(-safe_log(fake_label[:, -1]))
        lossD = lossD_d_l + lossD_d_u + lossD_g

        # summaries
        if training:
            # generated/real images
            tf.summary.image('fake', fake)
            tf.summary.image('real', self.input_x)
            # classifier performance
            pred = tf.argmax(real_label_l[:, :-1] / (1 - real_label_l[:, -1, None]), 1)
            tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
            tf.summary.image('confusion_matrix', tf.reshape(tf.confusion_matrix(
                tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16), [1, self.n_y, self.n_y, 1]))
            # discriminator performance
            tf.summary.histogram('D_fake', fake_label[:, -1])
            tf.summary.histogram('D_real', real_label_l[:, -1])
            # GAN loss
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d_l', lossD_d_l)
            tf.summary.scalar('lossD_d_u', lossD_d_u)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('loss', lossG + lossD)

        return lossG, lossD

    def train(self, trainx_u, trainx_l, trainy, testx, testy, n_epochs=25, n_batch=128, learning_rate=2e-4, label_strength=1.):

        # handle data
        n_train_u = trainx_u.shape[0]
        n_train_l = trainx_l.shape[0]
        n_test = testx.shape[0]

        # setup learning
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD = self._build_loss(label_strength=label_strength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            adamG = tf.contrib.layers.optimize_loss(loss=lossG,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optG',
                                                    variables=tvarsG)
            adamD = tf.contrib.layers.optimize_loss(loss=lossD,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optD',
                                                    variables=tvarsD)

        # summary
        merged_summary = tf.summary.merge_all()

        # start session
        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())

            # if self.debug:
            #     from tensorflow.python import debug as tf_debug
            #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            #     sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            # train
            self._start_logging_and_saving(sess)
            for epoch in range(n_epochs):
                # train on epoch
                start = time()
                step = 0
                n, lg, ld = 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_train_u, n_batch):
                    n += n_batch_actual
                    # discriminator
                    randind = np.random.choice(n_train_l, n_batch_actual)
                    _, summary, temp = sess.run([adamD, merged_summary, lossD],
                                                {self.input_x: trainx_u[batch_index:batch_index + n_batch_actual],
                                                 self.input_x_l: trainx_l[randind],
                                                 self.input_y: trainy[randind],
                                                 self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    ld += temp * n_batch_actual
                    # generator
                    _, summary, step, temp = sess.run([adamG, merged_summary, global_step, lossG],
                                                      {self.input_x: trainx_u[batch_index:batch_index + n_batch_actual],
                                                       self.input_x_l: trainx_l[randind],
                                                       self.input_y: trainy[randind],
                                                       self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lg += temp * n_batch_actual
                    if n % (1 * n_batch) == 0:
                        self._log(summary, step)
                        print 'epoch {:d}/{:d} (part {:d}):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                            .format(epoch + 1, n_epochs, n, (lg + ld) / n, lg / n, ld / n, int(time() - start))
                    if (n % 100 * n_batch) == 0:
                        # evaluate
                        m, lge, lde = 0, 0, 0
                        for batch_index, n_batch_actual in batcher(n_test, n_batch):
                            m += n_batch_actual
                            out = sess.run([evalG, evalD],
                                           {self.input_x: testx[batch_index:batch_index + n_batch_actual],
                                            self.input_x_l: testx[batch_index:batch_index + n_batch_actual],
                                            self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                            self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                            lge += out[0] * n_batch_actual
                            lde += out[1] * n_batch_actual
                        print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                            .format(epoch + 1, n_epochs, (lge + lde) / m, lge / m, lde / m, int(time() - start))
                        self._save(sess, step)
                # save after each epoch
                self._save(sess, step)

                # evaluate
                n, lge, lde = 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                    n += n_batch_actual
                    out = sess.run([evalG, evalD],
                                   {self.input_x: testx[batch_index:batch_index + n_batch_actual],
                                    self.input_x_l: testx[batch_index:batch_index + n_batch_actual],
                                    self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                    self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lge += out[0] * n_batch_actual
                    lde += out[1] * n_batch_actual
                print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                    .format(epoch + 1, n_epochs, (lge + lde) / n, lge / n, lde / n, int(time() - start))


class VEEGAN (SSGAN):

    name = 'VEEGAN'

    def _build_encoder(self, tensor=None, training=False):

        with tf.variable_scope('encoder') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_discriminator_base(tensor, training)

            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], self.n_noise)):
                tensor = tf.reshape(tf.layers.conv2d(tensor, self.n_noise, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                     name='conv'), [-1, self.n_noise])

        return tensor

    def _build_discriminator(self, tensor=None, encoding_tensor=None, training=False):

        if encoding_tensor is None:
            encoding_tensor = self._build_encoder(training=training)

        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_discriminator_base(tensor, training)

            # concatenate encoding
            tensor = tf.contrib.layers.flatten(tensor)
            tensor = tf.concat((tensor, encoding_tensor), axis=1)
            nfilt = tensor.shape[-1]

            # # dense layer
            # with tf.variable_scope('dense.{0}-{1}'.format(nfilt, nfilt)):
            #     tensor = (tf.layers.dense(tensor, nfilt, activation=lrelu, kernel_initializer=init_normal(),
            #                                 use_bias=not self.batch_norm, name='dense'))

            # final layers
            with tf.variable_scope('final.{0}-{1}'.format(nfilt, 1 + self.n_y)):
                out_logits = tf.layers.dense(tensor, 1 + self.n_y, kernel_initializer=init_normal(), name='dense')

        return tf.nn.softmax(out_logits), out_logits

    def _build_loss(self, label_strength=1., training=False):
        # networks
        fake = self._build_generator(training=training)
        fake_encoding = self._build_encoder(fake, training=training)
        fake_label, fake_logits = self._build_discriminator(fake, self.input_n, training=training)
        real_label_u, real_logits_u = self._build_discriminator(training=training)
        real_label_l, real_logits_l = self._build_discriminator(self.input_x_l, training=training)

        # labels
        label_smooth = tf.concat((label_strength * self.input_y,
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

        # encoder loss
        lossF = tf.reduce_mean(tf.squared_difference(self.input_n, fake_encoding))

        # generator loss
        lossG = tf.reduce_mean(-safe_log(fake_label[:, 0])) + lossF

        # discriminator loss
        lossD_d_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits_l, labels=label_smooth))
        lossD_d_u = tf.reduce_mean(-safe_log(real_label_u[:, 0]))
        lossD_g = tf.reduce_mean(-safe_log(fake_label[:, -1]))
        lossD = lossD_d_l + lossD_d_u + lossD_g

        # summaries
        if training:
            # generated/real images
            tf.summary.image('fake', fake)
            tf.summary.image('real', self.input_x)
            # classifier performance
            pred = tf.argmax(real_label_l[:, :-1] / (1 - real_label_l[:, -1, None]), 1)
            tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
            tf.summary.image('confusion_matrix', tf.reshape(tf.confusion_matrix(
                tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16), [1, self.n_y, self.n_y, 1]))
            # discriminator performance
            tf.summary.histogram('D_fake', fake_label[:, -1])
            tf.summary.histogram('D_real', real_label_l[:, -1])
            # GAN loss
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d_l', lossD_d_l)
            tf.summary.scalar('lossD_d_u', lossD_d_u)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('lossF', lossF)
            tf.summary.scalar('loss', lossG + lossD + lossF)

        return lossG, lossD, lossF

    def train(self, trainx_u, trainx_l, trainy, testx, testy, n_epochs=25, n_batch=128, learning_rate=2e-4, label_strength=1.):

        # handle data
        n_train_u = trainx_u.shape[0]
        n_train_l = trainx_l.shape[0]
        n_test = testx.shape[0]

        # setup learning
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD, lossF = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD, evalF = self._build_loss(label_strength=label_strength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        tvarsF = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            adamG = tf.contrib.layers.optimize_loss(loss=lossG,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optG',
                                                    variables=tvarsG)
            adamD = tf.contrib.layers.optimize_loss(loss=lossD,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optD',
                                                    variables=tvarsD)
            adamF = tf.contrib.layers.optimize_loss(loss=lossF,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optF',
                                                    variables=tvarsF)

        # summary
        merged_summary = tf.summary.merge_all()

        # start session
        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())

            # if self.debug:
            #     from tensorflow.python import debug as tf_debug
            #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            #     sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            # train
            self._start_logging_and_saving(sess)
            for epoch in range(n_epochs):
                # train on epoch
                start = time()
                step = 0
                n, lg, ld, lf = 0, 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_train_u, n_batch):
                    n += n_batch_actual
                    # discriminator
                    randind = np.random.choice(n_train_l, n_batch_actual)
                    _, temp = sess.run([adamD, lossD],
                                       {self.input_x: trainx_u[batch_index:batch_index + n_batch_actual],
                                        self.input_x_l: trainx_l[randind],
                                        self.input_y: trainy[randind],
                                        self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    ld += temp * n_batch_actual
                    # encoder
                    _, temp = sess.run([adamF, lossF],
                                       {self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lf += temp * n_batch_actual
                    # generator
                    _, summary, step, temp = sess.run([adamG, merged_summary, global_step, lossG],
                                                      {self.input_x: trainx_u[batch_index:batch_index + n_batch_actual],
                                                       self.input_x_l: trainx_l[randind],
                                                       self.input_y: trainy[randind],
                                                       self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lg += temp * n_batch_actual
                    if n % (1 * n_batch) == 0:
                        self._log(summary, step)
                        print 'epoch {:d}/{:d} (part {:d}):  training loss: {:f} (G: {:f}  D: {:f}  F: {:f})  time: {:d} seconds' \
                            .format(epoch + 1, n_epochs, n, (lg + ld) / n, lg / n, ld / n, lf / n, int(time() - start))
                # save after each epoch
                self._save(sess, step)

                # evaluate
                n, lge, lde, lfe = 0, 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                    n += n_batch_actual
                    out = sess.run([evalG, evalD, evalF, merged_summary],
                                   {self.input_x: testx[batch_index:batch_index + n_batch_actual],
                                    self.input_x_l: testx[batch_index:batch_index + n_batch_actual],
                                    self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                    self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lge += out[0] * n_batch_actual
                    lde += out[1] * n_batch_actual
                    lfe += out[2] * n_batch_actual
                self._log(out[3], step, test=True)
                print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f}  F: {:f})  time: {:d} seconds' \
                    .format(epoch + 1, n_epochs, (lge + lde) / n, lge / n, lde / n, lfe / n, int(time() - start))


class MSSGAN (GANBase):

    name = 'MSSGAN'

    def __init__(self, n_y, additional_features=None, n_extra_generator_layers=0, n_extra_discriminator_layers=0, mask=None,
                 use_batch_norm_G=True, use_batch_norm_D=False, name=None,
                 seed=np.random.randint(int(1e8)), log_and_save=True, early_stopping=False):

        # label variables
        self.n_y = n_y
        self.input_y = tf.placeholder(tf.float32, (None, self.n_y), 'label')
        self.input_x_l = tf.placeholder(tf.float32, (None, 32, 32, 1), 'labeled_image')

        # additional feature sets
        self.additional_feature_names = []
        self.additional_feature_dimensions = []
        self.input_x_additional = []
        self.input_x_l_additional = []
        if additional_features is not None:
            assert isinstance(additional_features, dict), 'additional_features must be of type dict or None'
            for key, val in additional_features.iteritems():
                assert isinstance(key, str), 'additional_features keys must be of type str'
                assert isinstance(val, int), 'additional_features keys must be of type int'
                self.additional_feature_names.append(key)
                self.additional_feature_dimensions.append(val)
                self.input_x_additional.append(tf.placeholder(tf.float32, (None, val), key))
                self.input_x_l_additional.append(tf.placeholder(tf.float32, (None, val), 'labeled_' + key))

        # init
        super(MSSGAN, self).__init__(n_extra_generator_layers=n_extra_generator_layers,
                                     n_extra_discriminator_layers=n_extra_discriminator_layers,
                                     mask=mask, use_batch_norm_G=use_batch_norm_G, use_batch_norm_D=use_batch_norm_D,
                                     name=name, log_and_save=log_and_save, seed=seed, early_stopping=early_stopping)

    def _build_additional_generator(self, n_out, name, tensor=None, n_hidden_layers=4, n_hidden_nodes=128,
                                    training=False, batch_norm=None):
        if tensor is None:
            tensor = self.input_n
        if batch_norm is None:
            batch_norm = self.batch_norm_G
        if batch_norm:
            def bn(x):
                return tf.contrib.layers.batch_norm(x, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        with tf.variable_scope('generator_' + name) as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # initial layer
            with tf.variable_scope('initial.{0}-{1}'.format(self.n_noise, n_hidden_nodes)):
                tensor = tf.nn.relu(bn(tf.layers.dense(tensor, n_hidden_nodes, kernel_initializer=init_normal(),
                                                       use_bias=not batch_norm, name='dense')))
            # extra layers
            for it in range(n_hidden_layers-1):
                with tf.variable_scope('extra-{0}.{1}'.format(it, n_hidden_nodes)):
                    tensor = tf.nn.relu(bn(tf.layers.dense(tensor, n_hidden_nodes, kernel_initializer=init_normal(),
                                                           use_bias=not batch_norm, name='dense')))

            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(n_hidden_nodes, n_out)):
                tensor = tf.layers.dense(tensor, n_out, kernel_initializer=init_normal(), name='dense')

            return tf.identity(tensor, name='generated_' + name)

    def _build_additional_discriminator_base(self, tensor, name, n_hidden_layers=3, n_hidden_nodes=128,
                                             n_out_nodes=128, training=False, batch_norm=None):
        if batch_norm is None:
            batch_norm = self.batch_norm_D
        if batch_norm:
            def bn(x):
                return tf.contrib.layers.batch_norm(x, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        # initial layer
        with tf.variable_scope('initial.{0}-{1}'.format(tensor.shape[1], n_hidden_nodes)):
            tensor = tf.nn.relu(bn(tf.layers.dense(tensor, n_hidden_nodes, kernel_initializer=init_normal(),
                                                   use_bias=not batch_norm, name='dense')))
        # extra layers
        for it in range(n_hidden_layers-1):
            with tf.variable_scope('extra-{0}.{1}'.format(it, n_hidden_nodes)):
                tensor = tf.nn.relu(bn(tf.layers.dense(tensor, n_hidden_nodes, kernel_initializer=init_normal(),
                                                       use_bias=not batch_norm, name='dense')))

        with tf.variable_scope('base_final-{0}.{1}'.format(n_hidden_nodes, n_out_nodes)):
            tensor = tf.nn.relu(bn(tf.layers.dense(tensor, n_out_nodes, kernel_initializer=init_normal(),
                                                   use_bias=not batch_norm, name='dense')))

        return tensor

    def _build_multi_discriminator_base(self, tensor=None, training=False, batch_norm=None):
        base_out_size = 100

        if tensor is None:
            tensor = [self.input_x] + self.input_x_additional
        assert not isinstance(tensor, (list, tuple)) or len(self.additional_feature_names) == len(tensor) - 1, \
            'wrong number of input tensors'
        if batch_norm is None:
            batch_norm = self.batch_norm_D

        # discriminator base for image
        with tf.variable_scope('image'):
            if isinstance(tensor, (list, tuple)):
                tensor_pre = [self._build_discriminator_base(tensor[0], training, batch_norm=batch_norm)]
            else:
                tensor_pre = [self._build_discriminator_base(tensor, training, batch_norm=batch_norm)]

        # additional discriminator bases
        for it, name in enumerate(self.additional_feature_names):
            with tf.variable_scope(name):
                tensor_pre.append(self._build_additional_discriminator_base(tensor[it + 1],
                                                                            self.additional_feature_names[it],
                                                                            n_out_nodes=base_out_size,
                                                                            training=training,
                                                                            batch_norm=batch_norm))

        # concatenate
        for it, name in enumerate(self.additional_feature_names):
            tensor_pre[it + 1] = tf.expand_dims(tf.expand_dims(tensor_pre[it + 1], 1), 1)
            tensor_pre[it + 1] = tf.tile(tensor_pre[it + 1], [1, int(tensor_pre[0].shape[1]), int(tensor_pre[0].shape[2]), 1])

        return tf.concat(tensor_pre, axis=3)

    def _build_discriminator(self, tensor=None, training=False, batch_norm=None):

        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_multi_discriminator_base(tensor, training, batch_norm)

            # last conv layer
            d_out = 1 + self.n_y
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], d_out)):
                out_logits = tf.reshape(tf.layers.conv2d(tensor, d_out, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, d_out])

            return tf.nn.softmax(out_logits, name='pred_probs'), tf.identity(out_logits, name='pred_logits')

    def _build_loss(self, label_strength=1., training=False):

        fake = [self._build_generator(training=training)]
        for it in range(len(self.additional_feature_names)):
            fake.append(self._build_additional_generator(self.additional_feature_dimensions[it],
                                                         self.additional_feature_names[it], training=training))

        fake_label, fake_logits = self._build_discriminator(fake, training=training)
        real_label_u, real_logits_u = self._build_discriminator(training=training)
        real_label_l, real_logits_l = self._build_discriminator([self.input_x_l] + self.input_x_l_additional,
                                                                training=training)

        label_smooth = tf.concat((label_strength * self.input_y,
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

        # generator
        lossG = tf.reduce_mean(-safe_log(tf.reduce_sum(fake_label[:, :-1], 1)))

        # discriminator
        lossD_d_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits_l, labels=label_smooth))
        lossD_d_u = tf.reduce_mean(-safe_log(tf.reduce_sum(real_label_u[:, :-1], 1)))
        lossD_g = tf.reduce_mean(-safe_log(fake_label[:, -1]))
        lossD = lossD_d_l + lossD_d_u + lossD_g

        # summaries
        if training:
            with tf.name_scope('training_metrics'):
                # generated/real images
                tf.summary.image('fake_image', fake[0], max_outputs=1)
                tf.summary.image('real_image', self.input_x, max_outputs=1)
                # generated/real psds
                try:
                    tf.summary.image('fake_psd', tf.expand_dims(tf.expand_dims(fake[1], 0), 3))
                    tf.summary.image('real_psd', tf.expand_dims(tf.expand_dims(self.input_x_additional[0], 0), 3))
                except IndexError:
                    pass
                # classifier performance
                pred = tf.argmax(real_logits_l[:, :-1], 1)
                tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
                cmat = tf.reshape(tf.confusion_matrix(tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16),
                                  [1, self.n_y, self.n_y, 1])
                tf.summary.image('confusion_matrix', cmat)
                tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.001))
                # discriminator performance
                tf.summary.histogram('D_fake', fake_label[:, -1])
                tf.summary.histogram('D_real', real_label_l[:, -1])
                # GAN loss
                tf.summary.scalar('lossG', lossG)
                tf.summary.scalar('lossD_d_l', lossD_d_l)
                tf.summary.scalar('lossD_d_u', lossD_d_u)
                tf.summary.scalar('lossD_g', lossD_g)
                tf.summary.scalar('lossD', lossD)
                tf.summary.scalar('loss', lossG + lossD)

        else:
            collection = ['eval']
            update_collection = ['eval_update']
            true_val = tf.argmax(self.input_y, 1)
            pred = tf.argmax(real_logits_l[:, :-1], 1)

            with tf.name_scope('testing_metrics'):
                # classifier accuracy
                acc, update_eval_acc = tf.metrics.accuracy(true_val, pred, updates_collections=update_collection, name='avgacc')
                tf.summary.scalar('accuracy', acc, collections=collection)
                # classifier confusion matrix
                cmat = tf.reshape(tf.confusion_matrix(true_val, pred, self.n_y, tf.float32), [1, self.n_y, self.n_y, 1])
                cmat, update_eval_cmat = tf.metrics.mean_tensor(cmat, updates_collections=update_collection, name='avgcmat')
                tf.summary.image('confusion_matrix', cmat, collections=collection)
                tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.001), collections=collection)
                # discriminator performance
                tf.summary.histogram('D_fake', fake_label[:, -1], collections=collection)
                tf.summary.histogram('D_real', real_label_l[:, -1], collections=collection)
                # GAN loss
                mlossG, update_eval_lossG = tf.metrics.mean(lossG, updates_collections=update_collection, name='lossG')
                tf.summary.scalar('lossG', mlossG, collections=collection)
                mlossD_d_l, update_eval_lossG = tf.metrics.mean(lossD_d_l, updates_collections=update_collection, name='lossD_d_l')
                tf.summary.scalar('lossD_d_l', mlossD_d_l, collections=collection)
                mlossD_d_u, update_eval_lossG = tf.metrics.mean(lossD_d_u, updates_collections=update_collection, name='lossD_d_u')
                tf.summary.scalar('lossD_d_u', mlossD_d_u, collections=collection)
                mlossD_g, update_eval_lossG = tf.metrics.mean(lossD_g, updates_collections=update_collection, name='lossD_g')
                tf.summary.scalar('lossD_g', mlossD_g, collections=collection)
                mlossD, update_eval_lossD = tf.metrics.mean(lossD, updates_collections=update_collection, name='lossD')
                tf.summary.scalar('lossD', mlossD, collections=collection)
                tf.summary.scalar('loss', mlossG + mlossD, collections=collection)
                # save eval updates
                for node in ['avgacc', 'avgcmat', 'lossG', 'lossD_d_l', 'lossD_d_u', 'lossD_g', 'lossD']:
                    for var in ['total:0', 'count:0', 'total_tensor:0', 'count_tensor:0']:
                        name = 'testing_metrics/' + node + '/' + var
                        try:
                            temp = tf.get_default_graph().get_tensor_by_name(name)
                        except KeyError:
                            continue
                        tf.add_to_collection('reset_eval', tf.assign(temp, tf.zeros_like(temp)))

        return lossG, lossD

    def _train(self, trainx_u, trainx_l, trainy, testx, testy, n_epochs=25, n_batch=128, balance_labels=False,
               learning_rate=2e-4, label_strength=1.):

        # handle data
        n_train_u = trainx_u[0].shape[0]
        n_train_l = trainx_l[0].shape[0]
        n_test = testx[0].shape[0]

        # setup learning
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD = self._build_loss(label_strength=label_strength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            adamG = tf.contrib.layers.optimize_loss(loss=lossG,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optG',
                                                    variables=tvarsG)
            adamD = tf.contrib.layers.optimize_loss(loss=lossD,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optD',
                                                    variables=tvarsD)

        # summary
        merged_summary = tf.summary.merge_all()
        eval_summary = tf.summary.merge(tf.get_collection('eval'))

        # start session
        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())

            # if self.debug:
            #     from tensorflow.python import debug as tf_debug
            #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            #     sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            # train
            self._start_logging_and_saving(sess)
            while True:
                try:
                    for epoch in range(n_epochs):
                        # train on epoch
                        start = time()
                        step = 0
                        n, lg, ld = 0, 0, 0
                        for batch_index, n_batch_actual in batcher(n_train_u, n_batch):
                            # prep
                            n += n_batch_actual
                            if balance_labels:
                                randind = balanced_rand(trainy, n_batch_actual)
                            else:
                                randind = np.random.choice(n_train_l, n_batch_actual)
                            feed_dict = {self.input_x: trainx_u[0][batch_index:batch_index + n_batch_actual],
                                         self.input_x_l: trainx_l[0][randind],
                                         self.input_y: trainy[randind],
                                         self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                            for it in range(len(self.additional_feature_names)):
                                feed_dict[self.input_x_additional[it]] = trainx_u[it + 1][batch_index:batch_index + n_batch_actual]
                                feed_dict[self.input_x_l_additional[it]] = trainx_l[it + 1][randind]
                            # discriminator
                            temp = sess.run(adamD, feed_dict)
                            ld += temp * n_batch_actual
                            # generator
                            feed_dict[self.input_n] = np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)
                            temp, summary, step = sess.run([adamG, merged_summary, global_step], feed_dict)
                            lg += temp * n_batch_actual
                            if np.isnan([lg, ld]).any() or np.isinf([lg, ld]).any():
                                raise NanInfException
                            if n % (2 * n_batch) == 0:
                                self._log(summary, step)
                                print 'epoch {:d}/{:d} (part {:d}/{:d}):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                                    .format(epoch + 1, n_epochs, n, n_train_u, (lg + ld) / n, lg / n, ld / n, int(time() - start))
                            if n % (100 * n_batch) == 0:
                                m, lge, lde = 0, 0, 0
                                sess.run(tf.get_collection('reset_eval'))
                                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                                    m += n_batch_actual
                                    feed_dict = {self.input_x: testx[0][batch_index:batch_index + n_batch_actual],
                                                 self.input_x_l: testx[0][batch_index:batch_index + n_batch_actual],
                                                 self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                                 self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                                    for it in range(len(self.additional_feature_names)):
                                        feed_dict[self.input_x_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                                        feed_dict[self.input_x_l_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                                    out = sess.run([evalG, evalD, eval_summary] + tf.get_collection('eval_update'), feed_dict)
                                    lge += out[0] * n_batch_actual
                                    lde += out[1] * n_batch_actual
                                self._log(out[2], step, test=True)
                                print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                                    .format(epoch + 1, n_epochs, (lge + lde) / m, lge / m, lde / m, int(time() - start))
                                # save
                                self._save(sess, step)
                                lossD_d_l = sess.run(tf.get_default_graph().get_tensor_by_name('testing_metrics/lossD_d_l/value:0'))
                                self._early_stopping(lossD_d_l, sess, step, feed_dict)
                                # acc = sess.run(tf.get_default_graph().get_tensor_by_name('avgacc/value:0'))
                                # self._early_stopping(-acc, sess, step, feed_dict)

                        # save after each epoch
                        self._save(sess, step)
                    break

                except NanInfException:
                    if epoch >= 10:
                        a=1
                    print 'Got NaNs or infs. Resetting parameters and starting again.'
                    try:
                        self._restore(sess)
                    except:
                        step = sess.run(global_step)
                        sess.run(tf.global_variables_initializer())
                        tf.assign(global_step, step)
                    trainx_u = shuffle_data(trainx_u)
                    trainx_l, trainy = shuffle_data(trainx_l, trainy)
                    testx, testy = shuffle_data(testx, testy)

    def pred(self, data=None):

        if data is not None:
            # setup
            n = data[0].shape[0]
            n_batch = 100
            pred = np.zeros((n, self.n_y + 1))
            logits = np.zeros((n, self.n_y + 1))

            # desired tensors
            pred_tensor = self.session.graph.get_tensor_by_name('discriminator_2/pred_probs:0')
            logits_tensor = self.session.graph.get_tensor_by_name('discriminator_2/pred_logits:0')

            # run prediction
            for it in range(int(np.ceil(1. * n / n_batch))):
                # make feed_dict
                ind = np.arange(n_batch * it, np.minimum(n, n_batch * (it + 1)))
                feed_dict = {self.input_x_l: data[0][ind]}
                for it in range(len(self.additional_feature_names)):
                    feed_dict[self.input_x_l_additional[it]] = data[it + 1][ind]
                pred[ind], logits[ind] = self.session.run([pred_tensor, logits_tensor], feed_dict=feed_dict)

            # run prediction
            return pred, logits

        else:
            # make feed_dict
            feed_dict = {self.input_n: np.random.randn(20, 100)}

            # desired tensors
            pred_tensor = self.session.graph.get_tensor_by_name('discriminator/pred_probs:0')
            logits_tensor = self.session.graph.get_tensor_by_name('discriminator/pred_logits:0')
            gen = [self.session.graph.get_tensor_by_name('generator/generated_image:0')]

            # run prediction
            for it in range(len(self.additional_feature_names)):
                gen.append(self.session.graph.get_tensor_by_name('generator_' + self.additional_feature_names[it] + '/generated_' + self.additional_feature_names[it] + ':0'))
            return self.session.run([pred_tensor, logits_tensor] + gen, feed_dict=feed_dict)


    def inference_2_matfile(self, session=None, graph=None):
        from scipy.io import savemat

        # get graph
        if graph is None:
            if session is not None:
                graph = session.graph
            else:
                graph = tf.get_default_graph()

        # extract parameters
        params = dict()
        for op in graph.get_operations():
            name = op.name
            if name.startswith('discriminator/') and (name.endswith('kernel') or name.endswith('bias')):
                try:
                    name = name.replace('/', '__').replace('-', '_').replace('.', '_')
                    params[name] = graph.get_tensor_by_name(op.name + ':0').eval(session=session)
                except:
                    pass

        assert params, 'nothing saved'

        # save
        savemat(join(self.path, self.name, self.filename + '_inference'), params, long_field_names=True)

    def make_test_vals(self, session, feed_dict, graph=None, vals=None):
        from scipy.io import savemat

        # get graph
        if graph is None:
            graph = session.graph

        # extract values
        vals = dict()
        for op in graph.get_operations():
            name = op.name
            if name.startswith('discriminator_2/') \
                    and (name.endswith('BiasAdd') or name.endswith('Softmax')
                         or name.endswith('concat') or name.endswith('concat_1')
                         or name.endswith('Maximum') or name.endswith('Relu')):
                try:
                    name = name.replace('/', '__').replace('-', '_').replace('.', '_')
                    vals[name] = session.run(graph.get_tensor_by_name(op.name + ':0'), feed_dict=feed_dict)
                except:
                    pass

        # append inputs
        vals['in_image'] = feed_dict[self.input_x_l]
        vals['in_psd'] = feed_dict[graph.get_tensor_by_name('labeled_psd_med:0')]
        try:
            vals['in_autocorr'] = feed_dict[graph.get_tensor_by_name('labeled_autocorr:0')]
        except KeyError:
            pass

        assert vals, 'nothing saved'

        # save
        savemat(join(self.path, self.name, self.filename + '_test_vals'), vals, long_field_names=True)


class SymMSSGAN (MSSGAN):

    name = 'SymMSSGAN'

    def _build_loss(self, label_strength=1., training=False):

        fake = [self._build_generator(training=training)]
        for it in range(len(self.additional_feature_names)):
            fake.append(self._build_additional_generator(self.additional_feature_dimensions[it],
                                                         self.additional_feature_names[it], training=training))

        fake_label, fake_logits = self._build_discriminator(fake, training=training)
        real_label_u, real_logits_u = self._build_discriminator(training=training)
        real_label_l, real_logits_l = self._build_discriminator([self.input_x_l] + self.input_x_l_additional,
                                                                training=training)

        fake[0] *= -1
        _, fake_flipped_logits = self._build_discriminator(fake, training=training)
        _, real_flipped_logits_u = self._build_discriminator([-self.input_x] + self.input_x_additional, training=training)
        _, real_flipped_logits_l = self._build_discriminator([-self.input_x_l] + self.input_x_l_additional,
                                                             training=training)

        label_smooth = tf.concat((label_strength * self.input_y,
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

        # generator
        lossG = tf.reduce_mean(-safe_log(tf.reduce_sum(fake_label[:, :-1], 1)))

        # discriminator
        lossD_d_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits_l, labels=label_smooth))
        lossD_d_u = tf.reduce_mean(-safe_log(tf.reduce_sum(real_label_u[:, :-1], 1)))
        lossD_g = tf.reduce_mean(-safe_log(fake_label[:, -1]))
        lossD_sym = tf.sqrt(tf.reduce_mean(tf.squared_difference(fake_logits, fake_flipped_logits))
                            + tf.reduce_mean(tf.squared_difference(real_logits_u, real_flipped_logits_u))
                            + tf.reduce_mean(tf.squared_difference(real_logits_l, real_flipped_logits_l)))
        lossD = lossD_d_l + lossD_d_u + lossD_g + lossD_sym

        # summaries
        if training:
            # generated/real images
            tf.summary.image('fake_image', fake[0], max_outputs=1)
            tf.summary.image('real_image', self.input_x, max_outputs=1)
            # generated/real images
            tf.summary.image('fake_psd', tf.expand_dims(tf.expand_dims(fake[1], 0), 3))
            tf.summary.image('real_psd', tf.expand_dims(tf.expand_dims(self.input_x_additional[0], 0), 3))
            # classifier performance
            pred = tf.argmax(real_label_l[:, :-1] / (1 - real_label_l[:, -1, None]), 1)
            tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
            cmat = tf.reshape(tf.confusion_matrix(tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16),
                              [1, self.n_y, self.n_y, 1])
            tf.summary.image('confusion_matrix', cmat)
            tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.1))
            # discriminator performance
            tf.summary.histogram('D_fake', fake_label[:, -1])
            tf.summary.histogram('D_real', real_label_l[:, -1])
            # GAN loss
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d_l', lossD_d_l)
            tf.summary.scalar('lossD_d_u', lossD_d_u)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD_sym', lossD_sym)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('loss', lossG + lossD)

        return lossG, lossD


class ConvMSSGAN (MSSGAN):

    name = 'ConvMSSGAN'

    def _build_additional_generator(self, n_out, name, tensor=None, n_hidden_layers=3, n_hidden_nodes=None,
                                    training=False, batch_norm=None, nfilt=2000):
        if tensor is None:
            tensor = self.input_n
        if batch_norm is None:
            batch_norm = self.batch_norm_G
        if batch_norm:
            def bn(x):
                return tf.contrib.layers.batch_norm(x, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        with tf.variable_scope('generator_' + name) as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # reshape input
            tensor = tf.expand_dims(tensor, 2)

            # hidden layers
            for it in range(n_hidden_layers):
                with tf.variable_scope('hidden.{0}-{1}'.format(nfilt, nfilt/2)):
                    tensor = tf.nn.relu(bn(tf.layers.conv1d(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                            kernel_initializer=init_normal(),
                                                            name='conv')))
                nfilt /= 2

            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(nfilt, 1)):
                tensor = tf.layers.conv1d(tensor, self.n_channel, 3, 1, 'same', activation=tf.tanh,
                                          kernel_initializer=init_normal(),
                                          name='conv')

            return tf.reshape(tensor, (-1, 100), name='generated_' + name)

    def _build_additional_discriminator_base(self, tensor, name, n_hidden_layers=2, n_hidden_nodes=None,
                                             n_out_nodes=100, training=False, batch_norm=None):
        nfilt = self.n_filtd
        if batch_norm is None:
            batch_norm = self.batch_norm_D
        if batch_norm:
            def bn(x):
                return tf.contrib.layers.batch_norm(x, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        # reshape input
        tensor = tf.expand_dims(tensor, 2)

        # initial layer
        with tf.variable_scope('initial.{0}-{1}'.format(100, nfilt)):
            tensor = lrelu(bn(tf.layers.conv1d(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                               kernel_initializer=init_normal(),
                                               name='conv')))

        # hidden layers
        for it in range(n_hidden_layers - 1):
            with tf.variable_scope('hidden.{0}-{1}'.format(nfilt, nfilt * 2)):
                tensor = lrelu(bn(tf.layers.conv1d(tensor, nfilt * 2, 3, 1, 'same', use_bias=not batch_norm,
                                                   kernel_initializer=init_normal(),
                                                   name='conv')))
                nfilt *= 2

            # final layer
        with tf.variable_scope('final.{0}-{1}'.format(nfilt, 1)):
            tensor = lrelu(bn(tf.layers.conv1d(tensor, 1, 3, 1, 'same', use_bias=not batch_norm,
                                               kernel_initializer=init_normal(),
                                               name='conv')))

        return tf.reshape(tensor, (-1, n_out_nodes))


class AltConvMSSGAN (ConvMSSGAN):

    name = 'AltConvMSSGAN'

    def _build_generator(self, tensor=None, training=False, batch_norm=None, nfilt=2000):
        assert self.n_pixel % 16 == 0, "isize has to be a multiple of 16"
        csize = 4
        if tensor is None:
            tensor = self.input_n
        if batch_norm is None:
            batch_norm = self.batch_norm_G
        if batch_norm:
            def bn(x, name=None):
                return tf.contrib.layers.batch_norm(x, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        with tf.variable_scope('generator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # initial layer
            with tf.variable_scope('initial.{0}-{1}'.format(self.n_noise, nfilt)):
                tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tf.reshape(tensor, [-1, 1, 1, self.n_noise]),
                                                                  nfilt, 4, 2, 'valid', use_bias=not batch_norm,
                                                                  kernel_initializer=init_normal(),
                                                                  name='conv'), name='bn'))

            # upscaling layers
            while csize < self.n_pixel/2:
                with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt/2)):
                    tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt/2, 4, 2, 'same',
                                                                      use_bias=not batch_norm,
                                                                      kernel_initializer=init_normal(),
                                                                      name='conv'), name='bn'))
                csize *= 2
                nfilt /= 2

            # extra layers
            for it in range(self.n_extra_generator_layers):
                with tf.variable_scope('extra-{0}.{1}'.format(it, nfilt)):
                    tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                                      kernel_initializer=init_normal(),
                                                                      name='conv'), name='bn'))

            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(nfilt, self.n_channel)):
                tensor = tf.layers.conv2d_transpose(tensor, self.n_channel, 4, 2, 'same', activation=None,
                                                    kernel_initializer=init_normal(),
                                                    name='conv')
                tensor /= tf.reshape(tf.reduce_max(tf.reshape(tf.abs(tensor),
                                                              [-1, 32*32]), 1), [-1, 1, 1, 1]) / 0.99

            # mask layer
            return tf.identity(tensor * self.mask, name='generated_image')

    def _build_additional_generator(self, n_out, name, tensor=None, n_hidden_layers=5, n_hidden_nodes=None,
                                    training=False, batch_norm=None, nfilt=2000):
        csize = 4
        if tensor is None:
            tensor = self.input_n
        if batch_norm is None:
            batch_norm = self.batch_norm_G
        if batch_norm:
            def bn(x):
                return tf.contrib.layers.batch_norm(x, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        with tf.variable_scope('generator_' + name) as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # initial layer: size=4
            with tf.variable_scope('initial.{0}-{1}'.format(self.n_noise, nfilt)):
                tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tf.reshape(tensor, [-1, 1, 1, self.n_noise]),
                                                                  nfilt, [4, 1], [2, 1], 'valid', use_bias=not batch_norm,
                                                                  kernel_initializer=init_normal(),
                                                                  name='conv')))
            # upscale: size=12
            with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt/2)):
                tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt/2, [6, 1], [3, 1], 'same',
                                                                  use_bias=not batch_norm,
                                                                  kernel_initializer=init_normal(),
                                                                  name='conv')))
            csize *= 2
            nfilt /= 2

            # upscale: size=36
            with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt/2)):
                tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt/2, [6, 1], [3, 1], 'same',
                                                                  use_bias=not batch_norm,
                                                                  kernel_initializer=init_normal(),
                                                                  name='conv')))
            csize *= 2
            nfilt /= 2

            # extra layers
            for it in range(self.n_extra_generator_layers):
                with tf.variable_scope('extra-{0}.{1}'.format(it, nfilt)):
                    tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt, [3, 1], [1, 1], 'same', use_bias=not batch_norm,
                                                                      kernel_initializer=init_normal(),
                                                                      name='conv')))

            # final layer: size=100
            with tf.variable_scope('final.{0}-{1}'.format(nfilt, self.n_channel)):
                tensor = tensor[:, 1:]
                tensor = tf.layers.conv2d_transpose(tensor, self.n_channel, [6, 1], [3, 1], 'same', activation=None,
                                                    kernel_initializer=init_normal(),
                                                    name='conv')
                tensor = tensor[:, 3:-2]

            return tf.reshape(tensor, (-1, 100), name='generated_' + name)

    def _build_discriminator(self, tensor=None, training=False, batch_norm=None):

        if training:
            # get tensor
            if tensor is None:
                tensor = [self.input_x] + self.input_x_additional

            with tf.variable_scope('input_noise') as scope:
                # set reuse if necessary
                if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                    scope.reuse_variables()

                # add noise
                std_start = 0.5
                std_end = 0.05
                noise_std = tf.identity(std_end + tf.train.exponential_decay(std_start - std_end, tf.train.get_or_create_global_step(),
                                                                          25., 0.96, staircase=False), name='noise_std')
                tf.summary.scalar('noise_std', noise_std)
                for it in range(len(tensor)):
                    if not it:
                        tensor[it] += tf.random_normal(tf.shape(tensor[it]), stddev=noise_std) * self.mask
                    else:
                        tensor[it] += tf.random_normal(tf.shape(tensor[it]), stddev=noise_std)

        # build discriminator
        return super(AltConvMSSGAN, self)._build_discriminator(tensor=tensor, training=training, batch_norm=batch_norm)


class BadFMMSSGAN (AltConvMSSGAN):

    name = 'BadFMMSSGAN'

    def _build_loss(self, label_strength=1., training=False):

        fake = [self._build_generator(training=training, nfilt=512)]
        for it in range(len(self.additional_feature_names)):
            fake.append(self._build_additional_generator(self.additional_feature_dimensions[it],
                                                         self.additional_feature_names[it],
                                                         training=training, nfilt=512))

        fake_label, fake_logits = self._build_discriminator(fake, training=training)
        real_label_u, real_logits_u = self._build_discriminator(training=training)
        real_label_l, real_logits_l = self._build_discriminator([self.input_x_l] + self.input_x_l_additional,
                                                                training=training)

        with tf.name_scope('loss'):
            label_smooth = tf.concat((label_strength * self.input_y,
                                      (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

            # generator
            fake_features = tf.get_default_graph().get_tensor_by_name(fake_logits.name.split('/')[0] + '/concat:0')
            real_features = tf.get_default_graph().get_tensor_by_name(real_logits_u.name.split('/')[0] + '/concat:0')
            lossG_fm = tf.reduce_mean(tf.squared_difference(fake_features, real_features))
            batch_size = tf.shape(fake_features)[0]
            fake_features = tf.reshape(fake_features, [batch_size, -1])
            fake_features /= tf.norm(fake_features, axis=1, keep_dims=True)
            cov_mat = 1. / tf.cast(batch_size * (batch_size - 1), tf.float32) \
                      * tf.matmul(fake_features, fake_features, transpose_b=True) ** 2
            lossG_pt = tf.reduce_mean(cov_mat) - tf.reduce_mean(tf.diag(cov_mat))
            lossG = lossG_fm + lossG_pt

            # discriminator
            lossD_d_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits_l, labels=label_smooth))
            lossD_d_u = tf.reduce_mean(-safe_log(tf.reduce_sum(real_label_u[:, :-1], 1)))
            lossD_d_u_ce = -tf.reduce_mean(tf.reduce_sum(real_label_u[:, :-1] * safe_log(real_label_u[:, :-1]), axis=1))
            lossD_g = tf.reduce_mean(-safe_log(fake_label[:, -1]))

            lossD = lossD_d_l + lossD_d_u + lossD_d_u_ce + lossD_g

        # summaries
        if training:
            with tf.name_scope('training_metrics'):
                # generated/real images
                tf.summary.image('fake_image', fake[0], max_outputs=1)
                tf.summary.image('real_image', self.input_x, max_outputs=1)
                # generated/real psds
                try:
                    tf.summary.image('fake_psd', tf.expand_dims(tf.expand_dims(fake[1], 0), 3))
                    tf.summary.image('real_psd', tf.expand_dims(tf.expand_dims(self.input_x_additional[0], 0), 3))
                except IndexError:
                    pass
                # classifier performance
                pred = tf.argmax(real_logits_l[:, :-1], 1)
                tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
                cmat = tf.reshape(tf.confusion_matrix(tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16),
                                  [1, self.n_y, self.n_y, 1])
                tf.summary.image('confusion_matrix', cmat)
                tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.001))
                # discriminator performance
                tf.summary.histogram('D_fake', fake_label[:, -1])
                tf.summary.histogram('D_real', real_label_l[:, -1])
                # GAN loss
                tf.summary.scalar('lossG_fm', lossG_fm)
                tf.summary.scalar('lossG_pt', lossG_pt)
                tf.summary.scalar('lossG', lossG)
                tf.summary.scalar('lossD_d_l', lossD_d_l)
                tf.summary.scalar('lossD_d_u', lossD_d_u)
                tf.summary.scalar('lossD_d_u_ce', lossD_d_u_ce)
                tf.summary.scalar('lossD_g', lossD_g)
                tf.summary.scalar('lossD', lossD)
                tf.summary.scalar('loss', lossG + lossD)

        else:
            collection = ['eval']
            update_collection = ['eval_update']
            true_val = tf.argmax(self.input_y, 1)
            pred = tf.argmax(real_logits_l[:, :-1], 1)

            with tf.name_scope('testing_metrics'):
                # classifier accuracy
                acc, update_eval_acc = tf.metrics.accuracy(true_val, pred, updates_collections=update_collection, name='avgacc')
                tf.summary.scalar('accuracy', acc, collections=collection)
                # classifier confusion matrix
                cmat = tf.reshape(tf.confusion_matrix(true_val, pred, self.n_y, tf.float32), [1, self.n_y, self.n_y, 1])
                cmat, update_eval_cmat = tf.metrics.mean_tensor(cmat, updates_collections=update_collection, name='avgcmat')
                tf.summary.image('confusion_matrix', cmat, collections=collection)
                tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.001), collections=collection)
                # discriminator performance
                tf.summary.histogram('D_fake', fake_label[:, -1], collections=collection)
                tf.summary.histogram('D_real', real_label_l[:, -1], collections=collection)
                # GAN loss
                mlossG_fm, update_eval_lossG = tf.metrics.mean(lossG, updates_collections=update_collection, name='lossG_fm')
                tf.summary.scalar('lossG_fm', mlossG_fm, collections=collection)
                mlossG_pt, update_eval_lossG = tf.metrics.mean(lossG_fm, updates_collections=update_collection, name='lossG_pt')
                tf.summary.scalar('lossG_pt', mlossG_pt, collections=collection)
                mlossG, update_eval_lossG = tf.metrics.mean(lossG_pt, updates_collections=update_collection, name='lossG')
                tf.summary.scalar('lossG', mlossG, collections=collection)
                mlossD_d_l, update_eval_lossD = tf.metrics.mean(lossD_d_l, updates_collections=update_collection, name='lossD_d_l')
                tf.summary.scalar('lossD_d_l', mlossD_d_l, collections=collection)
                mlossD_d_u, update_eval_lossD = tf.metrics.mean(lossD_d_u, updates_collections=update_collection, name='lossD_d_u')
                tf.summary.scalar('lossD_d_u', mlossD_d_u, collections=collection)
                mlossD_d_u_ce, update_eval_lossD = tf.metrics.mean(lossD_d_u_ce, updates_collections=update_collection, name='lossD_d_u_ce')
                tf.summary.scalar('lossD_d_u_ce', mlossD_d_u_ce, collections=collection)
                mlossD_g, update_eval_lossD = tf.metrics.mean(lossD_g, updates_collections=update_collection, name='lossD_g')
                tf.summary.scalar('lossD_g', mlossD_g, collections=collection)
                mlossD, update_eval_lossD = tf.metrics.mean(lossD, updates_collections=update_collection, name='lossD')
                tf.summary.scalar('lossD', mlossD, collections=collection)
                tf.summary.scalar('loss', mlossG + mlossD, collections=collection)
                # save eval updates
                for node in ['avgacc', 'avgcmat', 'lossG_fm', 'lossG_pt', 'lossG',
                             'lossD_d_l', 'lossD_d_u', 'lossD_d_u_ce', 'lossD_g', 'lossD']:
                    for var in ['total:0', 'count:0', 'total_tensor:0', 'count_tensor:0']:
                        name = 'testing_metrics/' + node + '/' + var
                        try:
                            temp = tf.get_default_graph().get_tensor_by_name(name)
                        except KeyError:
                            continue
                        tf.add_to_collection('reset_eval', tf.assign(temp, tf.zeros_like(temp)))

        return lossG, lossD


class ConfMSSGAN (AltConvMSSGAN):

    name = 'ConfMSSGAN'

    def _build_loss(self, label_strength=1., training=False):

        # generator
        fake = [self._build_generator(training=training, nfilt=512)]
        for it in range(len(self.additional_feature_names)):
            fake.append(self._build_additional_generator(self.additional_feature_dimensions[it],
                                                         self.additional_feature_names[it],
                                                         training=training, nfilt=512))

        # discriminator
        n_y = self.n_y
        self.n_y = 1
        fake_label, fake_logits = self._build_discriminator(fake, training=training)
        real_label_u, real_logits_u = self._build_discriminator(training=training)
        real_label_l, real_logits_l = self._build_discriminator([self.input_x_l] + self.input_x_l_additional,
                                                                training=training)

        # confident classifier
        self.n_y = n_y - 1
        with tf.variable_scope('classifier') as scope:
            fake_cls, fake_cls_logits = self._build_discriminator(fake, training=training)
            real_cls, real_cls_logits = self._build_discriminator([self.input_x_l] + self.input_x_l_additional,
                                                                  training=training)
        self.n_y = n_y

        with tf.name_scope('loss'):

            # generator
            lossG_gan = tf.reduce_mean(-safe_log(tf.reduce_sum(fake_label[:, :-1], 1)), name='lossG_gan')
            tf.add_to_collection('losses', lossG_gan)
            batch_size = tf.shape(fake[0])[0]
            fake_flat = tf.concat([tf.reshape(fake[0], [batch_size, -1])] + fake[1:], axis=1)
            fake_flat = fake_flat / tf.norm(fake_flat, axis=1, keep_dims=True)
            cor_mat = tf.square(tf.matmul(fake_flat, fake_flat, transpose_b=True), name='cormat')
            lossG_pt = tf.identity((tf.reduce_sum(cor_mat) - tf.cast(batch_size, tf.float32))
                                   / tf.cast(batch_size * (batch_size - 1), tf.float32), name='lossG_pt')
            tf.add_to_collection('losses', lossG_pt)
            eps = 0.01
            kl = tf.identity(- tf.log(float(self.n_y)) - tf.reduce_mean(safe_log(fake_cls, name='kld1'), 1, name='kld2'), name='kld3')
            lossGC_kl = tf.reduce_sum(tf.boolean_mask(kl, kl > eps) / tf.cast(batch_size, tf.float32), name='lossGC_kl')
            tf.add_to_collection('losses', lossGC_kl)
            lossG = tf.identity(lossG_gan + lossG_pt + lossGC_kl, name='lossG')
            tf.add_to_collection('losses', lossG)

            # discriminator
            # lossD_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits_u, labels=label_smooth))
            lossD_d = tf.reduce_mean(-safe_log(tf.reduce_sum(real_label_u[:, :-1], 1)), name='lossD_d')
            tf.add_to_collection('losses', lossD_d)
            lossD_g = tf.reduce_mean(-safe_log(fake_label[:, -1]), name='lossD_g')
            tf.add_to_collection('losses', lossD_g)
            lossD = tf.identity(lossD_d + lossD_g, name='lossD')
            tf.add_to_collection('losses', lossD)

            # confident classifier
            lossC_sm = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_cls_logits,
                                                                              labels=self.input_y), name='lossC_sm')
            tf.add_to_collection('losses', lossC_sm)
            lossC = tf.identity(lossC_sm + lossGC_kl, name='lossC')
            tf.add_to_collection('losses', lossC)

            tf.add_to_collection('losses', tf.identity(lossG + lossD + lossC, name='loss'))


        # summaries
        if training:
            with tf.name_scope('training_metrics'):
                # generated/real images
                tf.summary.image('fake_image', fake[0], max_outputs=1)
                tf.summary.image('real_image', self.input_x, max_outputs=1)
                # generated/real psds
                try:
                    tf.summary.image('fake_psd', tf.expand_dims(tf.expand_dims(fake[1], 0), 3))
                    tf.summary.image('real_psd', tf.expand_dims(tf.expand_dims(self.input_x_additional[0], 0), 3))
                except IndexError:
                    pass
                # classifier performance
                pred = tf.argmax(real_logits_l[:, :-1], 1)
                tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
                cmat = tf.reshape(tf.confusion_matrix(tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16),
                                  [1, self.n_y, self.n_y, 1])
                tf.summary.image('confusion_matrix', cmat)
                tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.001))
                # discriminator performance
                tf.summary.histogram('D_fake', fake_label[:, -1])
                tf.summary.histogram('D_real', real_label_l[:, -1])
                # GAN loss
                tf.summary.scalar('lossG_gan', lossG_gan)
                tf.summary.scalar('lossG_pt', lossG_pt)
                tf.summary.scalar('lossG_kl', lossGC_kl)
                tf.summary.scalar('lossG', lossG)
                tf.summary.scalar('lossD_d', lossD_d)
                tf.summary.scalar('lossD_g', lossD_g)
                tf.summary.scalar('lossD', lossD)
                tf.summary.scalar('lossC_sm', lossC_sm)
                tf.summary.scalar('lossC', lossC)
                tf.summary.scalar('loss', lossG + lossD + lossC)

        else:
            collection = ['eval']
            update_collection = ['eval_update']
            true_val = tf.argmax(self.input_y, 1)
            pred = tf.argmax(real_cls_logits, 1)

            with tf.name_scope('testing_metrics'):
                # classifier accuracy
                acc, update_eval_acc = tf.metrics.accuracy(true_val, pred, updates_collections=update_collection, name='avgacc')
                tf.summary.scalar('accuracy', acc, collections=collection)
                # classifier confusion matrix
                cmat = tf.reshape(tf.confusion_matrix(true_val, pred, self.n_y, tf.float32), [1, self.n_y, self.n_y, 1])
                cmat, update_eval_cmat = tf.metrics.mean_tensor(cmat, updates_collections=update_collection, name='avgcmat')
                tf.summary.image('confusion_matrix', cmat, collections=collection)
                tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.001), collections=collection)
                # discriminator performance
                tf.summary.histogram('D_fake', fake_label[:, -1], collections=collection)
                tf.summary.histogram('D_real', real_label_l[:, -1], collections=collection)
                # GAN loss

                def update_stat(name, val):
                    mean, update_eval = tf.metrics.mean(val, updates_collections=update_collection,
                                                             name=name)
                    tf.summary.scalar(name, mean, collections=collection)

                update_stat('lossG_gan', lossG_gan)
                update_stat('lossG_pt', lossG_pt)
                update_stat('lossGC_kl', lossGC_kl)
                update_stat('lossG', lossG)
                update_stat('lossD_d', lossD_d)
                update_stat('lossD_g', lossD_g)
                update_stat('lossD', lossD)
                update_stat('lossC_sm', lossC_sm)
                update_stat('lossC', lossC)
                update_stat('loss', lossG + lossD + lossC)

                # save eval updates
                for node in ['avgacc', 'avgcmat', 'lossG_gan', 'lossG_pt', 'lossGC_kl', 'lossG',
                             'lossD_d', 'lossD_g', 'lossD', 'lossC_sm', 'lossC', 'loss']:
                    for var in ['total:0', 'count:0', 'total_tensor:0', 'count_tensor:0']:
                        name = 'testing_metrics/' + node + '/' + var
                        try:
                            temp = tf.get_default_graph().get_tensor_by_name(name)
                        except KeyError:
                            continue
                        tf.add_to_collection('reset_eval', tf.assign(temp, tf.zeros_like(temp)))

        return lossG, lossD, lossC

    def _train(self, trainx_u, trainx_l, trainy, testx, testy, n_epochs=25, n_batch=128, balance_labels=False,
               learning_rate=2e-4, label_strength=1.):

        # handle data
        n_train_u = trainx_u[0].shape[0]
        n_train_l = trainx_l[0].shape[0]
        n_test = testx[0].shape[0]

        # # load or train MANN
        # from tfMLP import ConvMANN
        # with tf.Graph().as_default():
        #     model = ConvMANN(icl_data[1][1][0].shape[1],
        #                      additional_features={x: y for x, y in zip(self.additional_feature_names,
        #                                                                self.additional_feature_dimensions)},
        #                      early_stopping=True, name=self.name)
        # # model.input_x, model.input_x_additional, model.input_y = self.input_x, self.input_x_additional, self.input_y
        #     model.path = join(self.path, self.name)
        #     if not isdir(model.path):
        #         model.train(trainx_l, trainy, testx, testy, balance_labels=balance_labels, learning_rate=learning_rate, n_epochs=2)
        # tf.reset_default_graph()
        # tf.import_graph_def(graph_def)

        # load values in session

        # setup learning
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD, lossC = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD, evalC = self._build_loss(label_strength=label_strength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        tvarsC = [var for var in tf.trainable_variables() if 'classifier' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            adamG = tf.contrib.layers.optimize_loss(loss=lossG,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optG',
                                                    variables=tvarsG)
            adamD = tf.contrib.layers.optimize_loss(loss=lossD,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optD',
                                                    variables=tvarsD)
            adamC = tf.contrib.layers.optimize_loss(loss=lossC,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optC',
                                                    variables=tvarsC)

        # summary
        merged_summary = tf.summary.merge_all()
        eval_summary = tf.summary.merge(tf.get_collection('eval'))

        # start session
        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())
            # try:
            #     model.early_stopping_saver.restore(sess, model.early_stopping_saver.last_checkpoints[-1])
            # except AttributeError:
            #     max_ckpt = max([int(x[ind1 + 5:ind2])
            #                     for x in glob(join(model.path, model.name, model.name + '_early_stopping.ckpt*.meta'))
            #                     for ind1, ind2 in re.search('ckpt-\d+', x).regs])
            #     path = join(model.path, model.name, model.name + '_early_stopping.ckpt-' + str(max_ckpt))
            #     match = '-' + str(self.n_y + 1) + '/'
            #     replace = '-' + str(self.n_y) + '/'
            #     var_list = {}
            #     extra_vars = []
            #     for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'):
            #         if match not in var.name:
            #             var_list[var.name[:-2]] = var
            #         else:
            #             ind = var.name.find(match)
            #             new_str = var.name[:ind] + replace + var.name[ind + len(match):]
            #             new_var = tf.Variable(tf.zeros([x if x is not self.n_y + 1 else self.n_y
            #                                             for x in var.shape.as_list()]), name=new_str[:-2])
            #             var_list[new_str[:-2]] = new_var
            #             extra_vars.append((var, new_var))
            #
            #     sess.run(tf.global_variables_initializer())
            #     tf.train.Saver(var_list=var_list).restore(sess, path)
            #     for old_var, new_var in extra_vars:
            #         tf.assign(old_var, tf.concat((new_var, old_var[..., -1:]), len(old_var.shape) - 1)).eval()

            # train
            self._start_logging_and_saving(sess)
            for epoch in range(n_epochs):
                # train on epoch
                start = time()
                step = 0
                n, lg, ld, lc = 0, 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_train_u, n_batch):
                    # prep
                    n += n_batch_actual
                    if balance_labels:
                        randind = balanced_rand(trainy, n_batch_actual)
                    else:
                        randind = np.random.choice(n_train_l, n_batch_actual)
                    feed_dict = {self.input_x: trainx_u[0][batch_index:batch_index + n_batch_actual],
                                 self.input_x_l: trainx_l[0][randind],
                                 self.input_y: trainy[randind],
                                 self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                    for it in range(len(self.additional_feature_names)):
                        feed_dict[self.input_x_additional[it]] = trainx_u[it + 1][batch_index:batch_index + n_batch_actual]
                        feed_dict[self.input_x_l_additional[it]] = trainx_l[it + 1][randind]

                    # discriminator
                    temp = sess.run(adamD, feed_dict)
                    ld += temp * n_batch_actual

                    # generator
                    feed_dict[self.input_n] = np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)
                    temp = sess.run(adamG, feed_dict)
                    lg += temp * n_batch_actual

                    # classifier
                    feed_dict[self.input_n] = np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)
                    temp, summary, step = sess.run([adamC, merged_summary, global_step], feed_dict)
                    lc += temp * n_batch_actual

                    # metrics
                    if np.isnan([lg, ld]).any() or np.isinf([lg, ld]).any():
                        raise NanInfException
                    if n % (2 * n_batch) == 0:
                        self._log(summary, step)
                        print 'epoch {:d}/{:d} (part {:d}/{:d}):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                            .format(epoch + 1, n_epochs, n, n_train_u, (lg + ld) / n, lg / n, ld / n, int(time() - start))
                    if n % (100 * n_batch) == 0:
                        m, lge, lde, lce = 0, 0, 0, 0
                        sess.run(tf.get_collection('reset_eval'))
                        for batch_index, n_batch_actual in batcher(n_test, n_batch):
                            m += n_batch_actual
                            feed_dict = {self.input_x: testx[0][batch_index:batch_index + n_batch_actual],
                                         self.input_x_l: testx[0][batch_index:batch_index + n_batch_actual],
                                         self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                         self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                            for it in range(len(self.additional_feature_names)):
                                feed_dict[self.input_x_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                                feed_dict[self.input_x_l_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                            out = sess.run([evalG, evalD, evalC, eval_summary] + tf.get_collection('eval_update'), feed_dict)
                            lge += out[0] * n_batch_actual
                            lde += out[1] * n_batch_actual
                            lce += out[2] * n_batch_actual
                        self._log(out[3], step, test=True)
                        print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                            .format(epoch + 1, n_epochs, (lge + lde) / m, lge / m, lde / m, int(time() - start))
                        # save
                        self._save(sess, step)
                        lossC_sm = sess.run(tf.get_default_graph().get_tensor_by_name('testing_metrics/lossC_sm/value:0'))
                        self._early_stopping(lossC_sm, sess, step, feed_dict)

                # save after each epoch
                self._save(sess, step)


class MVEEGAN (MSSGAN):

    name = 'MVEEGAN'

    def _build_encoder(self, tensor=None, training=False, batch_norm=None):

        with tf.variable_scope('encoder') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # encoder base
            tensor = self._build_multi_discriminator_base(tensor, training, batch_norm)

            # last conv layer
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], self.n_noise)):
                out_logits = tf.reshape(tf.layers.conv2d(tensor, self.n_noise, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, self.n_noise])

            # with tf.variable_scope('image'):
            #     if isinstance(tensor, (list, tuple)):
            #         tensor_pre = self._build_discriminator_base(tensor[0], training)
            #     else:
            #         tensor_pre = self._build_discriminator_base(tensor, training)
            #     tensor_pre = [tf.contrib.layers.flatten(tensor_pre)]
            #
            # # additional encoder bases
            # for it, name in enumerate(self.additional_feature_names):
            #     with tf.variable_scope(name):
            #         tensor_pre.append(self._build_additional_generator(self.additional_feature_dimensions[it],
            #                                                                  self.additional_feature_names[it],
            #                                                                  tensor[it + 1]))
            #
            # # flatten
            # tensor = tf.concat(tensor_pre, axis=1)
            #
            # # final layer
            # with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], self.n_noise)):
            #     out_logits = tf.layers.dense(tensor, self.n_noise, kernel_initializer=init_normal(), name='dense')

        return tf.identity(out_logits, name='encoding')

    def _build_discriminator(self, tensor=None, encoding_tensor=None, training=False, batch_norm=None):

        if encoding_tensor is None:
            encoding_tensor = self._build_encoder(tensor, training, batch_norm)

        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_multi_discriminator_base(tensor, training, batch_norm)

            # concatenate encoding tensor
            encoding_tensor_tiled = tf.expand_dims(tf.expand_dims(encoding_tensor, 1), 1)
            encoding_tensor_tiled = tf.tile(encoding_tensor_tiled, [1, int(tensor.shape[1]), int(tensor.shape[2]), 1])
            tensor = tf.concat((tensor, encoding_tensor_tiled), axis=3)

            # last conv layer
            d_out = 1 + self.n_y
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], d_out)):
                out_logits = tf.reshape(tf.layers.conv2d(tensor, d_out, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, d_out])

            return tf.nn.softmax(out_logits, name='pred_probs'), tf.identity(out_logits, name='pred_logits'), encoding_tensor

    def _build_loss(self, label_strength=1., training=False):

        fake = [self._build_generator(training=training)]
        for it in range(len(self.additional_feature_names)):
            fake.append(self._build_additional_generator(self.additional_feature_dimensions[it],
                                                         self.additional_feature_names[it], training=training))
        # fake_encoding = self._build_encoder(fake, training=training)
        fake_label, fake_logits, fake_encoding = self._build_discriminator(fake, training=training)
        real_label_u, real_logits_u, real_encoding_u = self._build_discriminator(training=training)
        real_label_l, real_logits_l, real_encoding_l = self._build_discriminator([self.input_x_l] + self.input_x_l_additional,
                                                                training=training)

        label_smooth = tf.concat((label_strength * self.input_y,
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)


        # encoder loss
        lossF = tf.reduce_mean(tf.squared_difference(self.input_n, fake_encoding))

        # generator
        lossG = tf.reduce_mean(-safe_log(tf.reduce_sum(fake_label[:, :-1], 1))) + lossF

        # discriminator
        lossD_d_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits_l, labels=label_smooth))
        # lossD_d_u = tf.reduce_mean(-safe_log(tf.reduce_sum(real_label_u[:, :-1], 1)))
        lossD_d_u = tf.reduce_mean(-label_strength * safe_log(1 - real_label_u[:, -1])
                                   - (1 - label_strength) * safe_log(real_label_u[:, -1]))
        lossD_g = tf.reduce_mean(-safe_log(fake_label[:, -1]))
        lossD = lossD_d_l + lossD_d_u + lossD_g

        # # summaries
        # if training:
        #     # generated/real images
        #     tf.summary.image('fake_image', fake[0], max_outputs=1)
        #     tf.summary.image('real_image', self.input_x, max_outputs=1)
        #     # generated/real psds
        #     tf.summary.image('fake_psd', tf.expand_dims(tf.expand_dims(fake[1], 0), 3))
        #     tf.summary.image('real_psd', tf.expand_dims(tf.expand_dims(self.input_x_additional[0], 0), 3))
        #     # generated/real autocorr
        #     tf.summary.image('fake_autocorr', tf.expand_dims(tf.expand_dims(fake[2], 0), 3))
        #     tf.summary.image('real_autocorr', tf.expand_dims(tf.expand_dims(self.input_x_additional[1], 0), 3))
        #     # classifier performance
        #     pred = tf.argmax(real_label_l[:, :-1] / (1 - real_label_l[:, -1, None]), 1)
        #     tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
        #     cmat = tf.reshape(tf.confusion_matrix(tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16),
        #                       [1, self.n_y, self.n_y, 1])
        #     tf.summary.image('confusion_matrix', cmat)
        #     tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.1))
        #     # discriminator performance
        #     tf.summary.histogram('D_fake', fake_label[:, -1])
        #     tf.summary.histogram('D_real', real_label_l[:, -1])
        #     # GAN loss
        #     tf.summary.scalar('lossG', lossG)
        #     tf.summary.scalar('lossD_d_l', lossD_d_l)
        #     tf.summary.scalar('lossD_d_u', lossD_d_u)
        #     tf.summary.scalar('lossD_g', lossD_g)
        #     tf.summary.scalar('lossD', lossD)
        #     tf.summary.scalar('lossF', lossF)
        #     tf.summary.scalar('loss', lossG + lossD + lossF)

        # summaries
        if training:
            with tf.name_scope('training_metrics'):
                # generated/real images
                tf.summary.image('fake_image', fake[0], max_outputs=1)
                tf.summary.image('real_image', self.input_x, max_outputs=1)
                # generated/real psds
                try:
                    tf.summary.image('fake_psd', tf.expand_dims(tf.expand_dims(fake[1], 0), 3))
                    tf.summary.image('real_psd', tf.expand_dims(tf.expand_dims(self.input_x_additional[0], 0), 3))
                except IndexError:
                    pass
                # generated/real autocorr
                try:
                    tf.summary.image('fake_autocorr', tf.expand_dims(tf.expand_dims(fake[2], 0), 3))
                    tf.summary.image('real_autocorr', tf.expand_dims(tf.expand_dims(self.input_x_additional[1], 0), 3))
                except IndexError:
                    pass
                # classifier performance
                pred = tf.argmax(real_label_l[:, :-1] / (1 - real_label_l[:, -1, None]), 1)
                tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
                cmat = tf.reshape(tf.confusion_matrix(tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16),
                                  [1, self.n_y, self.n_y, 1])
                tf.summary.image('confusion_matrix', cmat)
                tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.1))
                # discriminator performance
                tf.summary.histogram('D_fake', fake_label[:, -1])
                tf.summary.histogram('D_real', real_label_l[:, -1])
                # GAN loss
                tf.summary.scalar('lossG', lossG)
                tf.summary.scalar('lossD_d_l', lossD_d_l)
                tf.summary.scalar('lossD_d_u', lossD_d_u)
                tf.summary.scalar('lossD_g', lossD_g)
                tf.summary.scalar('lossD', lossD)
                tf.summary.scalar('lossF', lossF)
                tf.summary.scalar('loss', lossG + lossD + lossF)

        else:
            collection = ['eval']
            update_collection = ['eval_update']
            true_val = tf.argmax(self.input_y, 1)
            pred = tf.argmax(real_logits_l[:, :-1], 1)

            with tf.name_scope('testing_metrics'):
                # classifier accuracy
                acc, update_eval_acc = tf.metrics.accuracy(true_val, pred, updates_collections=update_collection,
                                                           name='avgacc')
                tf.summary.scalar('accuracy', acc, collections=collection)
                # classifier confusion matrix
                cmat = tf.reshape(tf.confusion_matrix(true_val, pred, self.n_y, tf.float32), [1, self.n_y, self.n_y, 1])
                cmat, update_eval_cmat = tf.metrics.mean_tensor(cmat, updates_collections=update_collection,
                                                                name='avgcmat')
                tf.summary.image('confusion_matrix', cmat, collections=collection)
                tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.001),
                                 collections=collection)
                # discriminator performance
                tf.summary.histogram('D_fake', fake_label[:, -1], collections=collection)
                tf.summary.histogram('D_real', real_label_l[:, -1], collections=collection)
                # GAN loss
                mlossG, update_eval_lossG = tf.metrics.mean(lossG, updates_collections=update_collection, name='lossG')
                tf.summary.scalar('lossG', mlossG, collections=collection)
                mlossD_d_l, update_eval_lossG = tf.metrics.mean(lossD_d_l, updates_collections=update_collection, name='lossD_d_l')
                tf.summary.scalar('lossD_d_l', mlossD_d_l, collections=collection)
                mlossD_d_u, update_eval_lossG = tf.metrics.mean(lossD_d_u, updates_collections=update_collection, name='lossD_d_u')
                tf.summary.scalar('lossD_d_u', mlossD_d_u, collections=collection)
                mlossD_g, update_eval_lossG = tf.metrics.mean(lossD_g, updates_collections=update_collection, name='lossD_g')
                tf.summary.scalar('lossD_g', mlossD_g, collections=collection)
                mlossD, update_eval_lossD = tf.metrics.mean(lossD, updates_collections=update_collection, name='lossD')
                tf.summary.scalar('lossD', mlossD, collections=collection)
                mlossF, update_eval_lossF = tf.metrics.mean(lossF, updates_collections=update_collection, name='lossF')
                tf.summary.scalar('lossF', lossF, collections=collection)
                tf.summary.scalar('loss', mlossG + mlossD + mlossF, collections=collection)
                # save eval updates
                for node in ['avgacc', 'avgcmat', 'lossG', 'lossD_d_l', 'lossD_d_u', 'lossD_g', 'lossD', 'lossF']:
                    for var in ['total:0', 'count:0', 'total_tensor:0', 'count_tensor:0']:
                        name = 'testing_metrics/' + node + '/' + var
                        try:
                            temp = tf.get_default_graph().get_tensor_by_name(name)
                        except KeyError:
                            continue
                        tf.add_to_collection('reset_eval', tf.assign(temp, tf.zeros_like(temp)))

        return lossG, lossD, lossF

    def _train(self, trainx_u, trainx_l, trainy, testx, testy, n_epochs=25, n_batch=128, balance_labels=False,
               learning_rate=2e-4, label_strength=1.):

        # handle data
        n_train_u = trainx_u[0].shape[0]
        n_train_l = trainx_l[0].shape[0]
        n_test = testx[0].shape[0]

        # setup learning
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD, lossF = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD, evalF = self._build_loss(label_strength=label_strength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        tvarsF = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            adamG = tf.contrib.layers.optimize_loss(loss=lossG,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optG',
                                                    variables=tvarsG)
            adamD = tf.contrib.layers.optimize_loss(loss=lossD,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optD',
                                                    variables=tvarsD)
            adamF = tf.contrib.layers.optimize_loss(loss=lossF,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optF',
                                                    variables=tvarsF)

        # summary
        merged_summary = tf.summary.merge_all()
        eval_summary = tf.summary.merge(tf.get_collection('eval'))

        # start session
        with tf.Session() as sess:

            # initialize variables
            init = tf.global_variables_initializer()
            sess.run(init)

            # if self.debug:
            #     from tensorflow.python import debug as tf_debug
            #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            #     sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            # train
            self._start_logging_and_saving(sess)
            while True:
                try:
                    for epoch in range(n_epochs):
                        # train on epoch
                        start = time()
                        step = 0
                        n, lg, ld, lf = 0, 0, 0, 0
                        for batch_index, n_batch_actual in batcher(n_train_u, n_batch):
                            # prep
                            n += n_batch_actual
                            if balance_labels:
                                randind = balanced_rand(trainy, n_batch_actual)
                            else:
                                randind = np.random.choice(n_train_l, n_batch_actual)
                            feed_dict = {self.input_x: trainx_u[0][batch_index:batch_index + n_batch_actual],
                                         self.input_x_l: trainx_l[0][randind],
                                         self.input_y: trainy[randind],
                                         self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                            for it in range(len(self.additional_feature_names)):
                                feed_dict[self.input_x_additional[it]] = trainx_u[it + 1][batch_index:batch_index + n_batch_actual]
                                feed_dict[self.input_x_l_additional[it]] = trainx_l[it + 1][randind]
                            # update
                            tempg, tempd, tempf, summary, step = \
                                sess.run([adamG, adamD, adamF, merged_summary, global_step], feed_dict)
                            lg += tempg * n_batch_actual
                            ld += tempd * n_batch_actual
                            lf += tempf * n_batch_actual
                            # check
                            if np.isnan([lg, ld, lf]).any() or np.isinf([lg, ld, lf]).any():
                                raise NanInfException
                            if n % (2 * n_batch) == 0:
                                self._log(summary, step)
                                print 'epoch {:d}/{:d} (part {:d}/{:d}):  training loss: {:f} (G: {:f}  D: {:f}  F: {:f})  time: {:d} seconds' \
                                    .format(epoch + 1, n_epochs, n, n_train_u, (lg + ld) / n, lg / n, ld / n, lf / n, int(time() - start))
                            if n % (100 * n_batch) == 0:
                                # evaluate
                                m, lge, lde, lfe = 0, 0, 0, 0
                                sess.run(tf.get_collection('reset_eval'))
                                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                                    m += n_batch_actual
                                    feed_dict = {self.input_x: testx[0][batch_index:batch_index + n_batch_actual],
                                                 self.input_x_l: testx[0][batch_index:batch_index + n_batch_actual],
                                                 self.input_y: testy[batch_index:batch_index + n_batch_actual],
                                                 self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                                    for it in range(len(self.additional_feature_names)):
                                        feed_dict[self.input_x_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                                        feed_dict[self.input_x_l_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                                    out = sess.run([evalG, evalD, evalF, eval_summary] + tf.get_collection('eval_update'), feed_dict)
                                    lge += out[0] * n_batch_actual
                                    lde += out[1] * n_batch_actual
                                    lfe += out[2] * n_batch_actual
                                # print
                                self._log(out[3], step, test=True)
                                print 'epoch {:d}/{:d} (part {:d}/{:d}):  evaluation loss: {:f} (G: {:f}  D: {:f}  F: {:f})  time: {:d} seconds' \
                                    .format(epoch + 1, n_epochs, m, n_train_u, (lge + lde) / m, lge / m, lde / m, lfe / m, int(time() - start))
                                # save
                                self._save(sess, step)
                                lossD_d_l = sess.run(tf.get_default_graph().get_tensor_by_name('testing_metrics/lossD_d_l/value:0'))
                                self._early_stopping(lossD_d_l, sess, step, feed_dict)
                                # acc = sess.run(tf.get_default_graph().get_tensor_by_name('avgacc/value:0'))
                                # self._early_stopping(-acc, sess, step, feed_dict)

                        # save after each epoch
                        self._save(sess, step)

                        # # evaluate
                        # n, lge, lde, lfe = 0, 0, 0, 0
                        # for batch_index, n_batch_actual in batcher(n_test, n_batch):
                        #     n += n_batch_actual
                        #     feed_dict = {self.input_x: testx[0][batch_index:batch_index + n_batch_actual],
                        #                  self.input_x_l: testx[0][batch_index:batch_index + n_batch_actual],
                        #                  self.input_y: testy[batch_index:batch_index + n_batch_actual],
                        #                  self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)}
                        #     for it in range(len(self.additional_feature_names)):
                        #         feed_dict[self.input_x_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                        #         feed_dict[self.input_x_l_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                        #     out = sess.run([evalG, evalD, evalF, merged_summary], feed_dict)
                        #     lge += out[0] * n_batch_actual
                        #     lde += out[1] * n_batch_actual
                        #     lfe += out[2] * n_batch_actual
                        # self._log(out[3], step, test=True)
                        # print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f}  F: {:f})  time: {:d} seconds' \
                        #     .format(epoch + 1, n_epochs, (lge + lde) / n, lge / n, lde / n, lfe / n, int(time() - start))
                    break

                except NanInfException:
                    if epoch >= 10:
                        a=1
                    print 'Got NaNs or infs. Resetting parameters and starting again.'
                    try:
                        self._restore(sess)
                    except:
                        step = sess.run(global_step)
                        sess.run(init)
                        tf.assign(global_step, step)
                    trainx_u = shuffle_data(trainx_u)
                    trainx_l, trainy = shuffle_data(trainx_l, trainy)
                    testx, testy = shuffle_data(testx, testy)

    def eval(self, input_):
        raise NotImplementedError

    def inference_2_matfile(self, session=None, graph=None):
        from scipy.io import savemat

        # get graph
        if graph is None:
            if session is not None:
                graph = session.graph
            else:
                graph = tf.get_default_graph()

        # extract parameters
        params = dict()
        for op in graph.get_operations():
            name = op.name
            if (name.startswith('encoder/') or name.startswith('discriminator/')) \
                    and (name.endswith('kernel') or name.endswith('bias')):
                try:
                    name = name.replace('/', '__').replace('-', '_').replace('.', '_')
                    params[name] = graph.get_tensor_by_name(op.name + ':0').eval(session=session)
                except:
                    pass

        assert params, 'nothing saved'

        # save
        savemat(join(self.path, self.name, self.filename + '_inference'), params, long_field_names=True)

    def make_test_vals(self, session, feed_dict, graph=None, vals=None):
        from scipy.io import savemat

        # get graph
        if graph is None:
            graph = session.graph

        # extract values
        vals = dict()
        for op in graph.get_operations():
            name = op.name
            if (name.startswith('encoder_2/') or name.startswith('discriminator_2/')) \
                    and (name.endswith('BiasAdd') or name.endswith('Softmax')
                         or name.endswith('concat') or name.endswith('concat_1')
                         or name.endswith('Maximum') or name.endswith('Relu')):
                try:
                    name = name.replace('/', '__').replace('-', '_').replace('.', '_')
                    vals[name] = session.run(graph.get_tensor_by_name(op.name + ':0'), feed_dict=feed_dict)
                except:
                    pass

        assert vals, 'nothing saved'

        # append inputs
        vals['in_image'] = feed_dict[self.input_x_l]
        vals['in_psd'] = feed_dict[graph.get_tensor_by_name('labeled_psd_med:0')]
        try:
            vals['in_autocorr'] = feed_dict[graph.get_tensor_by_name('labeled_autocorr:0')]
        except KeyError:
            pass

        # save
        savemat(join(self.path, self.name, self.filename + '_test_vals'), vals, long_field_names=True)


class ConvMVEEGAN (MVEEGAN, ConvMSSGAN):

    name = 'ConvMVEEGAN'


class AltConvMVEEGAN (ConvMVEEGAN, AltConvMSSGAN):

    name = 'AltConvMVEEGAN'

    def _build_discriminator(self, tensor=None, encoding_tensor=None, training=False, batch_norm=None):

        if training:
            # get tensor
            if tensor is None:
                tensor = [self.input_x] + self.input_x_additional

            with tf.variable_scope('input_noise') as scope:
                # set reuse if necessary
                if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                    scope.reuse_variables()

                # add noise
                std_start = 0.5
                std_end = 0.05
                noise_std = tf.identity(std_end + tf.train.exponential_decay(std_start - std_end, tf.train.get_or_create_global_step(),
                                                                          25., 0.96, staircase=False), name='noise_std')
                tf.summary.scalar('noise_std', noise_std)
                for it in range(len(tensor)):
                    if not it:
                        tensor[it] += tf.random_normal(tf.shape(tensor[it]), stddev=noise_std) * self.mask
                    else:
                        tensor[it] += tf.random_normal(tf.shape(tensor[it]), stddev=noise_std)

        # build discriminator
        return super(AltConvMVEEGAN, self)._build_discriminator(tensor=tensor, encoding_tensor=encoding_tensor,
                                                                training=training, batch_norm=batch_norm)

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


def safe_log(logit):
    return tf.log(tf.where(tf.equal(logit, 0.), tf.ones_like(logit), logit))


class NanInfException (Exception):
    pass


class EarlyStoppingException (Exception):
    pass


class ANNBase (object):

    name = 'ANNBase'

    def __init__(self, n_y, n_extra_discriminator_layers=0, use_batch_norm_D=False, name=None,
                 log_and_save=True, seed=None, early_stopping=False):
        # parameters
        self.n_noise = 100
        self.n_pixel = 32
        if not hasattr(self, 'n_channel'):
            self.n_channel = 1
        self.n_filtd = 128
        self.batch_norm_D = use_batch_norm_D
        if seed is None:
            seed = np.random.randint(int(1e8))
        self.seed = seed
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
        self.training = tf.placeholder(tf.bool, 0, 'training')
        self.input_x = tf.placeholder(tf.float32, (None, self.n_pixel, self.n_pixel, self.n_channel), 'image')
        self.input_n = tf.placeholder(tf.float32, (None, self.n_noise), 'noise')

        # label variables
        self.n_y = n_y
        self.input_y = tf.placeholder(tf.float32, (None, self.n_y), 'label')

        # logging'
        self.saver = None
        self.writer_train = None
        self.writer_test = None

        # etc
        self.session = None

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

    def pred(self, data):
        # setup
        n = data[0].shape[0]
        n_batch = 100
        pred = np.zeros((n, self.n_y))
        logits = np.zeros((n, self.n_y))

        # desired tensors
        pred_tensor = self.session.graph.get_tensor_by_name('discriminator/pred_probs:0')
        logits_tensor = self.session.graph.get_tensor_by_name('discriminator/pred_logits:0')

        for it in range(int(np.ceil(1. * n / n_batch))):
            # make feed_dict
            ind = np.arange(n_batch * it, np.minimum(n, n_batch * (it + 1)))
            feed_dict = {self.input_x: data[0][ind]}
            for it in range(len(self.additional_feature_names)):
                feed_dict[self.input_x_additional[it]] = data[it + 1][ind]

            # run prediction

            pred[ind], logits[ind] = self.session.run([pred_tensor, logits_tensor], feed_dict=feed_dict)

        return pred, logits


class MANN (ANNBase):

    name = 'BMANN'

    def __init__(self, n_y, additional_features=None, n_extra_discriminator_layers=0, use_batch_norm_D=False,
                 name=None, seed=np.random.randint(int(1e8)), log_and_save=True, early_stopping=False):

        # additional feature sets
        self.additional_feature_names = []
        self.additional_feature_dimensions = []
        self.input_x_additional = []
        if additional_features is not None:
            assert isinstance(additional_features, dict), 'additional_features must be of type dict or None'
            for key, val in additional_features.iteritems():
                assert isinstance(key, str), 'additional_features keys must be of type str'
                assert isinstance(val, int), 'additional_features keys must be of type int'
                self.additional_feature_names.append(key)
                self.additional_feature_dimensions.append(val)
                self.input_x_additional.append(tf.placeholder(tf.float32, (None, val), key))

        # init
        super(MANN, self).__init__(n_y, n_extra_discriminator_layers=n_extra_discriminator_layers,
                                   use_batch_norm_D=use_batch_norm_D,
                                   name=name, log_and_save=log_and_save, seed=seed, early_stopping=early_stopping)

    def _build_additional_dense_discriminator_base(self, tensor, name, n_hidden_layers=3, n_hidden_nodes=128,
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
                tensor_pre.append(self._build_additional_dense_discriminator_base(tensor[it + 1],
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
            d_out = self.n_y
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], d_out)):
                out_logits = tf.reshape(tf.layers.conv2d(tensor, d_out, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, d_out])

            return tf.nn.softmax(out_logits, name='pred_probs'), tf.identity(out_logits, name='pred_logits')

    def _build_loss(self, training=False):

        label, logits = self._build_discriminator(training=training)

        # discriminator
        lossD = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))

        # summaries
        if training:
            with tf.name_scope('training_metrics'):
                # classifier performance
                pred = tf.argmax(logits, 1)
                tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
                cmat = tf.reshape(tf.confusion_matrix(tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16),
                                  [1, self.n_y, self.n_y, 1])
                tf.summary.image('confusion_matrix', cmat)
                tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.001))
                # GAN loss
                tf.summary.scalar('lossD', lossD)

        else:
            collection = ['eval']
            update_collection = ['eval_update']
            true_val = tf.argmax(self.input_y, 1)
            pred = tf.argmax(logits, 1)

            with tf.name_scope('testing_metrics'):
                # classifier accuracy
                acc, update_eval_acc = tf.metrics.accuracy(true_val, pred, updates_collections=update_collection,
                                                           name='avgacc')
                tf.summary.scalar('accuracy', acc, collections=collection)
                # classifier confusion matrix
                cmat = tf.reshape(tf.confusion_matrix(true_val, pred, self.n_y, tf.float32), [1, self.n_y, self.n_y, 1])
                cmat, update_eval_cmat = tf.metrics.mean_tensor(cmat, updates_collections=update_collection, name='avgcmat')
                tf.summary.image('confusion_matrix', cmat, collections=collection)
                tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.001),
                                 collections=collection)
                # GAN loss
                mlossD, update_eval_lossD = tf.metrics.mean(lossD, updates_collections=update_collection, name='lossD')
                tf.summary.scalar('lossD', mlossD, collections=collection)
                # save eval updates
                for node in ['avgacc', 'avgcmat', 'lossD']:
                    for var in ['total:0', 'count:0', 'total_tensor:0', 'count_tensor:0']:
                        name = 'testing_metrics/' + node + '/' + var
                        try:
                            temp = tf.get_default_graph().get_tensor_by_name(name)
                        except KeyError:
                            continue
                        tf.add_to_collection('reset_eval', tf.assign(temp, tf.zeros_like(temp)))


        return lossD

    def _train(self, trainx, trainy, testx, testy, n_epochs=1000, n_batch=128, balance_labels=False, learning_rate=2e-4):

        # handle data
        n_train = trainx[0].shape[0]
        n_test = testx[0].shape[0]

        # setup learning
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossD = self._build_loss(training=True)
        evalD = self._build_loss()
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
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

            # train
            self._start_logging_and_saving(sess)
            # while True:
                # try:
            for epoch in range(n_epochs):
                # train on epoch
                start = time()
                step = 0
                n, ld = 0, 0
                for batch_index, n_batch_actual in batcher(n_train, n_batch):
                    # prep
                    n += n_batch_actual
                    if balance_labels:
                        randind = balanced_rand(trainy, n_batch_actual)
                        feed_dict = {self.input_x: trainx[0][randind],
                                     self.input_y: trainy[randind]}
                        for it in range(len(self.additional_feature_names)):
                            feed_dict[self.input_x_additional[it]] = trainx[it + 1][randind]
                    else:
                        feed_dict = {self.input_x: trainx[0][batch_index:batch_index + n_batch_actual],
                                     self.input_y: trainy[batch_index:batch_index + n_batch_actual]}
                        for it in range(len(self.additional_feature_names)):
                            feed_dict[self.input_x_additional[it]] = trainx[it + 1][batch_index:batch_index + n_batch_actual]
                    # discriminator
                    temp, summary, step = sess.run([adamD, merged_summary, global_step], feed_dict)
                    ld += temp * n_batch_actual
                    # generator
                    if np.isnan(ld) or np.isinf(ld):
                        raise NanInfException
                    if n % (2 * n_batch) == 0:
                        self._log(summary, step)
                        print 'epoch {:d}/{:d} (part {:d}/{:d}):  training loss: {:f}  time: {:d} seconds' \
                            .format(epoch + 1, n_epochs, n, n_train, ld / n, int(time() - start))
                # check performance after each epoch
                m, lde = 0, 0
                sess.run(tf.get_collection('reset_eval'))
                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                    m += n_batch_actual
                    feed_dict = {self.input_x: testx[0][batch_index:batch_index + n_batch_actual],
                                 self.input_y: testy[batch_index:batch_index + n_batch_actual]}
                    for it in range(len(self.additional_feature_names)):
                        feed_dict[self.input_x_additional[it]] = testx[it + 1][batch_index:batch_index + n_batch_actual]
                    out = sess.run([evalD, eval_summary] + tf.get_collection('eval_update'), feed_dict)
                    lde += out[0] * n_batch_actual
                self._log(out[1], step, test=True)
                print 'epoch {:d}/{:d}:  evaluation loss: {:f}  time: {:d} seconds' \
                    .format(epoch + 1, n_epochs, lde / m, int(time() - start))
                # save
                self._save(sess, step)
                self._early_stopping(lde / m, sess, step, feed_dict)
                # acc = sess.run(tf.get_default_graph().get_tensor_by_name('avgacc/value:0'))
                # self._early_stopping(-acc, sess, step, feed_dict)

                # except NanInfException:
                #     if epoch >= 10:
                #         a=1
                #     print 'Got NaNs or infs. Resetting parameters and starting again.'
                #     try:
                #         self._restore(sess)
                #     except:
                #         step = sess.run(global_step)
                #         sess.run(tf.global_variables_initializer())
                #         tf.assign(global_step, step)
                #     trainx, trainy = shuffle_data(trainx, trainy)
                #     testx, testy = shuffle_data(testx, testy)

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
            if name.startswith('discriminator/') \
                    and (name.endswith('BiasAdd') or name.endswith('Softmax')
                         or name.endswith('concat') or name.endswith('concat_1')
                         or name.endswith('Maximum') or name.endswith('Relu')):
                try:
                    name = name.replace('/', '__').replace('-', '_').replace('.', '_')
                    vals[name] = session.run(graph.get_tensor_by_name(op.name + ':0'), feed_dict=feed_dict)
                except:
                    pass

        # append inputs
        vals['in_image'] = feed_dict[self.input_x]
        vals['in_psd'] = feed_dict[graph.get_tensor_by_name('psd_med:0')]
        try:
            vals['in_autocorr'] = feed_dict[graph.get_tensor_by_name('autocorr:0')]
        except KeyError:
            pass

        assert vals, 'nothing saved'

        # save
        savemat(join(self.path, self.name, self.filename + '_test_vals'), vals, long_field_names=True)


class ConvMANN (MANN):

    name = 'ConvMANN'

    def _build_additional_dense_discriminator_base(self, tensor, name, n_hidden_layers=2, n_hidden_nodes=None,
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
                                               name='conv'), name='bn'))

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


class WeightedConvMANN (ConvMANN):

    name = 'WeightedConvMANN'

    def __init__(self, n_y, additional_features=None, n_extra_discriminator_layers=0, use_batch_norm_D=False,
                 name=None, seed=np.random.randint(int(1e8)), log_and_save=True, early_stopping=False,
                 weighting=None):

        if weighting is None:
            weighting = np.ones((n_y))
        self.weighting = weighting

        super(WeightedConvMANN, self).__init__(n_y, additional_features=additional_features,
                                               n_extra_discriminator_layers=n_extra_discriminator_layers,
                                               use_batch_norm_D=use_batch_norm_D,
                                               name=name, log_and_save=log_and_save, seed=seed,
                                               early_stopping=early_stopping)

    def _build_loss(self, training=False):

        label, logits = self._build_discriminator(training=training)

        # discriminator
        lossD = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=self.input_y,
                                                                        pos_weight=self.weighting))

        # summaries
        if training:
            with tf.name_scope('training_metrics'):
                # classifier performance
                pred = tf.argmax(logits, 1)
                tf.summary.scalar('accuracy', tf.contrib.metrics.accuracy(pred, tf.argmax(self.input_y, 1)))
                cmat = tf.reshape(tf.confusion_matrix(tf.argmax(self.input_y, 1), pred, self.n_y, tf.float16),
                                  [1, self.n_y, self.n_y, 1])
                tf.summary.image('confusion_matrix', cmat)
                tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.001))
                # GAN loss
                tf.summary.scalar('lossD', lossD)

        else:
            collection = ['eval']
            update_collection = ['eval_update']
            true_val = tf.argmax(self.input_y, 1)
            pred = tf.argmax(logits, 1)

            with tf.name_scope('testing_metrics'):
                # classifier accuracy
                acc, update_eval_acc = tf.metrics.accuracy(true_val, pred, updates_collections=update_collection,
                                                           name='avgacc')
                tf.summary.scalar('accuracy', acc, collections=collection)
                # classifier confusion matrix
                cmat = tf.reshape(tf.confusion_matrix(true_val, pred, self.n_y, tf.float32), [1, self.n_y, self.n_y, 1])
                cmat, update_eval_cmat = tf.metrics.mean_tensor(cmat, updates_collections=update_collection, name='avgcmat')
                tf.summary.image('confusion_matrix', cmat, collections=collection)
                tf.summary.image('confusion_matrix_normalized', cmat / tf.maximum(tf.reduce_sum(cmat, 2, True), 0.001),
                                 collections=collection)
                # GAN loss (weighted)
                mlossD_weighted, update_eval_lossD_weighted = tf.metrics.mean(lossD, updates_collections=update_collection,
                                                                              name='lossD_weighted')
                tf.summary.scalar('lossD_weighted', mlossD_weighted, collections=collection)
                # GAN loss (unweighted)
                mlossD, update_eval_lossD = tf.metrics.mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                                    labels=self.input_y),
                                                            updates_collections=update_collection, name='lossD')
                tf.summary.scalar('lossD', mlossD, collections=collection)
                # save eval updates
                for node in ['avgacc', 'avgcmat', 'lossD_weighted', 'lossD']:
                    for var in ['total:0', 'count:0', 'total_tensor:0', 'count_tensor:0']:
                        name = 'testing_metrics/' + node + '/' + var
                        try:
                            temp = tf.get_default_graph().get_tensor_by_name(name)
                        except KeyError:
                            continue
                        tf.add_to_collection('reset_eval', tf.assign(temp, tf.zeros_like(temp)))


        return lossD



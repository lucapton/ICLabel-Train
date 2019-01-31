from tfGAN_indvBN import *
from tfMLP import *
from icldata import ICLabelDataset  # available from https://github.com/lucapton/ICLabel-Dataset
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn import metrics
from os.path import isdir, join
import pandas as pd
from scipy import interp
from scipy.io import savemat
from matplotlib import pyplot as plt
import itertools


def ndarr2latex(arr, caption=None, label=None, row_names=None, col_names=None):
    n_row, n_col = arr.shape
    latex = [
        '\\begin{table}',
        '\\centering',
        '\\begin{tabular}{|' + 'c|' * (n_col + (row_names is not None)) + '}',
        '\\hline'
    ]
    if col_names is not None:
        latex.append(' & ' + ' & '.join(col_names) + ' \\\\')
        latex.append('\\hline')

    for it in range(n_row):
        strarr = ['{:.1f}'.format(x) for x in arr[it]]
        if row_names is not None:
            latex.append(' & '.join(row_names[it:it+1] + strarr) + ' \\\\')
        else:
            latex.append(' & '.join(strarr) + ' \\\\')
        latex.append('\\hline')

    latex.append('\\end{tabular}')

    if caption is not None:
        latex.append('\\caption{' + caption + '}')

    if label is not None:
        latex.append('\\label{' + label + '}')

    latex.append('\\end{table}')

    return latex


def make_cm_strong_and(label, pred):
    assert label.shape == pred.shape, 'label and prediciton must have same shape'
    return np.nansum(np.maximum(label[:, :,  np.newaxis] + pred[:, np.newaxis] - 1, 0), 0)


def make_cm_weak_and(label, pred):
    assert label.shape == pred.shape, 'label and prediciton must have same shape'
    return np.nansum(np.minimum(label[:, :,  np.newaxis], pred[:, np.newaxis]), 0)


def make_cm_prod(label, pred):
    assert label.shape == pred.shape, 'label and prediciton must have same shape'
    return np.nansum(label[:, :,  np.newaxis] * pred[:, np.newaxis], 0)


def make_cm_all(label, pred):
    # get raw soft confusion matices (cm)
    strong = make_cm_strong_and(label, pred)
    weak = make_cm_weak_and(label, pred)
    prod = make_cm_prod(label, pred)
    # combine strong and weak AND cms into optimistic and pessimistic cms
    cm_pes = weak.copy()
    np.fill_diagonal(cm_pes, np.diag(strong))
    cm_opt = strong.copy()
    np.fill_diagonal(cm_opt, np.diag(weak))

    return cm_pes, prod, cm_opt


def perf_soft(label, pred):
    n_cls = label.shape[1]
    cm_pes, cm_prod, cm_opt = make_cm_all(label, pred)

    ce = -np.nansum(label * np.log(pred), 1).mean()

    # get perf stats
    acc, pre, rec, spe = np.zeros(3), np.zeros((3, n_cls)), np.zeros((3, n_cls)), np.zeros((3, n_cls))
    for it, cm in enumerate((cm_pes, cm_prod, cm_opt)):
        acc[it] = np.diag(cm).sum() / cm.sum()
        pre[it, :] = np.diag(cm) / cm.sum(0)  # precision / PPV
        rec[it, :] = np.diag(cm) / cm.sum(1)  # recall / sensitivity / TPR
        spe[it, :] = (np.diag(cm).sum() - np.diag(cm)) / (cm.sum() - cm.sum(1))  # Specifity / 1 - FPR

    return acc, pre, rec, spe, ce


def perf_hard(labels, pred):
    # remove nans
    ind_keep = np.logical_not(np.isnan(pred).any(1))
    labels = labels[ind_keep]
    pred = pred[ind_keep]

    # get argmax
    label_argmax = labels.argmax(1)
    pred_argmax = pred.argmax(1)

    # get perf stats
    ce = -(labels * np.log(pred)).sum(1).mean()
    acc = metrics.accuracy_score(label_argmax, pred_argmax)
    pre = metrics.precision_score(label_argmax, pred_argmax, average=None)
    rec = metrics.recall_score(label_argmax, pred_argmax, average=None)
    auc = np.array([])

    # roc and prc
    roc = []
    prc = []
    thresh = np.zeros((1, pred.shape[1]))
    spacing = np.linspace(0, 1, 101)
    for it in range(pred.shape[1]):
        auc = np.append(auc, metrics.roc_auc_score(label_argmax == it, pred[:, it]))
        temp_roc = metrics.roc_curve(label_argmax == it, pred[:, it])
        roc.append([interp(spacing, temp_roc[2][::-1], x[::-1]) for x in temp_roc])
        temp_prc = metrics.precision_recall_curve(label_argmax == it, pred[:, it])
        temp_prc = temp_prc[:2] + (np.concatenate((temp_prc[2], [1])),)
        thresh[0, it] = temp_prc[2][np.argmax(f_beta_prc(temp_prc[0], temp_prc[1], 1))]
        prc.append([interp(spacing, temp_prc[2], x) for x in temp_prc])

    micro_pre = metrics.precision_score(label_argmax, pred_argmax, average='micro')
    micro_rec = metrics.recall_score(label_argmax, pred_argmax, average='micro')
    macro_pre = pre.mean()
    macro_rec = rec.mean()
    macro_auc = auc.mean()

    return thresh, ce, acc, pre, rec, auc, roc, prc, micro_pre, micro_rec, macro_pre, macro_rec, macro_auc


def soft_perf_plot(vals, classes=None):
    plt.figure()
    marker = itertools.cycle((',', '+', '.', 'o', '*'))
    for it, vals in enumerate(vals):
        plt.plot(vals, linestyle='', marker=marker.next())
        if classes is not None:
            plt.xticks(range(n_cls), labels=classes)


def soft_perf_plot2(vals, labels=None, new_fig=True):
    n_cls = vals.shape[1]
    if new_fig:
        plt.figure()
    plt.errorbar(range(n_cls), vals[1], yerr=np.abs(vals[1:2] - vals[[0, 2], :]))
    plt.xlim((0.1, n_cls + 0.1))
    plt.ylim((0, 1))
    if labels is not None:
        plt.xticks(range(n_cls), labels, rotation=20)


def reduce_labels(labels, n_cls):
    if n_cls == 2:
        labels = np.concatenate((labels[:, 0:1], labels[:, 1:].sum(1, keepdims=True)), 1)
    elif n_cls == 3:
        labels = np.concatenate((labels[:, 0:1], labels[:, 2:3], labels[:, [1, 3, 4, 5, 6]].sum(1, keepdims=True)), 1)
    elif n_cls == 5:
        labels = np.concatenate((labels[:, :4], labels[:, 4:].sum(1, keepdims=True)), 1)
    elif n_cls == 7:
        pass
    else:
        raise ValueError('n_cls must be 2, 3, or 5')
    return labels


def acc_cr(fpr, tpr, class_ratio=1):
    return (tpr + class_ratio * (1 - fpr)) / (1 + class_ratio)


def f_beta_roc(fpr, tpr, beta=1):
    return 2 * tpr / (tpr + beta * fpr + 1)


def f_beta_prc(precision, recall, beta):
    return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)


seed1 = 1979
seed2 = 1776
seed3 = 1492
n_folds = 10

icl_archs = [WeightedConvMANN, ConvMANN, AltConvMSSGAN]
ilc_methods = [x.name + ' w/ acor'*y for x, y in itertools.product(icl_archs, range(2))]
other_archs = ICLabelDataset().load_classifications(2, np.array([[1, 1]])).keys()
cls_map = {x: y for x, y in zip(ilc_methods + other_archs, range(len(ilc_methods + other_archs)))}
cls_imap = {y: x for x, y in cls_map.iteritems()}
cls_imap = [cls_imap[x] for x in range(len(cls_map))]

cols = ('n_cls', 'arch', 'fold',
        'cross_entropy', 'accuracy',
        'micro_precision', 'micro_recall',
        'macro_precision', 'macro_recall', 'macro_auc',
        'precision', 'recall', 'auc',
        'thresh', 'roc_fpr', 'roc_tpr', 'roc_thr', 'prc_pre', 'prc_rec', 'prc_thr',
        'soft_acccuracy_pessimistic', 'soft_acccuracy_expected', 'soft_acccuracy_optimistic',
        'soft_precision_pessimistic', 'soft_precision_expected', 'soft_precision_optimistic',
        'soft_recall_pessimistic', 'soft_recall_expected', 'soft_recall_optimistic',
        'soft_specificity_pessimistic', 'soft_specificity_expected', 'soft_specificity_optimistic')
scores = pd.DataFrame(columns=cols)
raw = {x: [[]] * n_folds for x in cls_imap}
raw.update({'label': [[]] * n_folds})
# train and extract performance statistics
for labels in ('all',):

    # load data
    icl = ICLabelDataset(label_type=labels, seed=seed1)
    icl_data = icl.load_semi_supervised()
    icl_data_val_labels = np.concatenate((icl_data[1][1][0], icl_data[3][1][0]), axis=0)
    icl_data_val_ilrlabels = np.concatenate((icl_data[1][1][1], icl_data[3][1][1]), axis=0)
    icl_data_val_ilrlabelscov = np.concatenate((icl_data[1][2][1], icl_data[3][2][1]), axis=0)

    # process topo maps
    topo_data = list()
    for it in range(4):
        temp = 0.99 * icl_data[it][0]['topo'] / np.abs(icl_data[it][0]['topo']).max(1, keepdims=True)
        topo_data.append(icl.pad_topo(temp).astype(np.float32).reshape(-1, 32, 32, 1))

    # generate mask
    mask = np.setdiff1d(np.arange(1024), icl.topo_ind)

    # K-fold
    kfold = StratifiedKFold(n_splits=n_folds, random_state=seed2)
    ind_fold = 0

    for ind_train_l, ind_test in kfold.split(icl_data_val_labels, icl_data_val_labels.argmax(1)):

        # create validation set
        sss = StratifiedShuffleSplit(1, len(ind_test), random_state=seed3)
        sss_gen = sss.split(icl_data_val_labels[ind_train_l], icl_data_val_labels[ind_train_l].argmax(1))
        ind_train_l_tr, ind_train_l_val = sss_gen.next()

        for use_autocorr in (False, True):

            # rescale features
            if use_autocorr:
                input_data = [[topo_data[x],
                               0.99 * icl_data[x][0]['psd'],
                               0.99 * icl_data[x][0]['autocorr'],
                               ] for x in range(4)]
            else:
                input_data = [[topo_data[x],
                               0.99 * icl_data[x][0]['psd'],
                               ] for x in range(4)]

            # create data fold
            temp = [np.concatenate((x, y), axis=0) for x, y in zip(input_data[1], input_data[3])]
            input_data[1] = [x[ind_train_l] for x in temp]                  # labeled train
            input_data[2] = [x[ind_train_l[ind_train_l_tr]] for x in temp]  # labeled train fold
            input_data[3] = [x[ind_train_l[ind_train_l_val]] for x in temp] # labeled validation fold
            input_data.append([x[ind_test] for x in temp])                  # test data
            test_ids = np.concatenate((icl_data[1][0]['ids'], icl_data[3][0]['ids']), axis=0)[ind_test]

            # create label fold
            train_labels = icl_data_val_labels[ind_train_l]
            train_labels_tr = icl_data_val_labels[ind_train_l[ind_train_l_tr]]
            train_labels_val = icl_data_val_labels[ind_train_l[ind_train_l_val]]
            test_labels = icl_data_val_labels[ind_test]

            train_ilrlabels = icl_data_val_ilrlabels[ind_train_l]
            train_ilrlabels_tr = icl_data_val_ilrlabels[ind_train_l[ind_train_l_tr]]
            train_ilrlabels_val = icl_data_val_ilrlabels[ind_train_l[ind_train_l_val]]
            test_ilrlabels = icl_data_val_ilrlabels[ind_test]

            train_ilrlabelscov = icl_data_val_ilrlabelscov[ind_train_l]
            train_ilrlabelscov_tr = icl_data_val_ilrlabelscov[ind_train_l[ind_train_l_tr]]
            train_ilrlabelscov_val = icl_data_val_ilrlabelscov[ind_train_l[ind_train_l_val]]
            test_ilrlabelscov = icl_data_val_ilrlabelscov[ind_test]

            # augment dataset by negating and/or horizontally flipping topo maps
            for it in range(5):
                input_data[it][0] = np.concatenate((input_data[it][0],
                                                    -input_data[it][0],
                                                    np.flip(input_data[it][0], 2),
                                                    -np.flip(input_data[it][0], 2)))
                for it2 in range(1, len(input_data[it])):
                    input_data[it][it2] = np.tile(input_data[it][it2], (4, 1))
            try:
                train_labels = np.tile(train_labels, (4, 1))
                train_labels_tr = np.tile(train_labels_tr, (4, 1))
                train_labels_val = np.tile(train_labels_val, (4, 1))
                test_labels = np.tile(test_labels, (4, 1))
                # ilr labels
                train_ilrlabels = np.tile(train_ilrlabels, (4, 1))
                train_ilrlabels_tr = np.tile(train_ilrlabels_tr, (4, 1))
                train_ilrlabels_val = np.tile(train_ilrlabels_val, (4, 1))
                test_ilrlabels = np.tile(test_ilrlabels, (4, 1))
                # ilr labels cov
                train_ilrlabelscov = np.tile(train_ilrlabelscov, (4, 1, 1))
                train_ilrlabelscov_tr = np.tile(train_ilrlabelscov_tr, (4, 1, 1))
                train_ilrlabelscov_val = np.tile(train_ilrlabelscov_val, (4, 1, 1))
                test_ilrlabelscov = np.tile(test_ilrlabelscov, (4, 1, 1))
            except ValueError:
                train_labels = 4 * train_labels
                train_labels_tr = 4 * train_labels_tr
                train_labels_val = 4 * train_labels_val
                test_labels = 4 * test_labels
                # ilr labels
                train_ilrlabels = 4 * train_ilrlabels
                train_ilrlabels_tr = 4 * train_ilrlabels_tr
                train_ilrlabels_val = 4 * train_ilrlabels_val
                test_ilrlabels = 4 * test_ilrlabels
                # ilr labels cov
                train_ilrlabelscov = 4 * train_ilrlabelscov
                train_ilrlabelscov_tr = 4 * train_ilrlabelscov_tr
                train_ilrlabelscov_val = 4 * train_ilrlabelscov_val
                test_ilrlabelscov = 4 * test_ilrlabelscov

            test_ids = np.tile(test_ids, (4, 1))

            # describe features and name
            additional_features = OrderedDict([('psd_med', input_data[1][1].shape[1])])
            name = 'ICLabel2_' + labels
            if use_autocorr:
                additional_features['autocorr'] = input_data[1][2].shape[1]
                name += '_autocorr'

            name += '_cv' + str(ind_fold)

            raw['label'][ind_fold] = test_labels

            for arch in icl_archs:

                # reset graph
                tf.reset_default_graph()

                if arch is ConvMANN:
                    # instantiate model
                    model = arch(icl_data[1][1][0].shape[1], additional_features=additional_features,
                                 early_stopping=True, name=name)

                    # check if already exists, if not train
                    if not isdir(join('output', arch.name, arch.name + '_' + name)):
                        model.train(input_data[2], train_labels_tr, input_data[3], train_labels_val,
                                    balance_labels=True, learning_rate=3e-4)

                elif arch is WeightedConvMANN:
                    # instantiate model
                    model = arch(icl_data[1][1][0].shape[1], additional_features=additional_features,
                                 early_stopping=True, name=name, weighting=np.array((2, 1, 1, 1, 1, 1, 1)))

                    # check if already exists, if not train
                    if not isdir(join('output', arch.name, arch.name + '_' + name)):
                        model.train(input_data[2], train_labels_tr, input_data[3], train_labels_val,
                                    balance_labels=True, learning_rate=3e-4)

                else:
                    # instantiate model
                    model = arch(icl_data[1][1][0].shape[1], additional_features=additional_features,
                                 mask=mask, early_stopping=True, name=name)

                    # check if already exists, if not train
                    if not isdir(join('output', arch.name, arch.name + '_' + name)):
                        model.train(input_data[0], input_data[2], train_labels_tr, input_data[3], train_labels_val,
                                    balance_labels=True, learning_rate=3e-4, label_strength=0.9, n_epochs=2)

                # calculate score
                model.load()
                out = model.pred(input_data[4])
                pred = out[0]
                if pred.shape[1] > 7:
                    pred = np.exp(out[1][:, :-1])
                    pred /= pred.sum(1, keepdims=True)
                for n_cls in (2, 3, 5, 7):
                    # get labels and predictions
                    temp_labels = reduce_labels(test_labels, n_cls)
                    temp_pred = reduce_labels(pred, n_cls)
                    # get perf stats
                    thresh, ce, acc, pre, rec, auc, roc, prc, micro_pre, micro_rec, macro_pre, macro_rec, macro_auc \
                        = perf_hard(temp_labels, temp_pred)
                    soft_acc, soft_pre, soft_rec, soft_spe, _ = perf_soft(temp_labels, temp_pred)
                    scores = scores.append(pd.DataFrame([[n_cls, arch.name + ' w/ acor' * use_autocorr, ind_fold, ce, acc,
                                                          micro_pre, micro_rec, macro_pre, macro_rec, macro_auc,
                                                          pre, rec, auc, thresh,
                                                          [x[0] for x in roc], [x[1] for x in roc], [x[2] for x in roc],
                                                          [x[0] for x in prc], [x[1] for x in prc], [x[2] for x in prc],
                                                          soft_acc[0], soft_acc[1], soft_acc[2],
                                                          soft_pre[0], soft_pre[1], soft_pre[2],
                                                          soft_rec[0], soft_rec[1], soft_rec[2],
                                                          soft_spe[0], soft_spe[1], soft_spe[2],
                                                          ]], columns=cols))
                raw[arch.name + ' w/ acor' * use_autocorr][ind_fold] = pred

            # compare to previous classifiers
            if use_autocorr:
                for n_cls in (2, 3, 5):
                    # get labels and predictions
                    temp_labels = reduce_labels(test_labels, n_cls)
                    other_cls = icl.load_classifications(n_cls, test_ids)
                    # get perf stats for each classifier
                    for cls, lab in other_cls.iteritems():
                        raw[cls][ind_fold] = lab
                        if cls == 'eye_catch':
                            continue
                        lab = np.concatenate((lab[np.logical_not(np.isnan(lab).any(1))],)*4)
                        thresh, ce, acc, pre, rec, auc, roc, prc, micro_pre, micro_rec, macro_pre, macro_rec, macro_auc \
                            = perf_hard(temp_labels, lab)
                        soft_acc, soft_pre, soft_rec, soft_spe, _ = perf_soft(temp_labels, lab)
                        scores = scores.append(pd.DataFrame([[n_cls, cls, ind_fold, ce, acc,
                                                              micro_pre, micro_rec, macro_pre, macro_rec, macro_auc,
                                                              pre, rec, auc, thresh,
                                                              [x[0] for x in roc], [x[1] for x in roc], [x[2] for x in roc],
                                                              [x[0] for x in prc], [x[1] for x in prc], [x[2] for x in prc],
                                                              soft_acc[0], soft_acc[1], soft_acc[2],
                                                              soft_pre[0], soft_pre[1], soft_pre[2],
                                                              soft_rec[0], soft_rec[1], soft_rec[2],
                                                              soft_spe[0], soft_spe[1], soft_spe[2],
                                                              ]], columns=cols))
        ind_fold += 1
scores = scores.reset_index(drop=True)

# save raw predictions and labels
savemat('output/cv_raw', {x.replace(' ', '_').replace('/', ''): y for x, y in raw.iteritems()})

# beautify names
better_names = {
    'FixedWeightedConvMANN': 'ICLabel wCNN2',
    'WeightedConvMANN': 'ICLabel wCNN',
    'ConvMANN': 'ICLabel CNN',
    'AltConvMSSGAN': 'ICLabel GAN',
    'FixedWeightedConvMANN w/ acor': 'ICLabel wCNN2(ac)',
    'WeightedConvMANN w/ acor': 'ICLabel wCNN(ac)',
    'ConvMANN w/ acor': 'ICLabel CNN(ac)',
    'AltConvMSSGAN w/ acor': 'ICLabel GAN(ac)',
    u'ic_marc': 'IC_MARC',
    u'adjust': 'ADJUST',
    u'mara': 'MARA',
    u'faster': 'FASTER',
}
for it in range(scores.shape[0]):
    scores.at[it, 'arch'] = better_names[scores.at[it, 'arch']]

# setup
marker = ('x', 'o', '*')
label_ind = {2: [0, 6], 3: [0, 2, 6], 5: [0, 1, 2, 3, 6], 7: slice(None)}
classes = np.array(('Brain', 'Muscle', 'Eye', 'Heart', 'Line Noise', 'Chan Noise', 'Other'))
colors = itertools.cycle(('r', 'g', 'b', 'm', 'c'))
linesty = itertools.cycle(('-', '--'))
color_and_linesty = {arch: (colors.next(), linesty.next()) for arch in np.sort(scores['arch'].unique())}

# learn thresholds
import re
from scipy.io import savemat
best_thresh = {}
for it_n, n_cls in enumerate((2, 3, 5, 7)):
    for it_cls in range(n_cls):
        tscores = scores[scores['n_cls'] == n_cls].applymap(
            lambda x, y=it_cls: np.array(x[y]) if isinstance(x, list) else x)
        gb = tscores.groupby(('arch',))
        for grp in gb.groups:
            vals = gb.get_group(grp).values[:, np.in1d(tscores.columns, ('roc_fpr', 'roc_tpr', 'roc_thr'))]
            roc_mean, roc_std = [], []
            for it in range(3):
                roc_mean.append(np.mean([x for x in vals[:, it]], axis=0))
                roc_std.append(np.std([x for x in vals[:, it]], axis=0))
            grp = re.sub('[^0-9a-zA-Z_]', '', grp)
            try:
                best_thresh[grp][it_n][it_cls] = roc_mean[2][np.argmax(f_beta_roc(roc_mean[0], roc_mean[1], 1))]
                # best_thresh[grp][it_n][it_cls] = roc_mean[2][np.argmax(acc_cr(roc_mean[0], roc_mean[1], 1))]
            except (IndexError, KeyError):
                try:
                    best_thresh[grp].append(np.zeros(n_cls))
                except (IndexError, KeyError):
                    best_thresh[grp] = [np.zeros(n_cls)]
                best_thresh[grp][it_n][it_cls] = roc_mean[2][np.argmax(f_beta_roc(roc_mean[0], roc_mean[1], 1))]
                # best_thresh[grp][it_n][it_cls] = roc_mean[2][np.argmax(acc_cr(roc_mean[0], roc_mean[1], 1))]
savemat('output/thresholds.mat', best_thresh)

# make figures
n_cls = 2
for meas in ('cross_entropy', 'accuracy', #  'micro_precision', 'micro_recall',
             'macro_precision', 'macro_recall', 'macro_auc'):
    try:
        # scores.boxplot(meas, ('autocorr', 'balance_labels', 'arch'))
        scores[scores['n_cls'] == n_cls].boxplot(meas, ('arch',))
    except TypeError:
        pass
latex_hard_perf = scores[scores['n_cls'] == n_cls].pivot_table(('cross_entropy', 'accuracy', 'macro_precision',
                                                                'macro_recall', 'macro_auc'),
                                                               ('n_cls', 'arch'), aggfunc=(np.mean, np.std)
                                                               ).to_latex(float_format='%0.3f')

# Code for plots
# # just brain
# for it in range(1):
#     def ind(y, k=it):
#         if isinstance(y, np.ndarray):
#             return y[k]
#         else:
#             return y
#
#     for meas in ('precision', 'recall', 'auc'):
#         tscores = scores.applymap(ind)
#         try:
#             tscores.boxplot(meas, ('arch',))
#         except TypeError:
#             pass
# latex_hard_perf_brain = scores.applymap(lambda x: np.array(x[0]) if isinstance(x, np.ndarray) else x)\
#     .pivot_table(('precision', 'recall', 'auc'), ('n_cls', 'arch'),
#                  aggfunc=(np.mean, np.std)).to_latex(float_format='%0.3f')
#
# # 2d roc brain across methods
# tscores = scores.applymap(lambda x: np.array(x[0]) if isinstance(x, list) else x)
# gb = tscores.groupby(('arch',))
# colors = itertools.cycle(('r', 'g', 'b', 'm', 'c'))
# linesty = itertools.cycle(('-', '--'))
# class_ratio = 1
# accs = np.linspace(0.6, 0.9, num=4)
# roc_mean, roc_std = {}, {}
# best_scores_roc = {}
# fig = plt.figure()
# ax = fig.gca()
# for acc in accs:
#     x = np.linspace(0, 1, 101)
#     # y = (class_ratio + x + (class_ratio + 1) * acc) / class_ratio
#     y = (class_ratio + 1) * acc - (1 - x) * class_ratio
#     l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#     ax.annotate('acc{}={:0.1f}'.format(class_ratio, acc), xy=(0.05, y[4] + 0.02))
# # for f in accs:
# #     x = np.linspace(0, 1, 101)
# #     # y = (class_ratio + x + (class_ratio + 1) * acc) / class_ratio
# #     y = f * (class_ratio * x + 1) / (2 - f)
# #     l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
# #     plt.annotate('f{}={:0.1f}'.format(class_ratio, acc), xy=(0.05, y[4] + 0.02))
# for grp in gb.groups:
#     vals = gb.get_group(grp).values[:, np.in1d(tscores.columns, ('roc_fpr', 'roc_tpr', 'roc_thr'))]
#     roc_mean[grp], roc_std[grp] = [], []
#     for it in range(3):
#         roc_mean[grp].append(np.mean([x for x in vals[:, it]], axis=0))
#         roc_std[grp].append(np.std([x for x in vals[:, it]], axis=0))
#
#     color = colors.next()
#     lsty = linesty.next()
#     ax.plot(roc_mean[grp][0], roc_mean[grp][1],
#              label=grp,
#              color=color, linestyle=lsty,
#              lw=2, alpha=.8)
#     ind = np.maximum(np.argmax(roc_mean[grp][2] <= 0.5) - 1, 0)
#     ax.scatter(roc_mean[grp][0][ind], roc_mean[grp][1][ind])
#     # thresh at best acc score
#     # ind = np.argmax(f_beta_roc(roc_mean[grp][0], roc_mean[grp][1], class_ratio))
#     ind = np.argmax(acc_cr(roc_mean[grp][0], roc_mean[grp][1], class_ratio))
#     ax.scatter(roc_mean[grp][0][ind], roc_mean[grp][1][ind], c='g', edgecolors='g')
#     best_scores_roc[grp] = (acc_cr(roc_mean[grp][0][ind], roc_mean[grp][1][ind], class_ratio), roc_mean[grp][2][ind])
#     ax.annotate('{:0.2f}'.format(roc_mean[grp][2][ind]), xy=(roc_mean[grp][0][ind], roc_mean[grp][1][ind]))
#     # plt.fill_between(roc_mean[grp][0], roc_mean[grp][1] - roc_std[grp][1], roc_mean[grp][1] + roc_std[grp][1],
#     #                  alpha=.2)
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Brain ROC')
# plt.legend(loc="lower right")
# plt.show()
#
# # 3d roc
# tscores = scores.applymap(lambda x: np.array(x[0]) if isinstance(x, list) else x)
# gb = tscores.groupby(('arch',))
# roc_mean, roc_std = {}, {}
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for grp in gb.groups:
#     vals = gb.get_group(grp).values[:, np.in1d(tscores.columns, ('roc_fpr', 'roc_tpr', 'roc_thr'))]
#     roc_mean[grp], roc_std[grp] = [], []
#     for it in range(3):
#         roc_mean[grp].append(np.mean([x for x in vals[:, it]], axis=0))
#         roc_std[grp].append(np.std([x for x in vals[:, it]], axis=0))
#     ax.plot(roc_mean[grp][0], roc_mean[grp][1], np.minimum(roc_mean[grp][2], 1),
#              label=str(grp) + ' ROC',  # (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#              lw=2, alpha=.8)
#     ind = np.maximum(np.argmax(roc_mean[grp][2] <= 0.5) - 1, 0)
#     ax.scatter(roc_mean[grp][0][ind], roc_mean[grp][1][ind], np.minimum(roc_mean[grp][2][ind], 1))
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# ax.set_zlim([0, 1])
# ax.set_xlabel('False Positive Rate')
# ax.set_ylabel('True Positive Rate')
# ax.set_zlabel('Threshold')
# plt.title('Brain ROC')
# plt.legend(loc="lower right")
# plt.show()
#
# # roc dist
# plt.figure()
# for grp in gb.groups:
#     plt.plot(np.minimum(roc_mean[grp][2], 1), np.linalg.norm(np.stack((1 - roc_mean[grp][1], roc_mean[grp][0])).T, axis=1),
#              label=str(grp) + ' ROC',
#              lw=2, alpha=.8)
#
# # 2d prc brain across methods
# gb = scores[scores['n_cls'] == 2].applymap(lambda x: np.array(x[0]) if isinstance(x, list) else x).groupby(('arch',))
# colors = itertools.cycle(('r', 'g', 'b', 'm', 'c'))
# linesty = itertools.cycle(('-', '--'))
# prc_mean, prc_std = {}, {}
# best_scores_prc = {}
# f_scores = np.linspace(0.6, 0.9, num=4)
# beta = 1
# plt.figure()
# for f_score in f_scores:
#     x = np.linspace(0, 1, 101)
#     y = f_score * x / ((1 + beta**2) * x - f_score * beta**2)
#     l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#     plt.annotate('f{}={:0.1f}'.format(beta, f_score), xy=(0.9, y[45] + 0.02))
# for grp in gb.groups:
#     vals = gb.get_group(grp).values[:, np.in1d(gb.get_group(grp).columns, ('prc_pre', 'prc_rec', 'prc_thr'))]
#     prc_mean[grp], prc_std[grp] = [], []
#     for it in range(3):
#         prc_mean[grp].append(np.mean([x for x in vals[:, it]], axis=0))
#         prc_std[grp].append(np.std([x for x in vals[:, it]], axis=0))
#
#     color = colors.next()
#     lsty = linesty.next()
#     plt.plot(prc_mean[grp][1], prc_mean[grp][0],
#              label=grp,
#              color=color, linestyle=lsty,
#              lw=2, alpha=.8)
#     # thresh at 0.5
#     ind = np.maximum(np.argmax(prc_mean[grp][2] >= 0.5) - 1, 0)
#     plt.scatter(prc_mean[grp][1][ind], prc_mean[grp][0][ind])
#     # thresh at best f_2 score
#     ind = np.argmax(f_beta_prc(prc_mean[grp][0], prc_mean[grp][1], beta))
#     plt.scatter(prc_mean[grp][1][ind], prc_mean[grp][0][ind], c='g', edgecolors='g')
#     best_scores_prc[grp] = (f_beta_prc(prc_mean[grp][0][ind], prc_mean[grp][1][ind], beta), prc_mean[grp][2][ind])
#     plt.annotate('{:0.2f}'.format(prc_mean[grp][2][ind]), xy=(prc_mean[grp][1][ind], prc_mean[grp][0][ind]))
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Brain PRC')
# plt.legend(loc="lower left")
# plt.show()
#
# # 3d roc-prc
# tscores = scores.applymap(lambda x: np.array(x[0]) if isinstance(x, list) else x)
# gb = tscores.groupby(('arch',))
# roc_mean, roc_std = {}, {}
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for grp in gb.groups:
#     vals = gb.get_group(grp).values[:, np.in1d(tscores.columns, ('roc_fpr', 'roc_tpr', 'roc_thr', 'prc_pre', 'prc_rec', 'prc_thr'))]
#     roc_mean[grp], roc_std[grp] = [], []
#     for it in range(len(vals)):
#         vals[it][0] = interp(np.linspace(0, 1, 101), vals[it][2][::-1], vals[it][0][::-1])
#         vals[it][1] = interp(np.linspace(0, 1, 101), vals[it][2][::-1], vals[it][1][::-1])
#         vals[it][2] = np.linspace(0, 1, 101)
#         vals[it][3] = interp(np.linspace(0, 1, 101), vals[it][5], vals[it][0])
#         vals[it][4] = interp(np.linspace(0, 1, 101), vals[it][5], vals[it][1])
#         vals[it][5] = np.linspace(0, 1, 101)
#     for it in range(6):
#         roc_mean[grp].append(np.mean([x for x in vals[:, it]], axis=0))
#         roc_std[grp].append(np.std([x for x in vals[:, it]], axis=0))
#     ax.plot(roc_mean[grp][0], roc_mean[grp][3], np.minimum(roc_mean[grp][1], 1),
#              label=str(grp) + ' ROC',  # (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#              lw=2, alpha=.8)
#     ind = np.maximum(np.argmax(roc_mean[grp][2] <= 0.5) - 1, 0)
#     ax.scatter(roc_mean[grp][0][ind], roc_mean[grp][3][ind], np.minimum(roc_mean[grp][1][ind], 1))
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# ax.set_zlim([0, 1])
# ax.set_xlabel('False Positive Rate')
# ax.set_ylabel('Precision')
# ax.set_zlabel('True Positive Rate / Recall')
# plt.title('Brain ROC')
# plt.legend(loc="lower right")
# plt.show()
#
# # soft vals
# n_cls = 5
# tscores = scores[scores['n_cls'] == n_cls].applymap(lambda x: np.array(x[0]) if isinstance(x, list) else x)
# gb = tscores.groupby(('arch',))
# for measure in ('precision', 'recall', 'specificity'):
#     plt.figure()
#     n = 0
#     for grp in gb.groups:
#         pes = gb.get_group(grp)['soft_' + measure + '_pessimistic'].mean(0)
#         exp = gb.get_group(grp)['soft_' + measure + '_expected'].mean(0)
#         opt = gb.get_group(grp)['soft_' + measure + '_optimistic'].mean(0)
#
#         plt.errorbar(np.arange(n_cls) + n / 30., exp, yerr=np.stack((exp - pes, opt - exp)), label=str(grp))
#         n += 1
#     plt.title('Soft ' + measure)
#     plt.xticks(range(n_cls), classes[label_ind[n_cls]], rotation=20)
#     plt.xlim((-0.5, n_cls - 0.5))
#     plt.ylim((0, 1))
#     plt.legend(loc="lower right")
#     plt.show()
# soft_cols = [x for x in scores.columns if x.startswith('soft')]
# soft_cols_short = ['_'.join([y[:3] for y in x.split('_')[1:]]) for x in soft_cols]
# # soft_cols_short = ['_'.join([y if not it else y[:3] for it, y in enumerate(x.split('_')[1:])]) for x in soft_cols]
# rename = {x: y for x, y in zip(soft_cols, soft_cols_short)}
# latex_soft_perf_brain = scores.applymap(lambda x: np.array(x[0]) if isinstance(x, np.ndarray) else x)\
#     .pivot_table([x for x in scores.columns if x.startswith('soft')], ('n_cls', 'arch'),
#                  aggfunc=(np.mean,)).rename_axis(rename, axis=1).to_latex(float_format='%0.3f')
#
# # soft ROC
# n_cls = 5
# for cls in range(n_cls):
#     gb = scores[scores['n_cls'] == n_cls].groupby(('arch',))
#     roc_mean, roc_std = {}, {}
#     best_scores_roc = {}
#     accs = np.linspace(0.6, 0.9, num=4)
#     class_ratio = 1
#     plt.figure()
#     # for acc in accs:
#     #     x = np.linspace(0, 1., 101)
#     #     y = (class_ratio + 1) * acc - (1 - x) * class_ratio
#     #     l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#     #     plt.annotate('acc{}={:0.1f}'.format(class_ratio, acc), xy=(0.05, y[4] + 0.02))
#     for f in accs:
#         x = np.linspace(0, 1, 101)
#         y = f * (class_ratio * x + 1) / (2 - f)
#         l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#         plt.annotate('f{}={:0.1f}'.format(class_ratio, acc), xy=(0.05, y[4] + 0.02))
#     for grp in gb.groups:
#         color, linesty = color_and_linesty[grp]
#         # soft values
#         fpr = 1 - np.stack((gb.get_group(grp)['soft_specificity_pessimistic'].mean(0),
#                             gb.get_group(grp)['soft_specificity_expected'].mean(0),
#                             gb.get_group(grp)['soft_specificity_optimistic'].mean(0)))[:, cls]
#         tpr = np.stack((gb.get_group(grp)['soft_recall_pessimistic'].mean(0),
#                         gb.get_group(grp)['soft_recall_expected'].mean(0),
#                         gb.get_group(grp)['soft_recall_optimistic'].mean(0)))[:, cls]
#         plt.plot(fpr, tpr, color=color, linestyle=linesty)
#         for it in range(3):
#             plt.scatter(fpr[it], tpr[it], 40, color=color, marker=marker[it])
#         # hard values
#         if np.any(np.diff(fpr)):
#             auc = np.mean(gb.get_group(grp).values[:, np.in1d(gb.get_group(grp).columns, ('auc',))], 0)[0][cls]
#             vals = gb.get_group(grp).values[:, np.in1d(gb.get_group(grp).columns, ('roc_fpr', 'roc_tpr', 'roc_thr'))]
#             roc_mean[grp], roc_std[grp] = [], []
#             for it in range(3):
#                 roc_mean[grp].append(np.mean([x[cls] for x in vals[:, it]], axis=0))
#                 roc_std[grp].append(np.std([x[cls] for x in vals[:, it]], axis=0))
#         else:
#             roc_mean[grp] = [
#                 np.array([0, fpr[1], 1]),
#                 np.array([0, tpr[1], 1]),
#                 np.array([0, 0.5, 1]),
#                 ]
#         plt.plot(roc_mean[grp][0], roc_mean[grp][1],
#                  label=grp + ' (AUC {:.3f})'.format(auc),
#                  color=color, linestyle=linesty,
#                  lw=2, alpha=.8)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Soft {} ROC'.format(classes[label_ind[n_cls]][cls]))
#     plt.xlim((0, 1))
#     plt.ylim((0, 1))
#     handles, labels = plt.gca().get_legend_handles_labels()
#     ord = np.argsort(labels)
#     plt.legend(np.array(handles)[ord], np.array(labels)[ord], loc="lower right")
#     plt.show()
#     fig = plt.gcf()
#     fig.set_size_inches((10, 10), forward=False)
#     fig.savefig('output/figures/{}cls_softROC_{}.png'.format(n_cls, classes[label_ind[n_cls]][cls].lower()), dpi=150, format='png')
#
# # latex_soft_perf = scores.pivot_table([x for x in scores.columns if x.startswith('soft')], ('n_cls', 'arch', 'autocorr'),
# #                                      aggfunc=(np.mean, np.std)).to_latex(float_format='%0.3f')
#
# # soft PRC
# n_cls = 5
# for cls in range(n_cls):
#     gb = scores[scores['n_cls'] == n_cls].groupby(('arch',))
#     prc_mean, prc_std = {}, {}
#     best_scores_prc = {}
#     f_scores = np.linspace(0.6, 0.9, num=4)
#     beta = 1
#     plt.figure()
#     for f_score in f_scores:
#         x = np.linspace(0, 1, 101)
#         y = f_score * x / ((1 + beta ** 2) * x - f_score * beta ** 2)
#         l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#         plt.annotate('f{}={:0.1f}'.format(beta, f_score), xy=(0.9, y[45] + 0.02))
#     for grp in gb.groups:
#         # hard values
#         vals = gb.get_group(grp).values[:, np.in1d(gb.get_group(grp).columns, ('prc_pre', 'prc_rec', 'prc_thr'))]
#         prc_mean[grp], prc_std[grp] = [], []
#         for it in range(3):
#             prc_mean[grp].append(np.mean([x[cls] for x in vals[:, it]], axis=0))
#             prc_std[grp].append(np.std([x[cls] for x in vals[:, it]], axis=0))
#
#         color, linesty = color_and_linesty[grp]
#         plt.plot(prc_mean[grp][1], prc_mean[grp][0],
#                  label=grp,
#                  color=color, linestyle=linesty,
#                  lw=2, alpha=.8)
#
#         # soft values
#         pre = np.stack((gb.get_group(grp)['soft_precision_pessimistic'].mean(0),
#                         gb.get_group(grp)['soft_precision_expected'].mean(0),
#                         gb.get_group(grp)['soft_precision_optimistic'].mean(0)))[:, cls]
#         rec = np.stack((gb.get_group(grp)['soft_recall_pessimistic'].mean(0),
#                         gb.get_group(grp)['soft_recall_expected'].mean(0),
#                         gb.get_group(grp)['soft_recall_optimistic'].mean(0)))[:, cls]
#
#         plt.plot(rec, pre, color=color, linestyle=linesty)
#         for it in range(3):
#             plt.scatter(rec[it], pre[it], 40, color=color, marker=marker[it])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Soft {} PRC'.format(classes[label_ind[n_cls]][cls]))
#     plt.xlim((0, 1))
#     plt.ylim((0, 1))
#     handles, labels = plt.gca().get_legend_handles_labels()
#     ord = np.argsort(labels)
#     plt.legend(np.array(handles)[ord], np.array(labels)[ord], loc="lower right")
#     plt.show()
#     fig = plt.gcf()
#     fig.set_size_inches((10, 10), forward=False)
#     fig.savefig('output/figures/{}cls_softPRC_{}.png'.format(n_cls, classes[label_ind[n_cls]][cls].lower()), dpi=150, format='png')
#
# # soft ROC with err
# n_cls = 7
# for cls in range(n_cls):
#     marker = ('x', 'o', '*')
#     marker_labels = ('Optimistic', 'Expected', 'Pessimistic')
#     colors = itertools.cycle(('r', 'g', 'b', 'm', 'c'))
#     linesty = itertools.cycle(('-', '--'))
#     gb = scores[scores['n_cls'] == n_cls].groupby(('arch',))
#     roc_mean, roc_std = {}, {}
#     accs = np.linspace(0.6, 0.9, num=4)
#     class_ratio = 1
#     plt.figure()
#     # for acc in accs:
#     #     x = np.linspace(0, 1, 101)
#     #     # y = (class_ratio + x + (class_ratio + 1) * acc) / class_ratio
#     #     y = (class_ratio + 1) * acc - (1 - x) * class_ratio
#     #     l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#     #     plt.annotate('acc{}={:0.1f}'.format(class_ratio, acc), xy=(0.05, y[4] + 0.02))
#     for f in accs:
#         x = np.linspace(0, 1, 101)
#         # y = (class_ratio + x + (class_ratio + 1) * acc) / class_ratio
#         y = f * (class_ratio * x + 1) / (2 - f)
#         l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#         plt.annotate('f{}={:0.1f}'.format(class_ratio, acc), xy=(0.05, y[4] + 0.02))
#     for grp in ('ICLabel wCNN(ac)', 'ICLabel wCNN'):
#         # hard values
#         if np.any(np.diff(fpr)):
#             auc = np.mean(gb.get_group(grp).values[:, np.in1d(gb.get_group(grp).columns, ('auc',))], 0)[0][cls]
#             vals = gb.get_group(grp).values[:, np.in1d(gb.get_group(grp).columns, ('roc_fpr', 'roc_tpr', 'roc_thr'))]
#             roc_mean[grp], roc_std[grp] = [], []
#             for it in range(3):
#                 roc_mean[grp].append(np.mean([x[cls] for x in vals[:, it]], axis=0))
#                 roc_std[grp].append(np.std([x[cls] for x in vals[:, it]], axis=0))
#         else:
#             roc_mean[grp] = [
#                 np.array([0, fpr[1], 1]),
#                 np.array([0, tpr[1], 1]),
#                 np.array([0, 0.5, 1]),
#             ]
#         # plot
#         plt.errorbar(roc_mean[grp][0], roc_mean[grp][1], xerr=roc_std[grp][0], yerr=roc_std[grp][1],
#                      label=grp + ' (hard)',
#                      color='k', linestyle='-', ecolor='r',
#                      lw=2, alpha=.8, errorevery=10)
#
#         # soft values
#         tpr_mean = np.stack((gb.get_group(grp)['soft_recall_pessimistic'].mean(0),
#                              gb.get_group(grp)['soft_recall_expected'].mean(0),
#                              gb.get_group(grp)['soft_recall_optimistic'].mean(0)))[:, cls]
#         fpr_mean = 1 - np.stack((gb.get_group(grp)['soft_specificity_pessimistic'].mean(0),
#                                  gb.get_group(grp)['soft_specificity_expected'].mean(0),
#                                  gb.get_group(grp)['soft_specificity_optimistic'].mean(0)))[:, cls]
#         tpr_std = np.stack((np.stack(gb.get_group(grp)['soft_recall_pessimistic'].values).std(0),
#                             np.stack(gb.get_group(grp)['soft_recall_expected'].values).std(0),
#                             np.stack(gb.get_group(grp)['soft_recall_optimistic'].values).std(0)))[:, cls]
#         fpr_std = np.stack((np.stack(gb.get_group(grp)['soft_specificity_pessimistic'].values).std(0),
#                             np.stack(gb.get_group(grp)['soft_specificity_expected'].values).std(0),
#                             np.stack(gb.get_group(grp)['soft_specificity_optimistic'].values).std(0)))[:, cls]
#         # plot
#         plt.errorbar(fpr_mean, tpr_mean, xerr=fpr_std, yerr=tpr_std,
#                      label=grp + ' (soft)',
#                      color='k', linestyle='-', ecolor='g',
#                      lw=2, alpha=.8)
#         for it in range(3):
#             plt.scatter(fpr_mean[it], tpr_mean[it], 100,
#                         color='k', marker=marker[it], label=marker_labels[it])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('{} ROC With Errorbars'.format(classes[label_ind[n_cls]][cls]))
#     plt.xlim((0, 1))
#     plt.ylim((0, 1))
#     plt.legend(loc="lower right", scatterpoints=1)
#     plt.show()
#
# # soft PRC w/ err
# cls = 0
# marker = ('x', 'o', '*')
# marker_labels = ('Optimistic', 'Expected', 'Pessimistic')
# gb = scores[scores['n_cls'] == 2].applymap(lambda x: np.array(x[0]) if isinstance(x, list) else x).groupby(('arch',))
# prc_mean, prc_std = {}, {}
# f_scores = np.linspace(0.6, 0.9, num=4)
# beta = 1
# plt.figure()
# for f_score in f_scores:
#     x = np.linspace(0, 1, 101)
#     y = f_score * x / ((1 + beta**2) * x - f_score * beta**2)
#     l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#     plt.annotate('f{}={:0.1f}'.format(beta, f_score), xy=(0.9, y[45] + 0.02))
# for grp in ('ConvMANN w/ acor',):
#     # hard values
#     vals = gb.get_group(grp).values[:, np.in1d(gb.get_group(grp).columns, ('prc_pre', 'prc_rec', 'prc_thr'))]
#     prc_mean[grp], prc_std[grp] = [], []
#     for it in range(3):
#         prc_mean[grp].append(np.mean([x for x in vals[:, it]], axis=0))
#         prc_std[grp].append(np.std([x for x in vals[:, it]], axis=0))
#     # plot
#     plt.errorbar(prc_mean[grp][0], prc_mean[grp][1], xerr=prc_std[grp][0], yerr=prc_std[grp][1],
#                  label=grp + ' (hard)',
#                  color='k', linestyle='-', ecolor='r',
#                  lw=2, alpha=.8, errorevery=10)
#
#     # soft values
#     rec_mean = np.stack((gb.get_group(grp)['soft_recall_pessimistic'].mean(0),
#                          gb.get_group(grp)['soft_recall_expected'].mean(0),
#                          gb.get_group(grp)['soft_recall_optimistic'].mean(0)))[:, cls]
#     pre_mean = np.stack((gb.get_group(grp)['soft_precision_pessimistic'].mean(0),
#                          gb.get_group(grp)['soft_precision_expected'].mean(0),
#                          gb.get_group(grp)['soft_precision_optimistic'].mean(0)))[:, cls]
#     rec_std = np.stack((np.stack(gb.get_group(grp)['soft_recall_pessimistic'].values).std(0),
#                         np.stack(gb.get_group(grp)['soft_recall_expected'].values).std(0),
#                         np.stack(gb.get_group(grp)['soft_recall_optimistic'].values).std(0)))[:, cls]
#     pre_std = np.stack((np.stack(gb.get_group(grp)['soft_precision_pessimistic'].values).std(0),
#                         np.stack(gb.get_group(grp)['soft_precision_expected'].values).std(0),
#                         np.stack(gb.get_group(grp)['soft_precision_optimistic'].values).std(0)))[:, cls]
#     # plot
#     plt.errorbar(rec_mean, pre_mean, xerr=rec_std, yerr=pre_std,
#                  label=grp + ' (soft)',
#                  color='k', linestyle='-', ecolor='g',
#                  lw=2, alpha=.8)
#     for it in range(3):
#         plt.scatter(rec_mean[it], pre_mean[it], 100, color='k', marker=marker[it], label=marker_labels[it])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Brain PRC with Errobars')
# plt.xlim((0, 1))
# plt.ylim((0, 1))
# plt.legend(loc="lower left")
# plt.show()
#
# # performance differences
# n_cls = 5
# measure = 'auc'
# marker = ('x', 'o', '*')
# marker_labels = ('Optimistic', 'Expected', 'Pessimistic')
# colors = itertools.cycle(('r', 'g', 'b', 'm', 'c'))
# linesty = itertools.cycle(('-', '--'))
# gb = scores[scores['n_cls'] == n_cls].applymap(lambda x: np.array(x[0]) if isinstance(x, list) else x).groupby(('arch',))
# means = {}
# stds = {}
# accs = np.linspace(0.6, 0.9, num=4)
# class_ratio = 1
# grp0 = 'WeightedConvMANN w/ acor'
# # hard values
# vals0 = np.stack(gb.get_group(grp0)[measure])
# plt.figure()
# for grp in gb.groups:
#     # hard values
#     vals = np.stack(gb.get_group(grp)[measure])
#     means[grp] = np.mean(vals - vals0, 0)
#     stds[grp] = np.std(vals - vals0, 0)
#     plt.errorbar(np.random.randn(n_cls) / 100 + range(n_cls), means[grp], stds[grp], label=grp + ' - ' + grp0)
# plt.xlim((-0.1, n_cls - 1 + 0.1))
# plt.xticks(range(n_cls), classes[label_ind[n_cls]], rotation=20)
# plt.ylabel(measure + ' difference')
# plt.title('{}-class comparison of {}'.format(n_cls, measure))
# plt.legend(loc='lower left')
#
# plt.figure()
# for grp in gb.groups:
#     plt.plot(np.minimum(roc_mean[grp][2], 1), np.linalg.norm(np.stack((1 - roc_mean[grp][1], roc_mean[grp][0])).T, axis=1),
#              label=str(grp) + ' ROC',
#              lw=2, alpha=.8)


# train final
labels = 'all'
arch = WeightedConvMANN
for use_autocorr in (False, True):

    # rescale features
    if use_autocorr:
        input_data = [[topo_data[x],
                       0.99 * icl_data[x][0]['psd'],
                       0.99 * icl_data[x][0]['autocorr'],
                       ] for x in range(4)]
    else:
        input_data = [[topo_data[x],
                       0.99 * icl_data[x][0]['psd'],
                       ] for x in range(4)]

    # augment dataset by negating and/or horizontally flipping topo maps
    for it in range(len(input_data)):
        input_data[it][0] = np.concatenate((input_data[it][0],
                                            -input_data[it][0],
                                            np.flip(input_data[it][0], 2),
                                            -np.flip(input_data[it][0], 2)))
        for it2 in range(1, len(input_data[it])):
            input_data[it][it2] = np.tile(input_data[it][it2], (4, 1))
    try:
        train_labels, test_labels = np.tile(icl_data[1][1][0], (4, 1)), np.tile(icl_data[3][1][0], (4, 1))
    except ValueError:
        train_labels, test_labels = (4 * icl_data[1][1], 4 * icl_data[3][1])

    # describe features and name
    additional_features = OrderedDict([('psd_med', input_data[1][1].shape[1])])
    name = 'ICLabel2_' + labels
    if use_autocorr:
        additional_features['autocorr'] = input_data[1][2].shape[1]
        name += '_autocorr'
    name += '_cvFinal10'

    # reset graph
    tf.reset_default_graph()
    if arch is ConvMANN:
        # instantiate model
        model = arch(icl_data[1][1][0].shape[1], additional_features=additional_features,
                     early_stopping=True, name=name)
        # check if already exists, if not train
        if not isdir(join('output', arch.name, arch.name + '_' + name)):
            model.train(input_data[1], train_labels, input_data[3], test_labels,
                        balance_labels=True, learning_rate=3e-4)
    elif arch in WeightedConvMANN:
        # instantiate model
        model = arch(icl_data[1][1][0].shape[1], additional_features=additional_features,
                     early_stopping=True, name=name, weighting=np.array((2, 1, 1, 1, 1, 1, 1)))
        # check if already exists, if not train
        if not isdir(join('output', arch.name, arch.name + '_' + name)):
            model.train(input_data[1], train_labels, input_data[3], test_labels,
                        balance_labels=True, learning_rate=3e-4)
    else:
        # instantiate model
        model = arch(icl_data[1][1][0].shape[1], additional_features=additional_features,
                     mask=mask, early_stopping=True, name=name)
        # check if already exists, if not train
        if not isdir(join('output', arch.name, arch.name + '_' + name)):
            model.train(input_data[0], input_data[1], train_labels, input_data[3], test_labels,
                        balance_labels=True, learning_rate=3e-4, label_strength=0.9, n_epochs=2)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import datetime
import os
from itertools import product

import numpy as np
import scipy.io as scio

# import wpll
from sklearn import metrics
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import pairwise_kernels, linear_kernel, rbf_kernel


def kernelmatrix(ker, X, X2=None, gamma=None):
    # kernel matrix stores several kernel function
    K = None
    if ker == 'rbf':
        K = rbf_kernel(X, X2, gamma)
    elif ker == 'linear':
        K = linear_kernel(X, X2)
    else:
        print("Unsupported kernel {}".format(ker))

    return K


# scan and return the files in 'directory' directory
def scan_files(directory, prefix=None, postfix=None):
    files_list = []

    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(os.path.join(root, special_file))
            elif prefix:
                if special_file.startswith(prefix):
                    files_list.append(os.path.join(root, special_file))
            else:
                files_list.append(os.path.join(root, special_file))

    return files_list


def calculate(func, args):
    return func(*args)


def calculate_star(args):
    return calculate(*args)


def normalize(a, axis=-1):
    """Normalize the last dimension of a

    Parameters
    ----------
    a : np.ndarray

    Returns
    -------
    the normalized array of a

    Example
    -------
    (N, M, K) = a.shape()
    b = normalize(a)
    np.sum(b[i, j, :]) = 1.0
    """
    return np.moveaxis((np.moveaxis(a, axis, 0) / np.sum(a, axis)), 0, axis)


def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)


# def softmax(X):
#     r"""
#     The softmax function for each row of matrix X
#     softmax(X[i])
#
#     Parameters
#     ----------
#     X : numpy.ndarray (m, n)
#
#     Returns
#     -------
#     The softmax matrix of X
#     """
#     X[X > 300.0] = 300.0
#     X[X < -300.0] = -300.0
#
#     exp_X = np.exp(X)
#     return (exp_X.T / np.sum(exp_X, 1)).T


def bootstrap(population, k):
    r"""
    Return a k length list of elements chosen from the population sequence.
    Used for random sampling with replacement.

    Parameters
    ----------
    population : np.ndarray
        Source sequence
    k : int
        number of samples taken from population >0

    Returns
    -------
    the samples array
    """
    samples = []
    for i in range(k):
        samples.append(population[np.random.randint(0, len(population) - 1)])
    return np.asarray(samples)


def normalize_min_max_matrix(X, mode=0):
    r"""
    Min-Max normalization [0, 1] for 2-d array by column

    Parameters
    ---------
    X : numpy.ndarray (M, N)
    mode : 0 or 1
        0 means by row, 1 means by column, default is 0

    Returns
    -------
    Normalize X
    """
    X_min = np.min(X, axis=mode)
    X_max = np.max(X, axis=mode)

    if mode == 1:
        return ((X.T - X_min) / (X_max - X_min)).T
    elif mode == 0:
        return (X - X_min) / (X_max - X_min)


def normalize_z_score(X):
    r"""
    Z-scores normalization
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    return (X - X_mean) / X_std


def normalize_min_max(X):
    r"""
    Min-Max normalization [0, 1]
    """
    X_min = np.min(X)
    X_max = np.max(X)

    return (X - X_min) / (X_max - X_min)


def normalize_min_max1(X):
    r"""
    Min-Max normalization [-1, 1]
    """
    X_min = np.min(X)
    X_max = np.max(X)
    X_mean = np.mean(X)

    return (X - X_mean) / (X_max - X_min)


def normalize_min_max2(X):
    r"""
    Min-Max normalization [-1, 1]
    """
    X_temp = normalize_min_max(X)

    return X_temp**2 - 1.0


def normalize_min_max3(X, a, b):
    r"""
    Min-Max normalization [a, b]
    """
    X_min = np.min(X)
    X_max = np.max(X)
    k = (b - a) / (X_max - X_min)

    return a + k * (X - X_min)


def normalize_tfidf(docs_wordidx, docs_wordnum, word_df):
    r"""
    Return a normalized tfidf weight array similar to docs_wordnum.
    tfidf(t_k, d_i) = tf(t_k, d_i) x log(|D|/|D(t_k)|)
    w_{ki} = tfidf(t_k, d_i)/sqrt(\sum (tfidf(t_j, d_i))^2)

    Parameters
    ----------
    docs_wordidx: np.ndarray
        word index
    docs_wordnum: np.ndarray
        term frequency
    word_df: np.ndarray
        document frequency of word

    Return
    ---------
    docs_tfidf: np.ndarray
        tf_idf
    docs_w: np.ndarray
        normalized tf_idf
    """
    docs_tfidf = []
    docs_w = []
    num_docs = len(docs_wordidx)

    for d in range(num_docs):
        doc_tfidf = docs_wordnum[d] * np.log(
            num_docs / word_df[docs_wordidx[d]])
        docs_tfidf.append(doc_tfidf)

        doc_w = doc_tfidf / np.linalg.norm(doc_tfidf)
        docs_w.append(doc_w)

    docs_tfidf = np.array(docs_tfidf)
    docs_w = np.array(docs_w)

    return docs_tfidf, docs_w


def precision_recall_auc_score(test_target, predict_score, average='samples'):
    (test_ins_num, label_num) = np.shape(test_target)
    pr_auc_instances = []
    for i in range(test_ins_num):
        precision_i, recall_i, _ = metrics.precision_recall_curve(
            test_target[i], predict_score[i])
        pr_auc_instances.append(metrics.auc(precision_i, recall_i))

    return np.mean(pr_auc_instances)


def macro_auc_score(test_target, predict_score):
    (test_ins_num, label_num) = np.shape(test_target)

    Y = test_target.T

    macro_auc = []
    for i in range(label_num):
        label_true = len(Y[i][Y[i] == 1]) + 1
        label_false = len(Y[i][Y[i] == 0]) + 1
        label_true_idx = np.where(Y[i] == 1)[0]
        label_false_idx = np.where(Y[i] == 0)[0]
        # auc_i = [
        #     1 if predict_score[j][i] >= predict_score[k][i] else 0
        #     for (j, k) in product(label_true_idx, label_false_idx)
        # ]
        # macro_auc.append(np.sum(auc_i) / (label_true * label_false))

        # auc_i = 0
        # for (j, k) in product(label_true_idx, label_false_idx):
        #     if predict_score[j][i] >= predict_score[k][i]:
        #         auc_i += 1
        # macro_auc.append(auc_i / (label_true * label_false))

        auc_i = [
            len(
                np.where(
                    predict_score[label_false_idx][:, i] <= predict_score[j][i]
                )[0]) for j in label_true_idx
        ]
        macro_auc.append(np.sum(auc_i) / (label_true * label_false))

    return np.mean(macro_auc)


def micro_auc_score(test_target, predict_score):
    (test_ins_num, label_num) = np.shape(test_target)

    rel_ins_label_set = []
    irrel_ins_label_set = []
    for i in range(test_ins_num):
        rel_ins_label_set += list(predict_score[i][test_target[i] == 1])
        irrel_ins_label_set += list(predict_score[i][test_target[i] == 0])

    rel_ins_label_num = len(rel_ins_label_set)
    irrel_ins_label_num = len(irrel_ins_label_set)

    # auc = [
    #     1 if x >= y else 0
    #     for (x, y) in product(rel_ins_label_set, irrel_ins_label_set)
    # ]
    # micro_auc = np.sum(auc) / (rel_ins_label_num * irrel_ins_label_num)

    # auc = 0
    # for (x, y) in product(rel_ins_label_set, irrel_ins_label_set):
    #     if x >= y:
    #         auc += 1
    # micro_auc = auc / (rel_ins_label_num * irrel_ins_label_num)

    auc = [
        len(np.where(irrel_ins_label_set <= x)[0]) for x in rel_ins_label_set
    ]
    micro_auc = np.sum(auc) / (rel_ins_label_num * irrel_ins_label_num)

    return micro_auc


def one_error(test_target, Outputs):

    (test_ins_num, label_num) = np.shape(test_target)
    max_score = np.max(Outputs, axis=1)
    # max_score_idx = [
    #     np.where(Outputs[i] == max_score[i])[0] for i in range(test_ins_num)
    # ]
    flag_false = [
        0 if
        (test_target[i][np.where(Outputs[i] == max_score[i])[0]]).any() else 1
        for i in range(test_ins_num)
    ]
    oneerr = np.sum(flag_false) / test_ins_num

    return oneerr


def is_error(test_target, Outputs):
    (test_ins_num, label_num) = np.shape(test_target)
    sort_idx = np.argsort(-Outputs, axis=1)
    flag_false = [
        1 if np.where(test_target[i][sort_idx] == 1)[0][-1] > np.where(
            test_target[i][sort_idx] == 0)[0][0] else 0
        for i in range(test_ins_num)
    ]
    iserr = np.sum(flag_false) / test_ins_num

    return iserr


def margin_error(test_target, Outputs):
    (test_ins_num, label_num) = np.shape(test_target)
    sort_idx = np.argsort(-Outputs, axis=1)
    margin_ins = [
        abs(
            np.where(test_target[i][sort_idx] == 1)[0][-1] -
            np.where(test_target[i][sort_idx] == 0)[0][0])
        for i in range(test_ins_num)
    ]
    margin = np.sum(margin_ins) / test_ins_num

    return margin


def ranking_metrics_ml(test_target, predict_score):
    (test_ins_num, label_num) = np.shape(test_target)

    # Coverage
    coverage = (
        metrics.coverage_error(test_target, predict_score) - 1) / label_num
    # coverage = 0

    # Ranking Loss
    rloss = metrics.label_ranking_loss(test_target, predict_score)

    # Average Precision
    ravgprec = metrics.label_ranking_average_precision_score(
        test_target, predict_score)

    # # Margin
    # margin = margin_error(test_target, predict_score)

    # # Is-Error
    # iserr = is_error(test_target, predict_score)

    # # One-Error
    # oneerr = one_error(test_target, predict_score)

    margin = 0
    iserr = 0
    oneerr = 0

    # # The area under the precision-recall curve
    # macro_pr_auc = metrics.average_precision_score(
    #     test_target, predict_score, average='macro')
    # micro_pr_auc = metrics.average_precision_score(
    #     test_target, predict_score, average='micro')
    # exam_pr_auc = metrics.average_precision_score(
    #     test_target, predict_score, average='samples')
    # # weighted_pr_auc = metrics.average_precision_score(
    # #     test_target, predict_score, average='weighted')
    # # exam_pr_auc1 = precision_recall_auc_score(test_target, predict_score)
    macro_pr_auc = 0
    micro_pr_auc = 0
    exam_pr_auc = 0
    exam_pr_auc1 = 0

    # # # ROC AUC
    # # macro_roc_auc = metrics.roc_auc_score(
    # #     test_target, predict_score, average='macro')
    # # micro_roc_auc = metrics.roc_auc_score(
    # #     test_target, predict_score, average='micro')
    # exam_roc_auc = metrics.roc_auc_score(
    #     test_target, predict_score, average='samples')
    # # # weighted_roc_auc = metrics.roc_auc_score(
    # # #     test_target, predict_score, average='weighted')
    exam_roc_auc = 0
    macro_auc = 0
    micro_auc = 0
    # macro_auc = macro_auc_score(test_target, predict_score)
    # micro_auc = micro_auc_score(test_target, predict_score)

    return (coverage, rloss, ravgprec, margin, iserr, oneerr, macro_pr_auc,
            micro_pr_auc, exam_pr_auc, exam_pr_auc1, macro_auc, micro_auc,
            exam_roc_auc)


def binary_prediction_metrics_ml(test_target, predict_label):

    # # Macro Precision, Recall and F_beta
    # macro_prec = metrics.precision_score(
    #     test_target, predict_label, average='macro')
    # macro_recall = metrics.recall_score(
    #     test_target, predict_label, average='macro')
    macro_prec = 0
    macro_recall = 0
    macro_f1 = metrics.f1_score(test_target, predict_label, average='macro')

    # # Micro Precision, Recall and F_beta
    # micro_prec = metrics.precision_score(
    #     test_target, predict_label, average='micro')
    # micro_recall = metrics.recall_score(
    #     test_target, predict_label, average='micro')
    micro_prec = 0
    micro_recall = 0
    micro_f1 = metrics.f1_score(test_target, predict_label, average='micro')

    # # Example Precision, Recall and F_beta
    # exam_prec = metrics.precision_score(
    #     test_target, predict_label, average='samples')
    # exam_recall = metrics.recall_score(
    #     test_target, predict_label, average='samples')
    # exam_f1 = metrics.f1_score(test_target, predict_label, average='samples')
    # exam_f1_1 = 2 * exam_prec * exam_recall / (exam_prec + exam_recall)

    exam_prec = 0
    exam_recall = 0
    exam_f1 = 0
    exam_f1_1 = 0

    # # Weighted Precision, Recall and F_beta
    # weighted_prec = metrics.precision_score(
    #     test_target, predict_label, average='weighted')
    # weighted_recall = metrics.recall_score(
    #     test_target, predict_label, average='weighted')
    # weighted_f1 = metrics.f1_score(
    #     test_target, predict_label, average='weighted')

    # Hamming Loss
    hamming_loss = metrics.hamming_loss(test_target, predict_label)

    # # Subset Accuracy
    # subset_accuracy = metrics.accuracy_score(test_target, predict_label)

    # # Example Accuracy
    # jaccard_similarity = metrics.jaccard_similarity_score(
    #     test_target, predict_label)

    # # Others
    # log_loss = metrics.log_loss(test_target, predict_label)
    # zero_one_loss = metrics.zero_one_loss(test_target, predict_label)

    # hamming_loss = 0
    subset_accuracy = 0
    jaccard_similarity = 0
    log_loss = 0
    zero_one_loss = 0

    return (macro_prec, macro_recall, macro_f1, micro_prec, micro_recall,
            micro_f1, exam_prec, exam_recall, exam_f1, exam_f1_1, hamming_loss,
            subset_accuracy, jaccard_similarity, log_loss, zero_one_loss)


def metrics_ml(Outputs, test_target, avg_num_rel=-1):

    (test_ins_num, label_num) = np.shape(test_target)

    predict_label_idxes = []
    if avg_num_rel != -1:
        predict_label_idxes = np.argsort(-1.0 * Outputs)[:, :avg_num_rel]
    else:
        predict_label_idxes = [
            np.argsort(-1.0 * Outputs[i])[:np.sum(test_target[i])]
            for i in range(test_ins_num)
        ]

    predict_label = np.zeros(np.shape(test_target))
    for i in range(test_ins_num):
        predict_label[i][predict_label_idxes[i]] = 1

    ranking_metrics = ranking_metrics_ml(test_target, Outputs)
    binary_prediction_metrics = binary_prediction_metrics_ml(
        test_target, predict_label)

    return ranking_metrics, binary_prediction_metrics


def metrics_sl(Outputs, test_target):

    (test_ins_num, _) = np.shape(test_target)
    maxidx = np.argmax(Outputs, axis=1)

    # true_count = 0
    # for i in range(test_ins_num):
    #     if test_target[i][maxidx[i]] == 1:
    #         true_count += 1
    # accuarcy = true_count / test_ins_num

    predict_label = np.zeros(np.shape(test_target))
    for i in range(test_ins_num):
        predict_label[i][maxidx[i]] = 1

    macro_f1 = f1_score(test_target, predict_label, average='macro')
    micro_f1 = f1_score(test_target, predict_label, average='micro')

    return (macro_f1, micro_f1)


def generate_train_val(num_ins, split, cv_folds):
    """split dataset as train set and validation set"""
    # test_num = round(num_ins / cv_folds)
    # start_ind = split * test_num
    # if split == cv_folds:
    #     end_ind = num_ins
    # else:
    #     end_ind = start_ind + test_num

    # index_te = np.zeros(num_ins, dtype=bool)
    # index_te[start_ind:end_ind] = True
    # index_tr = ~index_te

    index = np.arange(num_ins)
    index_te = (index % cv_folds == split)
    index_tr = ~index_te

    return index_tr, index_te


if __name__ == '__main__':
    # Partial Multi-Label Learning
    datasets_dir = './datasets/'
    results_dir = './results/'
    datasets = [
        # 'Image',
        # 'emotions',
        'Yeast'
    ]

    # Partial Multi-Label Learning
    datasets_dir = './datasets/PML/'
    results_dir = './results/PML/'
    datasets = [
        # 'Image',
        # 'emotions',
        'Yeast'
    ]

    for dataset in datasets:
        if dataset == 'Image':
            avg_CLs = [2, 3, 4]
        elif dataset == 'emotions' or dataset == 'scene':
            avg_CLs = [3, 4, 5]
        elif dataset == 'Yeast':
            avg_CLs = [9, 10, 11, 12, 13]
        elif dataset == 'eurlex_dc' or dataset == 'eurlex_sm':
            avg_CLs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        for avg_CL in avg_CLs:
            dataset_filename = dataset + '_' + str(avg_CL)
            print("dataset : {}".format(dataset_filename))

            # ---------------------------------------------------------------------------------
            (coverage_data, rloss_data, ravgprec_data, pr_auc_data,
             roc_auc_data, macro_f1_data, micro_f1_data,
             hamming_loss_data) = np.load(results_dir + dataset_filename +
                                          "_" + str(LAMBDA_REG) + "_" +
                                          str(BETA_REG) + "_WPLL_PML.npy")

            coverage_mean = np.mean(coverage_data, axis=0)
            rloss_mean = np.mean(rloss_data, axis=0)
            ravgprec_mean = np.mean(ravgprec_data, axis=0)
            pr_auc_mean = np.mean(pr_auc_data, axis=0)
            roc_auc_mean = np.mean(roc_auc_data, axis=0)
            macro_f1_mean = np.mean(macro_f1_data, axis=0)
            micro_f1_mean = np.mean(micro_f1_data, axis=0)
            hamming_loss_mean = np.mean(hamming_loss_data, axis=0)

            coverage_std = np.std(coverage_data, axis=0)
            rloss_std = np.std(rloss_data, axis=0)
            ravgprec_std = np.std(ravgprec_data, axis=0)
            pr_auc_std = np.std(pr_auc_data, axis=0)
            roc_auc_std = np.std(roc_auc_data, axis=0)
            macro_f1_std = np.std(macro_f1_data, axis=0)
            micro_f1_std = np.std(micro_f1_data, axis=0)
            hamming_loss_std = np.std(hamming_loss_data, axis=0)

            print(
                "Coverage Error: {} {}\tRanking Loss: {} {}\tRanking-based Average Precision: {} {}\nAUC_PR: {} {}\tAUC_ROC: {} {}"
                .format(coverage_mean[-1], coverage_std[-1], rloss_mean[-1],
                        rloss_std[-1], ravgprec_mean[-1], ravgprec_std[-1],
                        pr_auc_mean[-1], pr_auc_std[-1], roc_auc_mean[-1],
                        roc_auc_std[-1]))
            print(
                "Macro F1: {} {}\tMicro F1: {} {}\tHamming Loss: {} {}".format(
                    macro_f1_mean[-1], macro_f1_std[-1], micro_f1_mean[-1],
                    micro_f1_std[-1], hamming_loss_mean[-1],
                    hamming_loss_std[-1]))
            # -------------------------------------------------------------------------------------

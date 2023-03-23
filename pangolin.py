#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import datetime
import os
import pickle
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import scipy.io as scio
from numpy import linalg as LA
from scipy.linalg import cholesky, solve
from scipy.spatial.distance import cdist

from cvxopt import matrix, solvers
from load_data import loaddata
from updates import (adadelta, adagrad, adam, adamax, amsgrad, nadam, rmsprop,
                     sgd)
from utils import (bootstrap, calculate_star, generate_train_val, kernelmatrix,
                   metrics_sl, normalize, scan_files, softmax)

PROCESSES = 30

LAMBDA_REGs = [
    0,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]
BETA_REGs = [
    0,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]
GAMMA_REGs = [
    0,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]


def mylinsolve(A, b):
    r"""
    Solve Ax=b (A and b are both dense) by cholesky factorization
    """
    L = cholesky(A, lower=True)
    y = solve(L, b)
    x = solve(L.T, y)

    return x


def qp_ops(y, q, d):
    r"""
    QP procedure to solve
    .. math ::
    OPS = \min_{p}||p-q||_2^2 + d^T p
    s.t. \mathbf{1}^T p = 1
         \mathbf{0} \preceq p \preceq y

    The standard form of OPS is
    .. math ::
    OPS = \min_{p}p^T\mathbb{1}p + (-2) * q^T p + d^T p
    s.t. \mathbf{1}^T p = 1
         -p \preceq \mathbf{0}
         p \preceq y

    Parameters
    ----------
    y : numpy.array (l,)
    q : numpy.array (l,)
    beta_reg : float

    Returns
    -------
    p : numpy.array (l,)
    """
    n = np.size(q)

    # -----------------------------------------------------
    # 1/2 p^T * I * p - q^T * p + 1/2 * d^T p
    # -----------------------------------------------------
    I = matrix(0.0, (n, n))
    I[::n + 1] = 1.0
    q = matrix(q - 0.5 * d)

    # ---------------------------------------------------------
    # 1. -p \preceq \mathbf{0}
    # 3. p \preceq y
    # ---------------------------------------------------------
    G = matrix([-I, I])
    h = matrix(n * [0.0] + y.tolist())

    # \mathbf{1}^T * p =1
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    # Quadratic Programming
    solvers.options['show_progress'] = False
    # solvers.options['maxiters'] = 10
    p = solvers.qp(I, -q, G, h, A, b)['x']
    p_array = np.array(p)

    return np.reshape(p_array, (np.shape(p_array)[0], ))


def qp_ops_batch(Y, H, Q):
    (batch, _) = np.shape(Y)

    P = np.zeros(np.shape(Y))
    # begin = datetime.datetime.now()
    for i in range(batch):
        p = qp_ops(Y[i], H[i], Q[i])
        P[i] += p
    # end = datetime.datetime.now()
    # print(end - begin)
    return P


def calculate_P(Y, H, Q):
    (ins_num, _) = np.shape(Y)

    batch = 1000
    bat_iter = ins_num // batch
    bat_iter = bat_iter if ins_num % batch == 0 else bat_iter + 1

    TASKS = []
    for i in range(bat_iter):
        head_idx = i * batch
        tail_idx = i * batch + batch
        tail_idx = tail_idx if tail_idx < ins_num else ins_num

        TASKS.append((qp_ops_batch, (Y[head_idx:tail_idx], H[head_idx:tail_idx],
            Q[head_idx:tail_idx])))

    processes = bat_iter if bat_iter < PROCESSES else PROCESSES
    with Pool(processes) as pool:
        results = pool.map(calculate_star, TASKS)

    P = results[0]

    for i in range(bat_iter - 1):
        P = np.concatenate((P, results[i + 1]), axis=0)

    return P


def calculate_Q(p, q, kq, X, sigma):
    epsilon = 1e-10
    (N, D) = np.shape(X)
    g = 1 / (sigma**2) * np.sum(p * kq * (X - q).T, axis=1)

    H = np.zeros((D, D))
    for i in range(N):
        H = H + p[i]*kq[i]*np.outer(X[i]-q, X[i]-q)

    H = 1 / (sigma**4) * H
    H = H - 1 / (sigma**2) * np.sum(p * kq) * np.eye(D)

    # Avoid that the Hessian matrix is a singular matrix
    H = H + np.eye(D) * epsilon

    return np.matmul(LA.inv(H), np.matmul(H, q) - g)


def pangolin(X,
             Y,
             Y0,
             X_val,
             Y_val,
             Y0_val,
             max_iter,
             Test_Freq=1,
             split=0,
             LAMBDA_REG=0.5,
             BETA_REG=0.05,
             GAMMA_REG=0.0001):
    r"""
    Parameters
    ----------
    X : numpy.ndarray (m, n)
        train data with m instances, and each instance has n features
    Y : numpy.ndarray (m, l)
        partial label matrix with l labels for m instances in `X`,
        (i, j) is `1` means the i-th instance is assigned the j-th label
    Y0 : numpy.ndarray (m, l)
        ground-truth labels of m instances in `test_data`,
        (i, j) is `1` means the ground-truth label of the i-th instance
        is the j-th label, only one label is `1` for each instance
    max_iter : float
        number of iterations
    Test_Freq : float
        frequency of testing
    Returns
    -------
    A : numpy.ndarray (n, l)
        model parameter `A`
    b : numpy.ndarray (l,)
        model parameter `b`
    """

    accuracy_all = []
    loss_all = []

    (ins_num, feature_num) = np.shape(X)
    (_, label_num) = np.shape(Y)

    epsilon = 1e-10
    # Kernel
    ker = 'rbf'
    D = cdist(X, X)
    sigma = np.mean(D)
    K = kernelmatrix(ker, X, gamma=1.0 / (2 * sigma**2))
    Kt = kernelmatrix(ker, X_val, X, gamma=1.0 / (2 * sigma**2))

    # Indicating matrix
    label_sim = 1 - cdist(Y, Y, 'cosine')
    IMat = np.zeros((ins_num, ins_num))
    IMat[label_sim == 0] = 1

    # Laplacian matrix
    LMat = np.diag(np.sum(IMat,1)) - IMat

    one_mat = np.ones((ins_num, ins_num))

    # Model parameters
    A = np.ones((ins_num, label_num))
    Q = np.array([np.mean(X[Y[:, l] == 1], axis=0) for l in range(label_num)])
    # the confidence matrix, intialized using partial_target_tr
    # (ins_num, label_num)
    P = (Y.T / np.sum(Y, axis=1)).T

    lambda_reg = LAMBDA_REG  # parameter regularization term of model for `P`
    beta_reg = BETA_REG  # parameter regularization term of model for `W`
    gamma_reg = GAMMA_REG  # parameter regularization term of model for local `P`

    for iter in range(max_iter):
        # .. math :: KA
        Z = np.matmul(K, A)

        print("Calculating loss...")

        # The loss function of SURE
        loss = 0.0
        loss_L2 = np.sum(np.power(Z - P, 2))

        Q[np.isnan(Q)] = 0.0
        KQ = kernelmatrix(ker, X, Q, gamma=1.0 / (2 * sigma**2))
        feature_prototype = lambda_reg * 2.0 * np.sum(P * (1 - KQ))
        regularizer_W = beta_reg * np.trace(np.matmul(np.matmul(A.T, K), A))
        local_P = - gamma_reg * np.trace(np.matmul(np.matmul(P.T, LMat), P))

        loss = loss_L2 + feature_prototype + regularizer_W + local_P
        loss_all.append(loss)

        print(
                "Loss: {}\tloss_L2: {}\tfeature_prototype: {}\tregularizer_W: {}\tlocal_P: {}".
            format(loss, loss_L2, feature_prototype, regularizer_W, local_P))

        # Testing process
        if iter % Test_Freq == 0:
            print("Testing...")
            test_accuracy = predict_kernel(A, Kt, Y0_val)
            train_accuracy = predict_kernel(A, K, Y0)
            print("Iteration {}/{}/{}/{}/{}: {}".format(iter, split, lambda_reg,
                                                     beta_reg, gamma_reg, loss))
            # print("Iteration {}/{}: {}".format(iter, split, loss))
            print("Train set: Macro F1 {}\tMicro F1 {}".format(
                train_accuracy[0], train_accuracy[1]))
            print("Test set: Macro F1 {}\tMicro F1 {}".format(
                test_accuracy[0], test_accuracy[1]))
        else:
            # print("Iteration {}/{}: {}".format(iter, split, loss))
            print("Iteration {}/{}/{}/{}/{}: {}".format(iter, split, lambda_reg,
                                                     beta_reg, gamma_reg, loss))

        begin = datetime.datetime.now()
        # # Update parameters `A`, `Q` and `P`
        A = mylinsolve(K + (beta_reg + epsilon) * np.eye(ins_num), P)
        # middle1 = datetime.datetime.now()
        # print(middle1 - begin)

        H = np.matmul(K, A)
        LP = np.matmul(LMat, P)

        CMat = 2.0 * lambda_reg * (1 - KQ) - 2.0 * gamma_reg * LP
        P = calculate_P(Y, H, CMat)
        middle = datetime.datetime.now()
        print(middle - begin)

        if lambda_reg != 0:
            with Pool(PROCESSES) as pool:
                TASKS = [(calculate_Q, (P[:, l], Q[l], KQ[:, l], X, sigma))
                         for l in range(label_num)]
                Q = np.array(pool.map(calculate_star, TASKS))

            end = datetime.datetime.now()
            print(end - middle)

        accuracy_all.append(test_accuracy)

    return accuracy_all, loss_all, (A, Q, P)


def train(data,
          partial_target,
          target,
          results_dir,
          dataset,
          method='kernel',
          LAMBDA_REG=0.5,
          BETA_REG=0.05,
          GAMMA_REG=0.0001):
    r"""
    Parameters
    ----------
    data : numpy.ndarray (m, n)
        data with m instances, and each instance has n features
    partial_target : numpy.ndarray (l, m)
        partial label matrix with l labels for m instances in `data`,
        (j, i) is `1` means the i-th instance is assigned the j-th label
    target : numpy.ndarray (l, m)
        ground-truth labels of m instances in `data`,
        (j, i) is `1` means the ground-truth label of the i-th instance
        is the j-th label, only one label is `1` for each instance
    method : `kernel` or `linear`
        use kernel method or linear method
    """

    # get the the number of instances, features and label, respectively
    (ins_num, feature_num) = np.shape(data)
    (label_num, _) = np.shape(partial_target)

    partial_target_T = partial_target.T
    target_T = target.T

    # Optimization parameters
    max_iter = 50  # number of iterations
    save_frequency = max_iter  # frequency of saving results
    Test_Freq = 1  # frequency of testing

    model_parameters_data = []
    accuracy_data = []
    loss_data = []

    cv_folds = 10  # number of folds for cross-validation
    for split in range(cv_folds):
        save_counter = 0

        # split dataset into `cv_folds`, one as validation and others as train
        idx_tr, idx_val = generate_train_val(ins_num, split, cv_folds)

        # validation dataset
        data_val = data[idx_val]
        partial_target_val_T = partial_target_T[idx_val]
        target_val_T = target_T[idx_val]
        X_val = data_val
        Y_val = partial_target_val_T
        Y0_val = target_val_T

        # training dataset
        data_tr = data[idx_tr]
        partial_target_tr_T = partial_target_T[idx_tr]
        target_tr_T = target_T[idx_tr]
        X = data_tr
        Y = partial_target_tr_T
        Y0 = target_tr_T

        (accuracy_split, loss_split,
         model_parameters_split) = pangolin(X,
                                            Y,
                                            Y0,
                                            X_val,
                                            Y_val,
                                            Y0_val,
                                            max_iter,
                                            split=split,
                                            LAMBDA_REG=LAMBDA_REG,
                                            BETA_REG=BETA_REG,
                                            GAMMA_REG=GAMMA_REG)

        np.save(
            results_dir + dataset + "_" + str(LAMBDA_REG) + "_" +
            str(BETA_REG) + "_" + str(GAMMA_REG) + "_PANGOLIN_kernel_" + str(split) + ".npy",
            (accuracy_split))
        np.save(
            results_dir + dataset + "_" + str(LAMBDA_REG) + "_" +
            str(BETA_REG) + "_" + str(GAMMA_REG) + "_PANGOLIN_kernel_loss_" + str(split) + ".npy",
            (loss_split))
        np.save(
            results_dir + dataset + "_" + str(LAMBDA_REG) + "_" +
            str(BETA_REG) + "_" + str(GAMMA_REG) + "_PANGOLIN_kernel_para_" + str(split) + ".npy",
            (model_parameters_split))

        accuracy_data.append(accuracy_split)
        loss_data.append(loss_split)
        model_parameters_data.append(model_parameters_split)

    return np.array(accuracy_data), np.array(loss_data), model_parameters_data


def predict_kernel(A, K, Y0_val):
    r"""
    Parameters
    ----------
    A : numpy.ndarray (m, l)
        model parameter `A` in wpll
    Y0_val : numpy.ndarray (m, l)
        ground-truth labels of m instances in `test_data`,
        (i, j) is `1` means the ground-truth label of the i-th instance
        is the j-th label, only one label is `1` for each instance

    Returns
    -------
    accuarcy : float
        Predictive accuarcy on the test set
    """
    Z_val = np.matmul(K, A)
    return metrics_sl(Z_val, Y0_val)


def main(datasets_dir, results_dir, dataset):
    data, partial_target, target = loaddata(datasets_dir + dataset +
                                                    '.mat')
    for LAMBDA_REG in LAMBDA_REGs:
        for BETA_REG in BETA_REGs:
            for GAMMA_REG in GAMMA_REGs:
                accuracy_data, loss_data, model_parameters_data = train(
                    data,
                    partial_target,
                    target,
                    LAMBDA_REG=LAMBDA_REG,
                    BETA_REG=BETA_REG,
                    GAMMA_REG=GAMMA_REG)
                np.save(
                    results_dir + dataset + "_" + str(LAMBDA_REG) + "_" +
                    str(BETA_REG) + "_" + str(GAMMA_REG) + "_PANGOLIN_kernel.npy",
                    (accuracy_data))
                np.save(
                    results_dir + dataset + "_" + str(LAMBDA_REG) + "_" +
                    str(BETA_REG) + "_" + str(GAMMA_REG) + "_PANGOLIN_kernel_loss.npy",
                    (loss_data))
                np.save(
                    results_dir + dataset + "_" + str(LAMBDA_REG) + "_" +
                    str(BETA_REG) + "_" + str(GAMMA_REG) + "_PANGOLIN_kernel_para.npy",
                    (model_parameters_data))


if __name__ == "__main__":

    datasets_dir = 'datasets/'
    results_dir = 'results/'
    datasets = [
        'lost_Z',
        'MSRCv2_Z',
        'BirdSong_Z',
        'SoccerPlayer_Z',
        'YahooNews_Z',
        'FG-NET_Z',
    ]

    for dataset in datasets:
        print(dataset)
        data, partial_target, target = loaddata(datasets_dir + dataset +
                                                '.mat')
        for LAMBDA_REG in LAMBDA_REGs:
            for BETA_REG in BETA_REGs:
                for GAMMA_REG in GAMMA_REGs:
                    accuracy_data, loss_data, model_parameters_data = train(
                        data,
                        partial_target,
                        target,
                        results_dir,
                        dataset,
                        LAMBDA_REG=LAMBDA_REG,
                        BETA_REG=BETA_REG,
                        GAMMA_REG=GAMMA_REG)
                    np.save(
                        results_dir + dataset + "_" + str(LAMBDA_REG) + "_" +
                        str(BETA_REG) + "_" + str(GAMMA_REG) + "_PANGOLIN_kernel.npy",
                        (accuracy_data))
                    np.save(
                        results_dir + dataset + "_" + str(LAMBDA_REG) + "_" +
                        str(BETA_REG) + "_" + str(GAMMA_REG) + "_PANGOLIN_kernel_loss.npy",
                        (loss_data))
                    np.save(
                        results_dir + dataset + "_" + str(LAMBDA_REG) + "_" +
                        str(BETA_REG) + "_" + str(GAMMA_REG) + "_PANGOLIN_kernel_para.npy",
                        (model_parameters_data))

    # datasets_dir = 'datasets/controlled_UCI/'
    # results_dir = 'results/controlled_UCI/'
    # datasets = [
    #         'abalone_Z',
    #         'pendigits_Z',
    #         'satimage_Z',
    #         'segment_Z',
    #         'usps_Z',
    #         'vehicle_Z'
    #     ]

    # Ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # Rs = [1, 2, 3]
    # Epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # for dataset in datasets:
    #     for epsilon in Epsilons:
    #         dataset_filename = dataset + '_' + str(1) + '_' + str(1) + '_' + str(epsilon)
    #         print(dataset_filename)
    #         main(datasets_dir, results_dir, dataset_filename)

    #     for r in Rs:
    #         for p in Ps:
    #             dataset_filename = dataset + '_' + str(p) + '_' + str(r)
    #             print(dataset_filename)
    #             main(datasets_dir, results_dir, dataset_filename)

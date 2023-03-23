#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import random

import numpy as np
import scipy.io as scio
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

from utils import normalize_min_max_matrix, normalize_z_score

# def generate_train_val(num_ins, split, cv_folds):
#     """split dataset as train set and validation set"""
#     index = np.arange(num_ins)
#     index_te = (index % cv_folds == split)
#     index_tr = ~index_te
#     return index_tr, index_te


def loaddata(filename_dataset):
    dataset = scio.loadmat(filename_dataset)
    data = dataset['data']
    partial_target = dataset['partial_target']
    target = dataset['target']

    return data, partial_target, target


def loaddata_multi_label(filename_dataset):
    dataset = scio.loadmat(filename_dataset)
    data = dataset['data']
    target = dataset['target']

    target[target == -1] = 0

    return data, target


def loaddata_partial_multi_label_train_test(filename_dataset):
    dataset = scio.loadmat(filename_dataset)
    trainX = dataset['trainX']
    trainY = dataset['trainY']
    trainY0 = dataset['trainY0']
    testX = dataset['testX']
    testY = dataset['testY']
    testY0 = dataset['testY0']

    trainY[trainY == -1] = 0
    trainY0[trainY0 == -1] = 0
    testY[testY == -1] = 0
    testY0[testY0 == -1] = 0

    return trainX, trainY, trainY0, testX, testY, testY0


def loaddata_partial_multi_label(filename_dataset):
    dataset = scio.loadmat(filename_dataset)
    data = dataset['data']
    partial_target = dataset['partial_labels']
    target = dataset['target']

    partial_target[partial_target == -1] = 0
    target[target == -1] = 0

    return data, partial_target, target




def process_Yeast():
    filename = './dataset/PML/Original/Yeast.mat'
    dataset = scio.loadmat(filename)
    Xapp = dataset['Xapp']
    Xgen = dataset['Xgen']
    Yapp = dataset['Yapp']
    Ygen = dataset['Ygen']

    print(np.shape(Xapp))
    print(np.shape(Xgen))
    print(np.shape(Yapp))
    print(np.shape(Ygen))

    data = np.concatenate((Xapp, Xgen), axis=0)
    target = np.concatenate((Yapp, Ygen), axis=0)
    target = target.T
    target[target == -1] = 0

    scio.savemat('./dataset/PML/Yeast.mat', {'data': data, 'target': target})


def process_emotions_image():
    filename = './dataset/PML/Image.mat'
    dataset = scio.loadmat(filename)

    data = dataset['data']
    target = dataset['target']
    target[target == -1] = 0

    scio.savemat('./dataset/PML/Image.mat', {'data': data, 'target': target})


def get_multi_label_dataset_statistics(filename_dataset):
    data, target = loaddata_multi_label(filename_dataset)
    target_T = target.T
    (num_ins, num_feature) = np.shape(data)
    (num_label, _) = np.shape(target)
    lcard = np.sum(target) / num_ins
    lden = lcard / num_label
    dl = len(set([''.join(map(lambda x: str(x), y)) for y in target_T]))
    pdl = dl / num_ins

    print(np.sum(target, axis=1))
    print(
        "Number of instances: {}\tNumber of features: {}\tNumber of possible class labels: {}"
        .format(num_ins, num_feature, num_label))
    print(
        "Label cardinality: {}\tLable density: {}\tDistinct label sets: {}\tProportion of distinct label set: {}"
        .format(lcard, lden, dl, pdl))


def supervised_select_feature(dataset):
    """select features square_chi_max score with respect to the category set
    following ``RCV1: A New Benchmark Collection for Text Categorization Research``
    """
    dataset_dir = "./"

    #------------------------------------------------------------------------------------------
    # clean data
    data, target = loaddata_multi_label(dataset_dir + dataset + ".mat")
    data_T = data.T
    data_T_save = np.array([feature for feature in data_T if len(feature[feature != 0]) != 0])

    print(np.shape(data_T_save))

    scio.savemat('./' + dataset + '_clean' + '.mat', {
        'data': data_T_save.T,
        'target': target
    })
    #-----------------------------------------------------------------------------------------

    # #-----------------------------------------------------------------------------------------
    # # calculate square_chi_score for each feature with respect to all categories
    # data, target = loaddata_multi_label(dataset_dir + dataset + '_clean.mat')

    # target[target == 1] = 2
    # feature = data.T
    # feature[feature != 0] = 3
    # (N, num_feature) = np.shape(data)
    # (num_label, _) = np.shape(target)

    # square_chi_score = np.zeros((num_feature, num_label))

    # for i in range(num_feature):
    #     feature_i = feature[i] * np.ones((num_label, N))
    #     coocurrence = feature_i - target
    #     A = np.array([len(temp[temp == 1]) for temp in coocurrence])
    #     B = np.array([len(temp[temp == 3]) for temp in coocurrence])
    #     C = np.array([len(temp[temp == -2]) for temp in coocurrence])
    #     D = np.array([len(temp[temp == 0]) for temp in coocurrence])
    #     square_chi_score[i] += ((N * ((A * D - B * C) ** 2)) / (
    #         (A + B) * (A + C) * (B + D) * (C + D)))

    # np.save(dataset_dir + dataset + "_square_chi_score.npy",
    #         (square_chi_score))
    # #---------------------------------------------------------------------------------------

    # #---------------------------------------------------------------------------------------
    # # select features with maximum of square_chi_score over all categories
    # K = 500
    # (square_chi_score) = np.load(dataset_dir + dataset +
    #                              "_square_chi_score.npy")
    # data, target = loaddata_multi_label(dataset_dir + dataset + '_clean.mat')
    # max_square_chi_score = np.nanmax(square_chi_score, axis=1)
    # max_idx = np.argsort(-max_square_chi_score)[:K]
    # print(np.sum([len(temp[temp != 0]) for temp in data]))
    # data_save = data[:, max_idx]
    # print(np.sum([len(temp[temp != 0]) for temp in data_save]))
    # scio.savemat('./' + dataset + '_' + str(K) + '.mat', {
    #     'data': data[:, max_idx],
    #     'target': target
    # })
    # #-------------------------------------------------------------------------------------


def add_random_labeling_noise(filename):

    dataset_dir = './datasets/PML/'

    dataset = scio.loadmat(dataset_dir + filename + '.mat')

    target = dataset['target']

    (class_num, ins_num) = np.shape(target)

    # # the minimum number of ground-truth labels in single instance
    # multi_label_num = np.sum(target, axis=0)
    # max_label_num = np.max(multi_label_num)

    partial_labels = np.ones((ins_num, class_num)) * target.T

    avg_CLs = []

    if filename == 'Image':
        avg_CLs = [2, 3, 4]
    elif filename == 'emotions' or filename == 'scene':
        avg_CLs = [3, 4, 5]
    elif filename == 'Yeast':
        avg_CLs = [9, 10, 11, 12, 13]
    elif filename == 'eurlex_dc' or filename == 'eurlex_sm':
        avg_CLs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    for avg_CL in avg_CLs:
        for i in range(ins_num):
            label_num_ins = np.sum(partial_labels[i])
            irrelevant_label_index = [
                int(i) for i, v in enumerate(partial_labels[i]) if v == 0
            ]
            if avg_CL > label_num_ins:
                noise_label_index = random.sample(irrelevant_label_index,
                                                  int(avg_CL - label_num_ins))
                partial_labels[i][noise_label_index] = 1

        scio.savemat(
            './datasets/' + filename + '_' + str(avg_CL) + '.mat', {
                'data': dataset['data'],
                'partial_labels': partial_labels.T,
                'target': target
            })


def split_PML_dataset(filename, s_folds=2):
    datasets_dir = './datasets/PML/'

    avg_CLs = [100, 200, 300]

    for avg_CL in avg_CLs:
        dataset_filename = filename + '_' + str(avg_CL)

        data, partial_target, target = loaddata_partial_multi_label(
            datasets_dir + dataset_filename + '.mat')

        (class_num, ins_num) = np.shape(target)

        index = np.arange(ins_num)
        for i in range(s_folds):
            index_te = (index % s_folds == i)

            scio.savemat(
                    datasets_dir + filename + '_' + str(i) + '_' + str(avg_CL) + '.mat', {
                        'data': data[index_te],
                        'partial_labels': partial_target[:, index_te],
                        'target': target[:, index_te]
                    })


def split_ML_dataset(filename, s_folds=2):
    datasets_dir = './datasets/'

    data, target = loaddata_multi_label(
        datasets_dir + filename + '.mat')

    (class_num, ins_num) = np.shape(target)

    index = np.arange(ins_num)
    for i in range(s_folds):
        index_te = (index % s_folds == i)

        # print(np.sum(np.sum(data[index_te], axis=0) > 5))
        # print(np.sum(np.sum(target[:, index_te], axis=1) > 10))
        save_label_flag = (np.sum(target[:, index_te], axis=1) > 10)

        scio.savemat(
                datasets_dir + filename + '_' + str(i) + '.mat', {
                    'data': data[index_te],
                    'target': (target[:, index_te])[save_label_flag]
                })


def generate_save_train_test_split(dataset_dir, dataset, cv_folds=5, test_size=0.25):
    data, partial_target, target = loaddata_partial_multi_label(dataset_dir + dataset + '.mat')
    for split in range(cv_folds):
        (data_train, data_test, partial_target_train, partial_target_test,
                target_train, target_test) = train_test_split(data, partial_target.T, target.T, test_size=test_size)
        scio.savemat(
            dataset_dir + "test_size/" + dataset + '_' + str(test_size * 100) + '_' + str(split) + '.mat', {
                'trainX': data_train,
                'trainY': partial_target_train.T,
                'trainY0': target_train.T,
                'testX': data_test,
                'testY': partial_target_test.T,
                'testY0': target_test.T,
            })


def generate_save_train_test_split_PML(dataset_dir, dataset, cv_folds=5, test_size=0.25):
    # avg_CLs = [100, 200, 300]
    avg_CLs = [400]
    for avg_CL in avg_CLs:
        dataset_name = dataset + "_" + str(avg_CL)
        generate_save_train_test_split(dataset_dir, dataset_name, cv_folds, test_size)


if __name__ == '__main__':
    # dataset_dir = "../../dataset/Processed/"
    # datasets = [
    #     'BirdSong', 'FG-NET', 'lost', 'MSRCv2', 'SoccerPlayer', 'YahooNews'
    # ]
    # for dataset in datasets:
    #     data, partial_target, target = loaddata(dataset_dir + dataset + '.mat')
    #     print('data:')
    #     print(np.shape(data))
    #     print('partial_target:')
    #     print(np.shape(partial_target))
    #     print('target:')
    #     print(np.shape(target))

    # process_emotions_image()
    # process_Yeast()

    # dataset_dir = "./dataset/PML/"
    # datasets = ['Image', 'emotions', 'Yeast']
    # for dataset in datasets:
    #     add_random_labeling_noise(dataset)

    # datasets_dir = './datasets/PML/'
    # datasets = [
    #     'Image', 'emotions', 'mirflickr', 'music_emotion', 'music_style',
    #     'Yeast'
    # ]

    # K_save = 50
    # for dataset_name in datasets:
    #     dataset = scio.loadmat(datasets_dir + dataset_name + '.mat')
    #     X = dataset['data']
    #     (ins_num, _) = np.shape(X)

    #     # Get the K-nearest neighborhoods of each instance in `X`
    #     dists = cdist(X, X, metric='euclidean')
    #     neighbors = np.argsort(dists)[:, 1:(K_save + 1)]
    #     dists_neighbors = np.array(
    #         [dists[d, neighbors[d]] for d in range(ins_num)])
    #     np.save(
    #         datasets_dir + dataset_name + '_neighbors_euclidean_' + str(K_save)
    #         + '.npy', (neighbors, dists_neighbors))

    # dataset_dir = './datasets/ML/'
    # datasets = [
    #     # # 'birds',
    #     # # 'music',
    #     'CAL500',
    #     'emotions',
    #     'genbase',
    #     'medical',
    #     # # 'MEDICAL-F',
    #     # # 'llog',
    #     # # 'Image',
    #     # # 'scene',
    #     'enron',
    #     # # 'ENRON-F',
    #     'Yeast',
    #     # # 'slashdot',
    #     'corel5k',
    #     'rcv1subset1',
    #     'rcv1subset2',
    #     'rcv1subset3',
    #     'rcv1subset4',
    #     'rcv1subset5',
    #     'bibtex',
    #     'Corel16k001',
    #     'Corel16k002',
    #     'Corel16k003',
    #     'Corel16k004',
    #     'Corel16k005',
    #     'Corel16k006',
    #     'Corel16k007',
    #     'Corel16k008',
    #     'Corel16k009',
    #     'Corel16k010',
    #     'OHSUMED-F',
    # ]

    # for dataset in datasets:
    #     print(dataset)
    #     get_multi_label_dataset_statistics(dataset_dir + dataset + '.mat')
    #     # supervised_select_feature(dataset)

    #     data, target = loaddata_multi_label(dataset_dir + dataset + '.mat')
    #     print(np.shape(data))
    #     print(np.shape(target))
    #     data = normalize_min_max_matrix(data)
    #     scio.savemat(dataset_dir + dataset + '.mat', {'data':data, 'target':target})

    # for dataset in datasets:
    #     print(dataset)
    #     data, target = loaddata_multi_label(dataset_dir + dataset + '.mat')
    #     dist = cdist(target, target, metric='cosine')
    #     print(dist)
    #     # print(np.sum(target, axis=1))
    #     print((dist + np.eye(np.shape(dist)[0]) < 0.4).any())

    # data1, target1 = loaddata_multi_label('./datasets/ML/' + 'emotions' + '.mat')
    # data2, target2 = loaddata_multi_label('./datasets/ML/' + 'bibtex' + '.mat')

    # # print((data1-data2).any())
    # # print((target1-target1).any())

    # dist1 = cdist(target1, target1, metric='cosine')
    # dist2 = cdist(target2, target2, metric='cosine')

    # # print(np.sum(target2[2]))
    # # print(np.sum(target2[-4]))
    # print(dist1)
    # print(dist2[0])
    # print((dist2 + np.eye(np.shape(dist2)[0]) < 0.7).any())

    # # process original `birds` dataset
    # data, target = loaddata_multi_label("./datasets/ML/" + "birds" + ".mat")
    # print(np.shape(data))
    # print(np.shape(data[:, :-1]))
    # scio.savemat(
    #         './datasets/ML/' + "birds" + '.mat', {
    #             'data': data[:, :-1],
    #             'target': target
    #         })

    # # process original `genbase` dataset
    # data, target = loaddata_multi_label("./datasets/ML/" + "genbase" + ".mat")
    # print(np.shape(data))
    # scio.savemat(
    #         './datasets/ML/' + "genbase" + '.mat', {
    #             'data': data[:, 1:],
    #             'target': target
    #         })

    # dataset_dir = './datasets/'
    # datasets = ['eurlex-dc-fold1', 'eurlex-ev-fold1', 'eurlex-sm-fold1']
    # for dataset in datasets:
    #     print(dataset)
    #     # split_ML_dataset(dataset)
    #     get_multi_label_dataset_statistics(dataset_dir + dataset + '_0' + '.mat')

    # # Partial Multi-Label Learning
    # datasets_dir = './datasets/PML/'
    # datasets = [
    #     # # 'CAL500',
    #     'medical',
    #     # 'genbase',
    #     # 'emotions',  # ?
    #     # # 'enron',  # ?
    #     # # 'image',  # ?
    #     # # 'languagelog',  # something wrong in dataset
    #     # 'scene',  # ?
    #     'yeast',
    #     # 'slashdot',
    #     # # 'llog',
    #     # 'arts',
    #     # # 'Corel5k',
    #     # 'rcv1subset1_top944',
    #     # # 'eurlex-dc-fold1_0',
    #     # # 'Corel16k002',
    #     # # 'bibtex',
    #     # # 'delicious_label_top30',
    #     # # 'birds',
    #     # # 'bookmarks_10000',
    #     # 'tmc2007-500',
    #     # 'eurlex-dc-fold1_10000_label_top200',
    #     # # 'mediamill_10000_label_top20',
    #     # # 'tmc2007-500_10000_feature_leq20',
    #     # 'eurlex-dc-fold1_10000_label_top100',
    #     # 'eurlex-sm-fold1_10000_label_top100',
    #     # # 'eurlex-ev-fold1_10000_label_top100',
    #     # 'eurlex-dc-fold1_10000_label_top50',
    #     # 'eurlex-sm-fold1_10000_label_top50',
    #     # # 'eurlex-ev-fold1_10000_label_top50',
    #     # 'nuswide-bow_10000',
    #     # 'nuswide-cvlad_10000',
    #     # # 'mediamill_10000',
    #     # 'OHSUMED',
    #     # 'eurlex-dc-fold1',
    #     # 'eurlex-sm-fold1',
    #     # # 'delicious_0',
    #     # # 'delicious_1',
    #     # # 'Image',
    #     # # 'emotions',
    #     # # 'Yeast',
    #     # # 'scene',
    #     # # 'eurlex_dc',
    #     # # 'eurlex_sm',
    #     # # 'mirflickr',
    #     # # 'music_emotion',
    #     # # 'music_style',
    #     # 'education',
    #     # 'science',
    #     # 'eurlex-dc-fold1_label_top2_102'
    # ]

    # for dataset in datasets:
    #     generate_save_train_test_split_PML(datasets_dir, dataset, test_size=0.3)

    # data, partial_target, target = loaddata(
    #     './datasets/BirdSong.mat')
    # scio.savemat(
    #     './datasets/BirdSong.mat', {
    #         'data': normalize_z_score(data),
    #         'partial_target': partial_target,
    #         'target': target
    #     })

    # dataset_dir = "./datasets/"
    # datasets = ["BirdSong", "FG-NET", "MSRCv2", "lost", "SoccerPlayer", "YahooNews"]
    # for dataset in datasets:
    #     data, partial_target, target = loaddata(
    #         './datasets/' + dataset + '.mat')
    #     scio.savemat(
    #         './datasets/' + dataset + '_Z.mat', {
    #             'data': normalize_z_score(data),
    #             'partial_target': partial_target,
    #             'target': target
    #         })


    datasets_dir = 'datasets/controlled_UCI/'
    datasets = ['abalone', 'pendigits', 'satimage', 'segment', 'usps', 'vehicle']

    Ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    Rs = [1, 2, 3]
    Epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    for dataset in datasets:
        for epsilon in Epsilons:
            dataset_filename = dataset + '_' + str(1) + '_' + str(1) + '_' + str(epsilon)
            print(dataset_filename)
            data, partial_target, target = loaddata(datasets_dir + dataset_filename + '.mat')
            scio.savemat(
                dataset + '_Z_' + str(1) + '_' + str(1) + '_' + str(epsilon) + '.mat', {
                    'data': normalize_z_score(data),
                    'partial_target': partial_target,
                    'target': target
                })

        for r in Rs:
            for p in Ps:
                dataset_filename = dataset + '_' + str(p) + '_' + str(r)
                print(dataset_filename)
                data, partial_target, target = loaddata(datasets_dir + dataset_filename + '.mat')
                scio.savemat(
                    dataset + '_Z_' + '_' + str(p) + '_' + str(r) + '.mat', {
                        'data': normalize_z_score(data),
                        'partial_target': partial_target,
                        'target': target
                    })

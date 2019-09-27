import json
import re
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

import xml.dom.minidom
from scipy.stats import variation
import statistics
import math
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering
import scipy.optimize
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

from sklearn import datasets
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from gensim.models import KeyedVectors


# import cmath as math
import operator
import sys
import os
from xml.etree import cElementTree as ElementTree

import xmltodict
import joblib

"""Copied from a repo : Class XmlListConfig and class XmlDictConfig"""


class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


def dict_traverse(dictionary, temp):
    for key in dictionary:
        temp_new = [i for i in temp] + [key] #re.sub(r"([A-Z])", r" \1", key).split()
        new_val = dictionary[key]
        if type(new_val) == dict or isinstance(new_val, XmlDictConfig):
            dict_traverse(new_val, temp_new)
        elif type(new_val) == list or isinstance(new_val, XmlListConfig):
            list_traverse(new_val, temp_new)
        else:
            print(temp_new)
            print(new_val)


def list_traverse(list_in, temp):
    if len(list_in) > 0:
        for val in list_in:
            if type(val) == dict or isinstance(val, XmlDictConfig):
                dict_traverse(val, temp)
            elif type(val) == list or isinstance(val, XmlListConfig):
                list_traverse(val, temp)
            else:
                print(temp)
                print(val)
    else:
        print(temp)
        print(None)


def dict_test(dictionary, temp, name_list, path_list, val_list):
    for key in dictionary:
        print("\n")
        print(key)
        print(temp)
        if key == 'string' or key == 'int' or key == 'array' or key == 'boolean':
            temp_new = [i for i in temp]
        else:
            temp_new = [i for i in temp] + [key]
        new_val = dictionary[key]
        if type(new_val) == dict or isinstance(new_val, XmlDictConfig):
            dict_test(new_val, temp_new, name_list, path_list, val_list)
        elif type(new_val) == list or isinstance(new_val, XmlListConfig):
            list_test(new_val, temp_new, name_list, path_list, val_list)
        else:
            # name_list.append(re.sub(r"([A-Z])", r" \1", temp_new[-1]).split())
            name_list.append(re.split('_| |:',re.sub(r"([A-Z])", r" \1", temp_new[-1])))
            # print("\nNAME_LIST")
            # print(name_list)
            # path_list.append([re.sub(r"([A-Z])", r" \1", i).split() for i in temp_new[:-1]])
            path_list.append([re.split('_| |:',re.sub(r"([A-Z])", r" \1", i)) for i in temp_new[:-1]])
            # print("\nPATH_LIST")
            # print(path_list)
            val_list.append(new_val)
            # print("\nVAL_LIST")
            # print(val_list)


def list_test(list_in, temp, name_list, path_list, val_list):
    if len(list_in) > 0:
        for val in list_in:
            if type(val) == dict or isinstance(val, XmlDictConfig):
                dict_test(val, temp, name_list, path_list, val_list)
            elif type(val) == list or isinstance(val, XmlListConfig):
                list_test(val, temp, name_list, path_list, val_list)
            else:
                # name_list.append(re.sub(r"([A-Z])", r" \1", temp[-1]).split())
                name_list.append(re.split('_| |:', re.sub(r"([A-Z])", r" \1", temp[-1])))
                # print("\nNAME_LIST")
                # print(name_list)
                # path_list.append([re.sub(r"([A-Z])", r" \1", i).split() for i in temp[:-1]])
                path_list.append([re.split('_| |:', re.sub(r"([A-Z])", r" \1", i)) for i in temp[:-1]])
                # print("\nPATH_LIST")
                # print(path_list)
                val_list.append(val)
                # print("\nVAL_LIST")
                # print(val_list)
    else:
        # name_list.append(re.sub(r"([A-Z])", r" \1", temp[-1]).split())
        name_list.append(re.split('_| |:', re.sub(r"([A-Z])", r" \1", temp[-1])))
        # print("\nNAME_LIST")
        # print(name_list)
        # path_list.append([re.sub(r"([A-Z])", r" \1", i).split() for i in temp[:-1]])
        path_list.append([re.split('_| |:', re.sub(r"([A-Z])", r" \1", i)) for i in temp[:-1]])
        # print("\nPATH_LIST")
        # print(path_list)
        val_list.append(val)
        # print("\nVAL_LIST")
        # print(val_list)


def dict_test_wso2(dictionary, temp, name_list, path_list, val_list):
    # print("\nNEW")
    attr_list = ['id', 'title', 'type']  # 'properties', 'items',
    for key in dictionary:
        new_val = dictionary[key]
        # print("\nKEY: %s" % key)
        # print("VAL: %s" % new_val)
        # print("\n")
        # print(key)
        if key in attr_list:
            temp_new = [i for i in temp]
        else:
            # print("TEMP UPDATED")
            temp_new = [i for i in temp] + [key]
        if type(new_val) == dict or isinstance(new_val, XmlDictConfig):
            dict_test_wso2(new_val, temp_new, name_list, path_list, val_list)
        elif type(new_val) == list or isinstance(new_val, XmlListConfig):
            list_test_wso2(new_val, temp_new, name_list, path_list, val_list)
        else:
            if key == 'id':
                try:
                    if temp_new[-1].isupper():
                        name_list.append(re.split('_| |:', temp_new[-1].lower()))
                    else:
                        name_list.append(re.split('_| |:', re.sub(r"([A-Z])", r" \1", temp_new[-1])))
                    # print("NAME LIST UPDATED")
                except IndexError:
                    name_list.append('NONE')
                    # print("NAME LIST UPDATED")
                try:
                    temp_list = []
                    for i in temp_new[:-1]:
                        if i.isupper():
                            temp_list.append(re.split('_| |:', i.lower()))
                        else:
                            temp_list.append(re.split('_| |:', re.sub(r"([A-Z])", r" \1", i)))
                    path_list.append(temp_list)
                    # path_list.append([re.split('_| |:', re.sub(r"([A-Z])", r" \1", i)) for i in temp_new[:-1]])
                    # print("PATH LIST UPDATED")
                except IndexError:
                    path_list.append('NONE')
            elif key == 'title':
                continue
            elif key == 'type':
                val_list.append(new_val)
                # print("VAL LIST UPDATED")
            else:
                continue
                # try:
                #     # name_list.append(re.sub(r"([A-Z])", r" \1", temp_new[-1]).split())
                #     name_list.append(re.split('_| |:',re.sub(r"([A-Z])", r" \1", temp_new[-1])))
                # except IndexError:
                #     continue
                # try:
                #     # path_list.append([re.sub(r"([A-Z])", r" \1", i).split() for i in temp_new[:-1]])
                #     path_list.append([re.split('_| |:',re.sub(r"([A-Z])", r" \1", i)) for i in temp_new[:-1]])
                # except IndexError:
                #     path_list.append([])

        # print(temp)
        # if key == 'id' or key == 'title':
        #     continue
        # if key == 'type':
        #     val_list.append(dictionary[key])
        #     continue
        # if key == 'string' or key == 'int' or key == 'array' or key == 'boolean':
        #     print("YES")
        #     temp_new = [i for i in temp]
        # else:
        #     print("NO")
        #     temp_new = [i for i in temp] + [key]
        #
        # if type(new_val) == dict or isinstance(new_val, XmlDictConfig):
        #     dict_test_wso2(new_val, temp_new, name_list, path_list, val_list)
        # elif type(new_val) == list or isinstance(new_val, XmlListConfig):
        #     list_test_wso2(new_val, temp_new, name_list, path_list, val_list)
        # else:
        #     # name_list.append(re.sub(r"([A-Z])", r" \1", temp_new[-1]).split())
        #     name_list.append(re.split('_| |:',re.sub(r"([A-Z])", r" \1", temp_new[-1])))
        #     # path_list.append([re.sub(r"([A-Z])", r" \1", i).split() for i in temp_new[:-1]])
        #     path_list.append([re.split('_| |:',re.sub(r"([A-Z])", r" \1", i)) for i in temp_new[:-1]])
        #     # val_list.append(new_val)
    # print("\nNAME_LIST")
    # print(name_list)
    # print("\nPATH_LIST")
    # print(path_list)
    # print("\nVAL_LIST")
    # print(val_list)


def list_test_wso2(list_in, temp, name_list, path_list, val_list):
    if len(list_in) > 0:
        for val in list_in:
            if type(val) == dict or isinstance(val, XmlDictConfig):
                dict_test_wso2(val, temp, name_list, path_list, val_list)
            elif type(val) == list or isinstance(val, XmlListConfig):
                list_test_wso2(val, temp, name_list, path_list, val_list)
            else:
                if key == 'id':
                    try:
                        if temp[-1].isupper():
                            name_list.append(re.split('_| |:', temp[-1].lower()))
                        else:
                            name_list.append(re.split('_| |:', re.sub(r"([A-Z])", r" \1", temp[-1])))
                        # name_list.append(re.split('_| |:', re.sub(r"([A-Z])", r" \1", temp[-1])))
                        # print("NAME LIST UPDATED")
                    except IndexError:
                        name_list.append('NONE')
                        # print("NAME LIST UPDATED")
                    try:
                        temp_list = []
                        for i in temp[:-1]:
                            if i.isupper():
                                temp_list.append(re.split('_| |:', i.lower()))
                            else:
                                temp_list.append(re.split('_| |:', re.sub(r"([A-Z])", r" \1", i)))
                        path_list.append(temp_list)
                        # path_list.append([re.split('_| |:', re.sub(r"([A-Z])", r" \1", i)) for i in temp[:-1]])
                        # print("PATH LIST UPDATED")
                    except IndexError:
                        path_list.append('NONE')
                        # print("PATH LIST UPDATED")
                elif key == 'title':
                    continue
                elif key == 'type':
                    val_list.append(val)
                    # print("VAL LIST UPDATED")
                else:
                    continue
                # # name_list.append(re.sub(r"([A-Z])", r" \1", temp[-1]).split())
                # name_list.append(re.split('_| |:', re.sub(r"([A-Z])", r" \1", temp[-1])))
                # # print("\nNAME_LIST")
                # # print(name_list)
                # # path_list.append([re.sub(r"([A-Z])", r" \1", i).split() for i in temp[:-1]])
                # path_list.append([re.split('_| |:', re.sub(r"([A-Z])", r" \1", i)) for i in temp[:-1]])
                # # print("\nPATH_LIST")
                # # print(path_list)
                # val_list.append(val)
                # # print("\nVAL_LIST")
                # # print(val_list)
    else:
        if key == 'id':
            try:
                if temp[-1].isupper():
                    name_list.append(re.split('_| |:', temp[-1].lower()))
                else:
                    name_list.append(re.split('_| |:', re.sub(r"([A-Z])", r" \1", temp[-1])))
                # name_list.append(re.split('_| |:', re.sub(r"([A-Z])", r" \1", temp[-1])))
                # print("NAME LIST UPDATED")
            except IndexError:
                name_list.append("NONE")
                # print("ERROR")
            try:
                temp_list = []
                for i in temp[:-1]:
                    if i.isupper():
                        temp_list.append(re.split('_| |:', i.lower()))
                    else:
                        temp_list.append(re.split('_| |:', re.sub(r"([A-Z])", r" \1", i)))
                path_list.append(temp_list)
                # path_list.append([re.split('_| |:', re.sub(r"([A-Z])", r" \1", i)) for i in temp[:-1]])
                # print("PATH LIST UPDATED")
            except IndexError:
                path_list.append("NONE")
                # print("ERROR")
        elif key == 'title':
            print("ERROR")
        elif key == 'type':
            val_list.append(val)
            # print("VAL LIST UPDATED")
        else:
            print("ERROR")
            # print("ERROR")
        # # name_list.append(re.sub(r"([A-Z])", r" \1", temp[-1]).split())
        # name_list.append(re.split('_| |:', re.sub(r"([A-Z])", r" \1", temp[-1])))
        # # print("\nNAME_LIST")
        # # print(name_list)
        # # path_list.append([re.sub(r"([A-Z])", r" \1", i).split() for i in temp[:-1]])
        # path_list.append([re.split('_| |:', re.sub(r"([A-Z])", r" \1", i)) for i in temp[:-1]])
        # # print("\nPATH_LIST")
        # # print(path_list)
        # val_list.append(val)
        # # print("\nVAL_LIST")
        # # print(val_list)


def get_features(tag_list):
    path = '/Users/ayodhya/PycharmProjects/word2vec/vectors/default'
    # path = '/Users/ayodhya/Documents/GitHub/Data_mapping/word2vec_vectors'
    wv = KeyedVectors.load(path, mmap='r')

    num_keys = 0
    num_unavail_keys = 0
    mean_vectors = []

    for name in tag_list:
        vectors = []
        for word in name:
            num_keys += 1
            try:
                vectors.append(wv[word.lower()])
            except KeyError:
                vectors.append([0 * i for i in range(150)])
                num_unavail_keys += 1
        mean_vectors.append(np.mean(vectors, axis=0).tolist())  # Find mean embedding

    # Percentage of unavailable keys
    print("Number of keys: %u" % num_keys)
    print("Number of unavailable keys: %u" % num_unavail_keys)
    print(num_unavail_keys / num_keys * 100)

    return mean_vectors


def get_features_title(names, paths, values):
    tag_list = []
    feature_list = []
    dirname = os.getcwd()
    abspath = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(abspath, "default")

    wv = KeyedVectors.load(path, mmap='r')

    data_type = {'none': [0, 0, 0, 0, 0, 0, 0, 0], 'string': [1, 0, 0, 0, 0, 0, 0, 0], 'int': [0, 1, 0, 0, 0, 0, 0, 0], 'array': [0, 0, 1, 0, 0, 0, 0, 0],
                 'object': [0, 0, 0, 1, 0, 0, 0, 0], 'boolean': [0, 0, 0, 0, 1, 0, 0, 0], 'number': [0, 0, 0, 0, 0, 1, 0, 0], 'null': [0, 0, 0, 0, 0, 0, 1, 0]}
    for i in range(len(names)):
        if names[i] == ['id']:
            try:
                attribute = paths[i][-1]
                # print(attribute)
                word_embed = []
                for words in attribute:
                    # print(words)
                    try:
                        word_embed.append(wv[words.lower()])
                    except KeyError:
                        word_embed.append([0*i for i in range(150)])
                # print(word_embed)
                word_embed = np.mean(word_embed, axis=0).tolist()
                if names[i+1] == ['type']:
                    type_embed = data_type[values[i+1].lower()]
                else:
                    type_embed = data_type['none']

                norm_word_embed = []
                for val in word_embed:
                    try:
                        norm_word_embed.append((val - min(word_embed)) / (max(word_embed) - min(word_embed)))
                    except ZeroDivisionError:
                        if val>1:
                            norm_word_embed.append(1)
                        elif val<0:
                            norm_word_embed.append(0)
                        else:
                            norm_word_embed.append(val)

                feature = norm_word_embed + type_embed

                tag_list.append(attribute)
                feature_list.append(feature)
            except IndexError:
                continue
    path_list = []
    for paths_name in paths:
        path_list.append(paths_name[:-1])

    return [tag_list, feature_list, path_list]


def get_features_title_wso2(names, paths, values):
    tag_list = []
    feature_list = []
    path_list = []
    dirname = os.getcwd()
    abspath = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(abspath, "default")

    wv = KeyedVectors.load(path, mmap='r')

    data_type = {'none': [0, 0, 0, 0, 0, 0, 0, 0], 'string': [1, 0, 0, 0, 0, 0, 0, 0], 'int': [0, 1, 0, 0, 0, 0, 0, 0], 'array': [0, 0, 1, 0, 0, 0, 0, 0],
                 'object': [0, 0, 0, 1, 0, 0, 0, 0], 'boolean': [0, 0, 0, 0, 1, 0, 0, 0], 'number': [0, 0, 0, 0, 0, 1, 0, 0], 'null': [0, 0, 0, 0, 0, 0, 1, 0]}
    for i in range(len(names)):
        if values[i] in ['object', 'array']:
            continue
        else:
            attribute = names[i]
            # print("ATTRIBUTTE")
            # print(attribute)
            word_embed = []
            for words in attribute:
                try:
                    word_embed.append(wv[words.lower()])
                    # print(wv[words.lower()])
                except KeyError:
                    # print("ERROR KEY")
                    word_embed.append([0 * i for i in range(150)])
            word_embed = np.mean(word_embed, axis=0).tolist()
            type_embed = data_type[values[i]]

            norm_word_embed = []
            # print("WORD EMBEDD")
            # print(word_embed)
            for val in word_embed:
                try:
                    norm_word_embed.append((val - min(word_embed)) / (max(word_embed) - min(word_embed)))
                except ZeroDivisionError:
                    if val > 1:
                        norm_word_embed.append(1)
                    elif val < 0:
                        norm_word_embed.append(0)
                    else:
                        norm_word_embed.append(val)
            feature = norm_word_embed + type_embed

            tag_list.append(attribute)
            feature_list.append(feature)
            path_list.append(paths[i])
    return [tag_list, feature_list, path_list]
        # if names[i] == ['id']:
        #     try:
        #         attribute = paths[i][-1]
        #         print(attribute)
        #         word_embed = []
        #         for words in attribute:
                    # print(words)
                    # try:
                    #     word_embed.append(wv[words.lower()])
                    # except KeyError:
                    #     word_embed.append([0*i for i in range(150)])
                # print(word_embed)
                # word_embed = np.mean(word_embed, axis=0).tolist()
                # if names[i+1] == ['type']:
                #     type_embed = data_type[values[i+1].lower()]
                # else:
                #     type_embed = data_type['none']
                #
                # norm_word_embed = []
                # for val in word_embed:
                #     try:
                #         norm_word_embed.append((val - min(word_embed)) / (max(word_embed) - min(word_embed)))
                #     except ZeroDivisionError:
                #         if val>1:
                #             norm_word_embed.append(1)
                #         elif val<0:
                #             norm_word_embed.append(0)
                #         else:
                #             norm_word_embed.append(val)
                #
                # feature = norm_word_embed + type_embed
                #
                # tag_list.append(attribute)
                # feature_list.append(feature)
            # except IndexError:
            #     continue
    # path_list = []
    # for paths_name in paths:
    #     path_list.append(paths_name[:-1])
    #
    # return [tag_list, feature_list, path_list]


def xg_nn(batch_x, batch_y, num_class):
    d_train = xgb.DMatrix(batch_x, label=batch_y)
    param = {
        'eta': 0.3,
        'max_depth': 3,
        'objective': 'multi:softprob',  # 'reg:linear',  # 'multi:softprob',
        'num_class': num_class}
    steps = 1000  # The number of training iterations
    l1_params = [1, 10, 100]
    rmses_l1 = []

    model = xgb.train(param, d_train, steps)
    joblib.dump(model, "XG_trained_model")
    return


def train_nn(files, num_clustrs):
    tag_list = []
    feature_list = []
    i = 0
    for f in files:
        print(i)
        i = i + 1
        print(f)
        with open(f) as g:
            distros_dict = json.load(g)

        list_att_1 = []
        names_1 = []
        paths_1 = []
        values_1 = []
        # print("DICTIONARY")
        # print(distros_dict)
        dict_test_wso2(distros_dict, list_att_1, names_1, paths_1, values_1)

        out_feat = get_features_title_wso2(names_1, paths_1, values_1)
        tag_list = tag_list + out_feat[0]
        feature_list = feature_list + out_feat[1]

    ''' Hierarchical clustering '''
    plt.figure(figsize=(10, 7))
    plt.title("Clusters")
    get_link = shc.linkage(feature_list, method='ward')
    dend = shc.dendrogram(get_link, leaf_font_size=8, leaf_rotation=90., labels=tag_list)
    plt.axhline(y=1, color='r', linestyle='--')
    # plt.show()

    ''' Ground truth for training '''
    num_clusters = num_clustrs #13
    num_feat = 158

    cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    out_classes = cluster.fit_predict(feature_list)

    # print(out_classes)

    out = []
    for class_name in out_classes:
        one_hot_vec = [0 * i for i in range(class_name)] + [1] + [0 * i for i in range(num_clusters - class_name - 1)]
        # print (one_hot_vec)
        out.append(one_hot_vec)

    # print(out)

    xg_nn(feature_list, out_classes, num_clusters)


def predict_nn(file):
    with open(file) as f:
        distros_dict = json.load(f)

    list_att = []
    names = []
    paths = []
    values = []
    # print("DICTIONARY")
    # print(distros_dict)
    dict_test_wso2(distros_dict, list_att, names, paths, values)
    # print("FEATURES")
    # print("LIST ATTR")
    # print(list_att)
    # print("NAMES")
    # print(names)
    # print("PATHS")
    # print(paths)
    # print("VALUES")
    # print(values)

    out_feat = get_features_title_wso2(names, paths, values)
    tag_list = out_feat[0]
    feature_list = out_feat[1]
    path_list = out_feat[2]
    # print("FEATURES_2")
    # print("TAG LIST")
    # print(tag_list)
    # print("PATHS")
    # print(path_list)

    model = joblib.load("XG_trained_model")

    d_test = xgb.DMatrix(feature_list)

    preds = model.predict(d_test)

    # for name in range(len(tag_list)):
    #     print(tag_list[name])
    #     print(preds.argmax(1)[name])
    return [tag_list, preds.argmax(1), path_list]


def indices(list_in, value):
    new_list = []
    for i in range(len(list_in)):
        if list_in[i]==value:
            new_list.append(i)
    return new_list


def main():
    dirname = os.getcwd()
    abspath = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(abspath, "Datasets/GC/")
    path = os.path.join(abspath, "Datasets/Connector_schemas/")
    files = []

    for r, d, f in os.walk(path):
        for file in f:
            if '.json' in file:
                files.append(os.path.join(r, file))

    num_clusters = 20
    # train_nn(files, num_clusters)
    # predictions_1 = predict_nn('/Users/ayodhya/Documents/GitHub/Data_mapping/Datasets/Test/test_input_4.json')
    predictions_1 = predict_nn('/Users/ayodhya/Documents/GitHub/Data_mapping/Datasets/Test/in_15.json')
    # print(predictions)
    # print("\n")
    # predictions_2 = predict_nn('/Users/ayodhya/Documents/GitHub/Data_mapping/Datasets/Test/test_output_4.json')
    predictions_2 = predict_nn('/Users/ayodhya/Documents/GitHub/Data_mapping/Datasets/Test/out_15.json')

    tag_1 = predictions_1[0]
    tag_2 = predictions_2[0]
    # print("TAGS_1")
    # print(tag_1)
    # print("TAGS_2")
    # print(tag_2)

    class_name_1 = predictions_1[1]
    class_name_2 = predictions_2[1]

    path_1 = predictions_1[2]
    path_2 = predictions_2[2]

    # print(tag_1)
    # print(class_name_1)
    # print(tag_2)
    # print(class_name_2)

    path = os.path.join(abspath, "default")
    wv = KeyedVectors.load(path, mmap='r')

    print("\n")

    for i in range(num_clusters):
        path_list_1 = []
        path_list_2 = []
        label_list_1 = []
        label_list_2 = []

        index_list_1 = indices(class_name_1, i)
        index_list_2 = indices(class_name_2, i)

        # print("\n")
        # print("Class %u" % (i + 1))
        for item in index_list_1:
            label = tag_1[item]
            label_list_1.append(label)
            # print(label)
            path_list_1.append(path_1[item])
            # print(path_1[item])
        # print("___________")
        for item in index_list_2:
            label = tag_2[item]
            label_list_2.append(label)
            # print(label)
            path_list_2.append(path_2[item])
            # print(path_2[item])
        # print("_______________________________")

        num_attr = 0
        num_not_match = 0

        score_mat = []

        for j in range(len(label_list_1)):
            score_list = []
            num_attr += 1
            # print("\nJ: %u" % j)
            aim_list = path_list_1[j] + [label_list_1[j]]  # [label_list_1[j]]  #
            # print("AIM LIST")
            # print(aim_list)
            aim_vector = []
            for aim in aim_list:
                word_vector = []
                for word in aim:
                    try:
                        word_vector.append(wv[word.lower()])
                    except KeyError:
                        # print("WORD")
                        # print(word)
                        word_vector.append([0*i for i in range(150)])
                aim_vector.append(np.mean(word_vector, axis=0))
                # print(aim_vector[-1])
            # print(aim_list[-1])
            # index_list = []
            # print(features_1[j])
            for i in range(len(label_list_2)):
                score = 0
                candidate_list = path_list_2[i] + [label_list_2[i]]  # [label_list_2[i]]  #
                # print("CANDIDATE LIST")
                # print(candidate_list)
                cand_vector = []
                for cand in candidate_list:
                    word_vector = []
                    for word in cand:
                        try:
                            word_vector.append(wv[word.lower()])
                        except KeyError:
                            # print("KEY ERROR")
                            # print("WORD")
                            # print(word)
                            word_vector.append([0 * i for i in range(150)])
                    cand_vector.append(np.mean(word_vector, axis=0))
                    # print(cand_vector[-1])
                # print(candidate_list[-1])

                # Give priority to attribute name
                priority = 10
                # print(aim_list[-1])
                # print(candidate_list[-1])
                if aim_list[-1] == candidate_list[-1]:
                    score = 0
                # elif aim_list[-1].any() == candidate_list[-1].any():
                #     score = 1
                else:
                    # print("ELSE")
                    # print(aim_vector[-1])
                    # print(cand_vector[-1])
                    # print(np.dot(aim_vector[-1], cand_vector[-1]))
                    if (np.linalg.norm(aim_vector[-1], ord=2) * np.linalg.norm(cand_vector[-1], ord=2)) == 0:
                        score = 5
                        # print("ELSE")
                    else:
                        # print("ELSE")
                        score = priority - (priority * np.dot(aim_vector[-1], cand_vector[-1]) / (np.linalg.norm(aim_vector[-1], ord=2) * np.linalg.norm(cand_vector[-1], ord=2)))
                # print(score)
                simi = 0
                for vec_1 in aim_vector[:-1]:
                    for vec_2 in cand_vector[:-1]:
                        if (np.linalg.norm(vec_1, ord=2) * np.linalg.norm(vec_2, ord=2)) == 0:
                            simi = simi
                        else:
                            simi += np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1, ord=2) * np.linalg.norm(vec_2, ord=2))
                try:
                    simi = 10 - (simi / (len(aim_vector[:-1]) * len(cand_vector[:-1])))
                except ZeroDivisionError:
                    if len(aim_vector[:-1]) == 0 and len(cand_vector[:-1]) == 0:
                        simi = 9
                    else:
                        simi = 10
                # print("SCORE")
                # print(score)
                # print("SIMI")
                # print(simi)
                score_list.append(score + simi)
                # print(score_list)
            score_mat.append(score_list)
            # print(score_mat)

        #             score_row = []
        #             for aim in aim_list:
        #                 for cand_word in candidate:
        #                     for aim_word in aim:
        #                         if cand_word.lower() == aim_word.lower():
        #                             score_row.append(1)
        #                         else:
        #                             try:
        #                                 score_row.append(wv.similarity(w1=cand_word.lower(), w2=aim_word.lower()))
        #                                 # print(cand_word)
        #                                 # print(aim_word)
        #                                 # print(wv.similarity(w1=cand_word, w2=aim_word))
        #                             except KeyError:
        #                                 # print("KEYERROR")
        #                                 # print(cand_word)
        #                                 # print(aim_word)
        #                                 score_row.append(0)
        #             # print(score_row)
        #             score += max(score_row)  # Done
        #         score_list.append(score / (len(candidate_list)))  # Done
        #         # print(score / (len(candidate_list)))
        #     # print(score_list)
        #     score_mat.append(score_list)
        #
        try:
            matching_pairs_3 = scipy.optimize.linear_sum_assignment(score_mat)
        except ValueError:
            continue

        # print(matching_pairs_3)

        for ind in range(len(matching_pairs_3[0])):
            ind_1 = matching_pairs_3[0][ind]
            ind_2 = matching_pairs_3[1][ind]
            aim_list = path_list_1[ind_1] + [label_list_1[ind_1]]  # [label_list_1[ind_1]]  #
            selected_list = path_list_2[ind_2] + [label_list_2[ind_2]]  # [label_list_2[ind_2]]  #
            # print('%s : %s' % (label_list_1[], label_list_2[])),
            print("%s : \n%s\n" % ((",".join(["_".join(i) for i in aim_list])), (",".join(["_".join(i) for i in selected_list]))))

        # dict_traverse()

            # try:
            #     if max(score_list) > 0:
            #         selected_index = index_list[score_list.index(max(score_list))]
            #         selected_list = path_2[selected_index] + [tag_2[selected_index]]
            #
            #         # print("\nSelected attribute: ")
            #         # print(max(score_list))
            #         print("%s : %s\n" % ((",".join(["".join(i) for i in aim_list])), (",".join(["".join(i) for i in selected_list]))))
            #     else:
            #         num_not_match += 1
            # except ValueError:
            #     # print("No matching out")
            #     num_not_match += 1
            #     continue

        # print(num_not_match / num_attr * 100)
        # print(num_not_match)




        # simi_list = []
        # for i_attr in label_list_1:
        #     i_attr_list = i_attr.split("_")
        #     print(i_attr_list)
        #     simi_row = []
        #     for j_attr in label_list_2:
        #         j_attr_list = j_attr.split("_")
        #         print(j_attr_list)
        #         temp_list = []
        #         for word_part_1 in i_attr_list:
        #             for word_part_2 in j_attr_list:
        #                 try:
        #                     simi = wv.similarity(w1=word_part_1, w2=word_part_2)
        #                 except KeyError:
        #                     # simi = 0
        #                     # print("###########")
        #                     # print(word_part_1)
        #                     # print(word_part_2)
        #                     # print("###########")
        #                     try:
        #                         simi = wv_1.similarity(w1=word_part_1, w2=word_part_2)
        #                     except KeyError:
        #                         simi = 0
        #                 temp_list.append(simi)
        #                 # print("%s , %s, %f" % (word_part_1, word_part_2, simi))
        #
        #         simi_row.append(1 - max(temp_list))
        #     simi_list.append(simi_row)
        #
        # try:
        #     matching_pairs_3 = scipy.optimize.linear_sum_assignment(simi_list)
        # except ValueError:
        #     continue
        # for ind in range(len(matching_pairs_3[0])):
        #     print('%s : %s' % (label_list_1[matching_pairs_3[0][ind]], label_list_2[matching_pairs_3[1][ind]])),

    """

    path = '/Users/ayodhya/PycharmProjects/word2vec/vectors/default'
    # path = '/Users/ayodhya/Documents/GitHub/Data_mapping/word2vec_vectors'
    wv = KeyedVectors.load(path, mmap='r')
    #
    with open("/Users/ayodhya/Documents/GitHub/Data_mapping/Datasets/data-mapper-sample-config/1/in.json", 'r') as f:
        distros_dict = json.load(f)

    # with open("/Users/ayodhya/Documents/GitHub/Data_mapping/Datasets/data-mapper-sample-config/2/response.json", 'r') as f:
    #     distros_dict = json.load(f)

    # print(distros_dict)

    list_att_1 = []
    names_1 = []
    paths_1 = []
    dict_test(distros_dict, list_att_1, names_1, paths_1)

    with open("/Users/ayodhya/Documents/GitHub/Data_mapping/Datasets/data-mapper-sample-config/1/out.json", 'r') as f:
        distros_dict = json.load(f)

    # tree = ElementTree.parse(
    #     "/Users/ayodhya/Documents/GitHub/Data_mapping/Datasets/data-mapper-sample-config/2/input.xml")
    # root = tree.getroot()
    # distros_dict = XmlDictConfig(root)

    list_att_2 = []
    names_2 = []
    paths_2 = []
    dict_test(distros_dict, list_att_2, names_2, paths_2)

    mean_embeddings_1 = get_features(names_1)
    mean_embeddings_2 = get_features(names_2)

    mean_embeddings_total = mean_embeddings_1 + mean_embeddings_2

    num_clusters = 100  # 100  # 500  # math.ceil(len(mean_embeddings_total)/2)
    cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    features_total = cluster.fit_predict(mean_embeddings_total)

    features_1 = features_total[0:len(mean_embeddings_1)]
    features_2 = features_total[len(mean_embeddings_1):]

    num_attr = 0
    num_not_match = 0
    for j in range(len(names_1)):
        num_attr += 1
        print("\nJ: %u" % j)
        aim_list = paths_1[j] + [names_1[j]]
        # print(aim_list)
        score_list = []
        index_list = []
        # print(features_1[j])
        for i in range(len(features_2)):
            if features_2[i] == features_1[j]:
                index_list.append(i)
                # print("I: %u" % i)
                candidate_list = paths_2[i] + [names_2[i]]
                # print(candidate_list)

                # score = 0
                # for candidate in candidate_list:
                #     if candidate in aim_list:
                #         score += 1
                #
                # score_list.append(score / (len(candidate_list)))

                score = 0
                for candidate in candidate_list:
                    score_row = []
                    for aim in aim_list:
                        for cand_word in candidate:
                            for aim_word in aim:
                                try:
                                    score_row.append(wv.similarity(w1=cand_word.lower(), w2=aim_word.lower()))
                                    # print(cand_word)
                                    # print(aim_word)
                                    # print(wv.similarity(w1=cand_word, w2=aim_word))
                                except KeyError:
                                    score_row.append(0)
                    # print(score_row)
                    score += max(score_row)  # Done
                score_list.append(score / (len(candidate_list)))  # Done
                # print(score / (len(candidate_list)))
        # print(score_list)
        try:
            if max(score_list) > 0:
                selected_index = index_list[score_list.index(max(score_list))]
                selected_list = paths_2[selected_index] + [names_2[selected_index]]

                # print("\nSelected attribute: ")
                # print(max(score_list))
                print("%s : \n%s\n" % ((",".join(["".join(i) for i in aim_list])), (",".join(["".join(i) for i in selected_list]))))
            else:
                num_not_match += 1
        except ValueError:
            # print("No matching out")
            num_not_match += 1
            continue

    print(num_not_match/num_attr*100)
    print(num_not_match)

    # Traverse through doc

    tem_list = []
    # dict_traverse(distros_dict, tem_list)
    tree = ElementTree.parse(
        "/Users/ayodhya/Documents/GitHub/Data_mapping/Datasets/data-mapper-sample-config/2/input.xml")
    root = tree.getroot()
    tem_list = []
    distros_dict = XmlDictConfig(root)
    # dict_traverse(distros_dict, tem_list)
    
    """


if __name__ == "__main__":
    main()

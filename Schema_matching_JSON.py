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

"""Copied from a repo : XmlListConfig and XmlDictConfig"""


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


def dict_test(dictionary, temp, name_list, path_list):
    for key in dictionary:
        if key == 'string' or key == 'int' or key == 'array' or key == 'boolean':
            temp_new = [i for i in temp]
        else:
            temp_new = [i for i in temp] + [key]
        new_val = dictionary[key]
        if type(new_val) == dict or isinstance(new_val, XmlDictConfig):
            dict_test(new_val, temp_new, name_list, path_list)
        elif type(new_val) == list or isinstance(new_val, XmlListConfig):
            list_test(new_val, temp_new, name_list, path_list)
        else:
            # name_list.append(re.sub(r"([A-Z])", r" \1", temp_new[-1]).split())
            name_list.append(re.split('_| ',re.sub(r"([A-Z])", r" \1", temp_new[-1])))
            # path_list.append([re.sub(r"([A-Z])", r" \1", i).split() for i in temp_new[:-1]])
            path_list.append([re.split('_| ',re.sub(r"([A-Z])", r" \1", i)) for i in temp_new[:-1]])


def list_test(list_in, temp, name_list, path_list):
    if len(list_in) > 0:
        for val in list_in:
            if type(val) == dict or isinstance(val, XmlDictConfig):
                dict_test(val, temp, name_list, path_list)
            elif type(val) == list or isinstance(val, XmlListConfig):
                list_test(val, temp, name_list, path_list)
            else:
                # name_list.append(re.sub(r"([A-Z])", r" \1", temp[-1]).split())
                name_list.append(re.split('_| ', re.sub(r"([A-Z])", r" \1", temp[-1])))
                # path_list.append([re.sub(r"([A-Z])", r" \1", i).split() for i in temp[:-1]])
                path_list.append([re.split('_| ', re.sub(r"([A-Z])", r" \1", i)) for i in temp[:-1]])
    else:
        # name_list.append(re.sub(r"([A-Z])", r" \1", temp[-1]).split())
        name_list.append(re.split('_| ', re.sub(r"([A-Z])", r" \1", temp[-1])))
        # path_list.append([re.sub(r"([A-Z])", r" \1", i).split() for i in temp[:-1]])
        path_list.append([re.split('_| ', re.sub(r"([A-Z])", r" \1", i)) for i in temp[:-1]])


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


def main():
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


if __name__ == "__main__":
    main()

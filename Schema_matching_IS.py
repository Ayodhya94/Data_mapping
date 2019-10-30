import json
import re
# from gensim.models import KeyedVectors
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

import operator
import sys
import os
from xml.etree import cElementTree as ElementTree

import xmltodict
import joblib

"""class XmlListConfig and class XmlDictConfig were copied"""


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


def indices(list_in, value):
    new_list = []
    for i in range(len(list_in)):
        if list_in[i] == value:
            new_list.append(i)
    return new_list


def normalize(raw_list):
    normalized_list = []  # Need to normalize word embedding values
    for val in raw_list:
        try:
            normalized_list.append((val - min(raw_list)) / (max(raw_list) - min(raw_list)))
        except ZeroDivisionError:
            if val > 1:
                normalized_list.append(1)
            elif val < 0:
                normalized_list.append(0)
            else:
                normalized_list.append(val)
    return normalized_list


def add_element_to_list(word, updating_list):
    if word.isupper():
        updating_list.append(list(filter(''.__ne__, (re.split('[_ :]', word.lower())))))
    else:
        updating_list.append(list(filter(''.__ne__, (
            re.split('[_ :]', re.sub(r"([A-Z]+[a-z0-9_\W])", r" \1", word + "_").lower())))))
    return


def update_lists(key, info, temp_new, new_val):
    if key == 'id':
        try:
            add_element_to_list(temp_new[-1], info["names"])
        except IndexError:
            info["names"].append('NONE')
        try:
            temp_list = []
            for i in temp_new[:-1]:
                add_element_to_list(i, temp_list)
            info["paths"].append(temp_list)
        except IndexError:
            info["paths"].append('NONE')
    elif key == 'type':
        info["values"].append(new_val)
    return


def dict_traverse(dictionary, temp, info):
    attr_list = ['id', 'title', 'type']  # 'properties', 'items',
    for key in dictionary:
        new_val = dictionary[key]
        if key in attr_list:
            temp_new = [i for i in temp]
        else:
            temp_new = [i for i in temp] + [key]
        if type(new_val) == dict or isinstance(new_val, XmlDictConfig):
            dict_traverse(new_val, temp_new, info)
        elif type(new_val) == list or isinstance(new_val, XmlListConfig):
            list_traverse(new_val, temp_new, info)
        else:
            update_lists(key, info, temp_new, new_val)


def list_traverse(list_in, temp, info):
    if len(list_in) > 0:
        for val in list_in:
            if type(val) == dict or isinstance(val, XmlDictConfig):
                dict_traverse(val, temp, info)
            elif type(val) == list or isinstance(val, XmlListConfig):
                list_traverse(val, temp, info)
            else:
                update_lists('id', info, temp, val)
                # update_lists(key, info, temp, val)


def word_embed(attribute, wv):
    embed = []
    for words in attribute:
        try:
            embed.append(wv[words.lower()])
        except KeyError:  # 0 embedding is used for words that are not in w2v model
            embed.append([0 * i for i in range(150)])
    embed = np.mean(embed, axis=0).tolist()
    return embed


def type_embed(value):
    data_types = ['none', 'string', 'int', 'array', 'object', 'boolean', 'number', 'null']
    embed = []
    for ite in data_types:
        if value == ite:
            embed.append(1)
        else:
            embed.append(0)
    return embed


def class_embed(value):
    special_class = ['id', 'phone', 'name', 'num', 'age', 'date', 'value', 'type', 'code']
    embed = []
    for ite in special_class:
        if ite in value:
            embed.append(1)
        else:
            embed.append(0)
    return embed


def get_features(info, wv):
    tag_list = []
    feature_list = []
    path_list = []

    for i in range(len(info["names"])):
        if info["values"][i] in ['object', 'array', 'null']:
            # Neglect list, object, array names which are not real attributes
            continue
        else:
            word_embed_list = normalize(word_embed(info["names"][i], wv))
            type_embed_list = type_embed(info["values"][i])
            class_embed_list = class_embed((''.join(info["names"][i])).lower())

            feature = word_embed_list + type_embed_list + class_embed_list

            tag_list.append(info["names"][i])
            feature_list.append(feature)
            path_list.append(info["paths"][i])

    return [tag_list, feature_list, path_list]


def xg_train(batch_x, batch_y, num_class):
    d_train = xgb.DMatrix(batch_x, label=batch_y)
    param = {
        'eta': 0.3,
        'max_depth': 3,
        'objective': 'multi:softprob',  # 'reg:linear',  # 'multi:softprob',
        'num_class': num_class}
    steps = 1000  # The number of training iterations
    model = xgb.train(param, d_train, steps)
    joblib.dump(model, "Schema_matching_models/XG_trained_model_test")
    return


def hierarchical_clustering(feature_list, tag_list, num_clusters, plot):
    if plot:
        plt.figure(figsize=(10, 7))
        plt.title("Clusters")
        get_link = shc.linkage(feature_list, method='ward')
        dend = shc.dendrogram(get_link, leaf_font_size=8, leaf_rotation=90., labels=tag_list)
        plt.axhline(y=1, color='r', linestyle='--')
        plt.show()

    cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    out_classes = cluster.fit_predict(feature_list)
    return out_classes





def predict(files):
    """ This method is for predicting using model"""
    ''' Getting features '''
    with open('W2V_models/w2v_dict.json') as f:
        wv = json.load(f)

    with open(files) as f:
        distros_dict = json.load(f)

    list_att = []
    info = {"names": [], "paths": [], "values": []}

    dict_traverse(distros_dict, list_att, info)

    out_feat = get_features(info, wv)

    ''' Loading model and predicting'''
    model = joblib.load("Schema_matching_models/XG_trained_model")
    d_test = xgb.DMatrix(out_feat[1])
    preds = model.predict(d_test)

    return [out_feat[0], preds.argmax(1), out_feat[2]]


def info_class(index_list, label_list, path_list, predictions):
    for item in index_list:
        label_list.append(predictions[0][item])
        print(predictions[0][item])
        path_list.append(predictions[2][item])
    print("___________")
    return


def classifications(num_clusters, predictions_1, predictions_2):
    """ This method is for finding classifications"""
    classes = {}
    for i in range(num_clusters):
        path_list_1 = []
        path_list_2 = []
        label_list_1 = []
        label_list_2 = []
        index_list_1 = indices(predictions_1[1], i)
        index_list_2 = indices(predictions_2[1], i)

        class_name = "Class_" + str(i)

        print("\n")
        print("Class %u" % (i + 1))
        info_class(index_list_1, label_list_1, path_list_1, predictions_1)
        info_class(index_list_2, label_list_2, path_list_2, predictions_2)

        classes[class_name] = [label_list_1, path_list_1, label_list_2, path_list_2]
    return classes


def score_attribute_name(aim_list, candidate_list, aim_vector, cand_vector, weight):
    if aim_list[-1] == candidate_list[-1]:
        score = 0  # Give priority to attribute name
    # elif aim_list[-1].any() == candidate_list[-1].any():
    #     score = 1
    else:
        if (np.linalg.norm(aim_vector[-1], ord=2) * np.linalg.norm(cand_vector[-1], ord=2)) == 0:
            score = 5
        else:
            score = weight - ((weight - 1) * np.dot(aim_vector[-1], cand_vector[-1]) / (
                    np.linalg.norm(aim_vector[-1], ord=2) * np.linalg.norm(cand_vector[-1], ord=2)))
    # print(score)
    return score


def score_hierarchy(aim_vector, cand_vector, weight):
    similarity = 0
    for vec_1 in aim_vector:  # [:-1]:
        for vec_2 in cand_vector:  # [:-1]:
            if (np.linalg.norm(vec_1, ord=2) * np.linalg.norm(vec_2, ord=2)) == 0:
                similarity += 0
            else:
                similarity += np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1, ord=2) * np.linalg.norm(vec_2, ord=2))
    try:
        similarity = weight - (similarity / (len(aim_vector[:-1]) * len(cand_vector[:-1])))
    except ZeroDivisionError:
        if len(aim_vector[:-1]) == 0 and len(cand_vector[:-1]) == 0:
            similarity = weight - 1
        else:
            similarity = weight
    return similarity


def get_mappings(score_mat, path_list_1, label_list_1, path_list_2, label_list_2):
    try:
        matching_pairs = scipy.optimize.linear_sum_assignment(score_mat)
        for ind in range(len(matching_pairs[0])):
            ind_1 = matching_pairs[0][ind]
            ind_2 = matching_pairs[1][ind]
            aim_list = path_list_1[ind_1] + [label_list_1[ind_1]]  # [label_list_1[ind_1]]  #
            selected_list = path_list_2[ind_2] + [label_list_2[ind_2]]  # [label_list_2[ind_2]]  #
            print("%s : \n%s\n" % (
                (",".join(["_".join(l) for l in aim_list])), (",".join(["_".join(l) for l in selected_list]))))
        return
    except ValueError:
        return


def get_score_matrix(score_mat, label_list_1, path_list_1, label_list_2, path_list_2, wv):
    for j in range(len(label_list_1)):
        score_list = []

        aim_list = path_list_1[j] + [label_list_1[j]]  # [label_list_1[j]]  #
        aim_vector = []
        for aim in aim_list:
            aim_vector.append(word_embed(aim, wv))

        for k in range(len(label_list_2)):
            candidate_list = path_list_2[k] + [label_list_2[k]]  # [label_list_2[k]]  #
            cand_vector = []
            for cand in candidate_list:
                cand_vector.append(word_embed(cand, wv))

            weight = 10

            score = score_attribute_name(aim_list, candidate_list, aim_vector, cand_vector, weight)
            similarity = score_hierarchy(aim_vector, cand_vector, weight)

            score_list.append(score + similarity)

        score_mat.append(score_list)
    return


def map_attributes(class_info, num_clusters):
    """ This method is for getting mappings"""
    with open('W2V_models/w2v_dict.json') as f:
        wv = json.load(f)

    for i in range(num_clusters):
        class_name = "Class_" + str(i)

        label_list_1 = class_info[class_name][0]
        path_list_1 = class_info[class_name][1]
        label_list_2 = class_info[class_name][2]
        path_list_2 = class_info[class_name][3]

        score_mat = []
        get_score_matrix(score_mat, label_list_1, path_list_1, label_list_2, path_list_2, wv)
        get_mappings(score_mat, path_list_1, label_list_1, path_list_2, label_list_2)
    return


def train_model(file_name, num_clusters):
    dirname = os.getcwd()
    abspath = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(abspath, file_name)
    files = []

    for r, d, f in os.walk(path):
        for file in f:
            if '.json' in file:
                files.append(os.path.join(r, file))

    ''' Getting input data '''
    tag_list = []
    feature_list = []

    with open('W2V_models/w2v_dict.json') as f:
        wv = json.load(f)

    i = 0
    for f in files:
        print(i)
        i = i + 1
        print(f)
        with open(f) as g:
            distros_dict = json.load(g)

        list_att = []
        info = {"names": [], "paths": [], "values": []}
        dict_traverse(distros_dict, list_att, info)

        out_feat = get_features(info, wv)
        tag_list = tag_list + out_feat[0]
        feature_list = feature_list + out_feat[1]

    ''' Hierarchical Clustering '''
    out_classes = hierarchical_clustering(feature_list, tag_list, num_clusters, plot=0)

    ''' Training '''
    xg_train(feature_list, out_classes, num_clusters)
    return


def test_model(file_name_1, file_name_2, num_clusters):
    dirname = os.getcwd()
    abspath = os.path.dirname(os.path.abspath(__file__))
    path_1 = os.path.join(abspath, file_name_1)
    path_2 = os.path.join(abspath, file_name_2)

    ''' Read input data for testing/predicting '''
    predictions_1 = predict(path_1)
    predictions_2 = predict(path_2)

    ''' Get classifications '''
    class_info = classifications(num_clusters, predictions_1, predictions_2)

    ''' Get mappings '''
    map_attributes(class_info, num_clusters)
    return


def main():
    """ Main method"""
    ''' Parameters'''
    num_clusters = 20

    ''' Train model '''
    # file_name = "Datasets/Connector_schemas/"
    # train_model(file_name, num_clusters)

    ''' Test model '''
    test_schema_1 = 'Datasets/Test/test_input_1.json'
    test_schema_2 = 'Datasets/Test/test_output_1.json'

    # test_schema_1 = Datasets/Test/in_15.json
    # test_schema_2 = Datasets/Test/out_15.json

    test_model(test_schema_1, test_schema_2, num_clusters)



if __name__ == "__main__":
    main()

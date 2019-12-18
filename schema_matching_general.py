import os
import operator
import smart_open
import cmath as math
import re

import xml.dom.minidom
import xml.etree.ElementTree as ET

from scipy.stats import variation
import scipy.cluster.hierarchy as shc
import scipy.optimize

import statistics
import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model

from sklearn.cluster import AgglomerativeClustering

import xgboost as xgb
from xgboost import plot_importance
import joblib

from gensim.models import KeyedVectors


def get_elem(tag_name, doc):
    list_val = []
    tag_property = doc.getElementsByTagName(tag_name)
    for skill in tag_property:
        try:
            list_val.append(skill.firstChild.nodeValue)
        except AttributeError:
            continue
    return list_val


def indices(list_in, value):
    new_list = []
    for i in range(len(list_in)):
        if list_in[i] == value:
            new_list.append(i)
    return new_list


def range_extract(list):
    dif_values = {}
    for word in list:
        for letter in word:
            if letter in dif_values.keys():
                dif_values[letter] += 1
            else:
                dif_values[letter] = 1
    sorted_diff = sorted(dif_values.items(), key=operator.itemgetter(1), reverse=1)
    # print (sorted_diff)
    return sorted_diff


def predict_char2vec(word, embedding_model, char_dic, max_review_length):
    feature = []
    feature.append(get_features_char2vec(word, max_review_length, char_dic))

    features = np.asarray(feature)
    prediction = embedding_model.predict(features)
    return prediction


def get_features_char2vec(word, max_review_length, char_dic):
    chars = list(word)
    features = []

    for char in chars:
        try:
            char2vec = char_dic[char]
            features.append(char2vec)
        except KeyError:
            continue

    # Zero padding
    features_2 = []
    if len(features) < max_review_length:
        for num in range(max_review_length - len(features)):
            features_2.append(0)
    features = features_2 + features
    return features


def create_dic():
    alphanum = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=`~[]{}|;:,.<>/? ')
    char_dic = {}
    i = 1
    for char in alphanum:
        char_dic[char] = i
        i += 1
    # print(char_dic)
    return char_dic


def get_instances(tag_list, doc, obj):
    for tag_name in tag_list:
        list_val = []
        tag_property = doc.getElementsByTagName(tag_name)
        for skill in tag_property:
            try:
                list_val.append(skill.firstChild.nodeValue)
            except AttributeError:
                continue
        obj['list_' + tag_name] = list_val
    return


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


def get_stat_features(processed_list):
    try:
        stat_feat = [max(processed_list), min(processed_list), sum(processed_list) / len(processed_list),
                     variation(processed_list), statistics.stdev(processed_list)]
    except ValueError:
        stat_feat = [0, 0, 0, 0, 0]
    return stat_feat


def get_comp_features(digit_list, alpha_list, chara_list):
    total_char = len(digit_list) + len(alpha_list) + len(chara_list)
    try:
        comp_feat = [len(digit_list) / total_char, len(alpha_list) / total_char, len(chara_list) / total_char]
    except ZeroDivisionError:
        comp_feat = [0, 0, 0]
    return comp_feat


def get_dist_features(distinct_val):
    total_items = sum([distinct_val[key] for key in distinct_val.keys()])
    distr_list = [distinct_val[key] / total_items for key in distinct_val.keys()]
    distr_list_mod = [(i * math.log(i, 10)).real for i in distr_list]
    dist_feat = [len(distinct_val.keys()), -sum(distr_list_mod)]
    return dist_feat


def get_type_feature(tag_name):
    type_list = ['phone', 'name', 'address', 'city', 'email', 'id', 'age', 'number']
    norm_type_list = [0 * num for num in range(len(type_list))]
    for val in range(len(type_list)):
        if type_list[val] in tag_name.lower():
            norm_type_list[val] = 1
    return norm_type_list


def get_class_features(model, char_dic, max_review_length, num_iter, orig_list):
    type_list = []
    norm_type_list = []
    j = 0
    while j < num_iter:
        j += 1
        try:
            name_list_2 = orig_list[j].split()
            for word in name_list_2:
                type_list.append(predict_char2vec(word, model, char_dic, max_review_length).tolist()[0])
            if len(type_list) > 0:
                type_list_mean = np.mean(type_list, axis=0).tolist()
            else:
                type_list_mean = type_list.tolist()
            if type(type_list_mean) == list:
                norm_type_list.append(type_list_mean)
            else:
                continue
        except AttributeError:
            continue
        except IndexError:
            break
    if len(norm_type_list) > 0:
        norm_type_list = np.mean(norm_type_list, axis=0).tolist()
    return norm_type_list


def get_embed_feature(tag_name, wv):
    temp = []
    tag_name_list = list(filter(''.__ne__, (
        re.split('[_ :]', re.sub(r"([A-Z]+[a-z0-9_\W])", r" \1", tag_name + "_").lower()))))
    for tags in tag_name_list:
        try:
            word2_vec_list = wv[tags.lower()].tolist()
        except KeyError:
            word2_vec_list = [0 * count for count in range(15)]
        temp.append(word2_vec_list)
    if len(temp) > 0:
        word2_vec_list = np.mean(temp, axis=0).tolist()
    else:
        word2_vec_list = temp
    return word2_vec_list


def character_count(orig_list, processed_list, digit_list, alpha_list, chara_list, distinct_val):
    space = 0
    new_line = 0
    for item in orig_list:
        try:
            if item.isdigit():
                processed_list.append(int(item))
            elif type(item) == str:  # ##################### DOUBLE CHECK
                processed_list.append(len(item))
            else:
                processed_list.append(item)
            for letter in str(item):
                if letter.isdigit():
                    digit_list.append(letter)
                elif letter.isalpha():
                    alpha_list.append(letter)
                elif letter.isspace():
                    space += 1
                else:
                    chara_list.append(letter)
            if item == '\n' or item.isspace():
                new_line += 1
            else:
                if item in distinct_val.keys():
                    distinct_val[item] += 1
                else:
                    distinct_val[item] = 1
        except AttributeError:
            continue
    return


def get_features(tag_list, doc, num_iter):
    wv = KeyedVectors.load("W2V_models/default_2", mmap='r')
    model = load_model("Categorical_classifier_models/Categorical_classifier_embedd.h5")
    char_dic = create_dic()

    obj = {}
    final_feature_list_concat = []
    final_feature_list = []
    new_tag_list = []
    max_review_length = 50

    ''' Read instances for each atrribute '''
    get_instances(tag_list, doc, obj)

    ''' Get feature '''
    for tag_name in tag_list:
        name = 'list_' + tag_name
        orig_list = obj[name]

        ''' Count characters '''
        distinct_val = {}
        processed_list = []
        digit_list = []
        alpha_list = []
        chara_list = []
        character_count(orig_list, processed_list, digit_list, alpha_list, chara_list, distinct_val)

        if not processed_list:
            continue
        elif len(distinct_val) == 0:
            continue
        else:
            new_tag_list.append(tag_name)
            ''' Merge all the features'''
            norm_feature_map = normalize(get_stat_features(processed_list)) + \
                               normalize(get_comp_features(digit_list, alpha_list, chara_list)) + \
                               normalize(get_dist_features(distinct_val)) + normalize(get_embed_feature(tag_name, wv)) + \
                               get_type_feature(tag_name) + get_class_features(model, char_dic, max_review_length,
                                                                               num_iter, orig_list)

            final_feature_list.append(norm_feature_map)
            final_feature_list_concat = final_feature_list_concat + norm_feature_map

    return [final_feature_list_concat, final_feature_list, new_tag_list, len(norm_feature_map)]


def get_info(info, new_words, num_iter, data_paths):
    with smart_open.open(data_paths, "r") as f:
        fl = f.readlines()

    for line in fl:
        line = line[:-1]
        print(line)
        info_dict = {}
        xml_tree = ET.parse(line)  # ### Define line
        # parser = ET.XMLParser(encoding="utf-8")
        # xml_tree = ET.fromstring(line, parser=parser)
        elem_list = []
        for elem in xml_tree.iter():
            elem_list.append(elem.tag)

        elem_list = list(set(elem_list))
        doc = xml.dom.minidom.parse(line)
        features_tag = get_features(elem_list, doc, num_iter)
        info_dict['elem_list'] = elem_list
        info_dict['doc'] = doc
        info_dict['features_tag'] = features_tag

        info.append(info_dict)
    return


def extract_features(info):
    features = []
    label_list = []
    for i in range(len(info)):
        print("I")
        print(i)
        features = features + info[i]['features_tag'][1]
        label_list = label_list + info[i]['features_tag'][2]
    return [features, label_list]


def hierarchical_clustering(features, label_list, num_clusters, plot):
    if plot:
        plt.figure(figsize=(10, 7))
        plt.title("Clusters")
        get_link = shc.linkage(features, method='ward')
        # print(label_list)
        dend = shc.dendrogram(get_link, leaf_font_size=8, leaf_rotation=90., labels=label_list)
        plt.axhline(y=1, color='r', linestyle='--')
        plt.show()

    cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    out_classes = cluster.fit_predict(features)
    return out_classes


def xg_train(batch_x, batch_y, num_class):
    d_train = xgb.DMatrix(batch_x, label=batch_y)
    param = {
        'eta': 0.3,
        'max_depth': 10,  # 3,
        'objective': 'multi:softprob',
        'num_class': num_class}
    steps = 1000  # The number of training iterations
    model = xgb.train(param, d_train, steps)

    ''' Get feature importance '''
    print("Feature Importance:")
    print("Gain:")
    print(model.get_score(importance_type='gain'))
    plot_importance(model)

    ''' Save model'''
    joblib.dump(model, "Schema_matching_models/Schema_matching_1_test")
    return


def xg_feed_forward(test_feature):
    model = joblib.load("Schema_matching_models/Schema_matching_1")
    d_test = xgb.DMatrix(test_feature)
    preds = model.predict(d_test)
    return preds


def filter_class(preds, prediction, test_tag, test_tag_1):
    for num in range(len(preds)):
        classes = preds[num]
        if max(classes) > 0.9:
            prediction[0].append(classes.argmax())
            test_tag_1.append(test_tag[num])
    return


def predict(test_feature_1, test_feature_2, test_tag_1, test_tag_2):
    preds_1 = xg_feed_forward(test_feature_1)
    preds_2 = xg_feed_forward(test_feature_2)

    prediction = [preds_1.argmax(1), preds_2.argmax(1)]

    ''' Print prediction '''
    print("\n")
    print("Schema 1:")
    for i in range(len(test_feature_1)):
        print("%s  : %u" % (test_tag_1[i], prediction[0][i]))

    print("\n")
    print("Schema 2:")
    for i in range(len(test_feature_2)):
        print("%s  : %u" % (test_tag_2[i], prediction[1][i]))
    return prediction


def print_class(index_list, test_tag, label_list, test_doc, range_n, range_list):
    for item in index_list:
        label = test_tag[item]
        print(label)
        label_list.append(label)
        temp = range_extract(get_elem(label, test_doc))
        if len(temp) > range_n:
            range_list.append(temp[:range_n])
        else:
            range_list.append(temp)
    print("___________")
    return


def get_score_1(range_list_1, range_list_2, range_n):
    """ Method 1 : Use character frequency """
    score_list = []
    for i_attr in range_list_1:
        score_row = []
        i_attributes = [x[0] for x in i_attr]
        for j_attr in range_list_2:
            j_attributes = [y[0] for y in j_attr]
            score = 0
            for i_letter in i_attributes:
                if i_letter in j_attributes:
                    score += 1
            score_row.append(5 - score * range_n / len(i_attributes))
        score_list.append(score_row)
    return score_list


def get_score_2(range_list_1, range_list_2, ):
    """ Method 2 : Use character frequency and consider their reletive positions """
    cost_list = []
    for i_attr in range_list_1:
        cost_row = []
        i_attributes = [x[0] for x in i_attr]
        for j_attr in range_list_2:
            j_attributes = [y[0] for y in j_attr]
            cost = 0
            for i_letter in i_attributes:
                if i_letter in j_attributes:
                    cost += abs(i_attributes.index(i_letter) - j_attributes.index(i_letter))
                else:
                    cost += 6
            cost_row.append(cost)
        cost_list.append(cost_row)
    return cost_list


def get_score_3(label_list_1, label_list_2, wv, wv_1):
    """ Method 3 : Use word embeddings """
    simi_list = []
    ###########################################
    # simi_list = []
    # for i_attr in label_list_1:
    #     i_attr_list = i_attr.split("_")
    #     # print(i_attr_list)
    #     simi_row = []
    #     for j_attr in label_list_2:
    #         j_attr_list = j_attr.split("_")
    #         # print(j_attr_list)
    #         temp_list = []
    #         for word_part_1 in i_attr_list:
    #             for word_part_2 in j_attr_list:
    #                 if word_part_1.lower() == word_part_2.lower():
    #                     simi = 1
    #                 else:
    #                     try:
    #                         simi = wv.similarity(w1=word_part_1, w2=word_part_2)
    #                     except KeyError:
    #                         # simi = 0
    #                         # print("###########")
    #                         # print(word_part_1)
    #                         # print(word_part_2)
    #                         # print("###########")
    #                         try:
    #                             simi = wv_1.similarity(w1=word_part_1, w2=word_part_2)
    #                         except KeyError:
    #                             simi = 0
    #                 temp_list.append(simi)
    #                 # print("%s , %s, %f" % (word_part_1, word_part_2, simi))
    #         simi_row.append(1-max(temp_list))
    #     simi_list.append(simi_row)
    # # print (cost_list)
    #############################################
    for i_attr in label_list_1:
        i_attr_list = list(filter(''.__ne__, (re.split('[_ :]', re.sub(r"([A-Z0-9]+[a-z0-9_\W])", r" \1", i_attr + "_").lower()))))  # i_attr.split("_")
        simi_row = []
        for j_attr in label_list_2:
            j_attr_list = list(filter(''.__ne__, (re.split('[_ :]', re.sub(r"([A-Z0-9]+[a-z0-9_\W])", r" \1", j_attr + "_").lower()))))  # j_attr.split("_")
            # print(j_attr_list)
            temp_list = []
            for word_part_1 in i_attr_list:
                for word_part_2 in j_attr_list:
                    if word_part_1.lower() == word_part_2.lower():
                        simi = 1
                    else:
                        try:
                            simi = wv.similarity(w1=word_part_1, w2=word_part_2)
                        except KeyError:
                            try:
                                simi = wv_1.similarity(w1=word_part_1, w2=word_part_2)
                            except KeyError:
                                print("__________")
                                print("KEY ERROR")
                                print(word_part_1)
                                print(word_part_2)
                                print("__________")
                                simi = 0
                    temp_list.append(simi)
            simi_row.append(1 - sum(temp_list) / len(temp_list))
        simi_list.append(simi_row)
    return simi_list


def get_mapping(score_list, label_list_1, label_list_2):
    mapping_list = []
    try:
        matching_pairs = scipy.optimize.linear_sum_assignment(score_list)

        for ind in range(len(matching_pairs[0])):
            cost = score_list[matching_pairs[0][ind]][matching_pairs[1][ind]]
            if cost < 1: #.7:
                print('%s : %s' % (label_list_1[matching_pairs[0][ind]], label_list_2[matching_pairs[1][ind]])),
                mapping_list. append([label_list_1[matching_pairs[0][ind]], label_list_2[matching_pairs[1][ind]]])
            else:
                print(cost)
        return mapping_list
    except ValueError:
        return mapping_list


def mapping(num_clusters, prediction, test_tag_1, test_tag_2, test_doc_1, test_doc_2):
    print('\n')
    range_n = 5
    wv = KeyedVectors.load("W2V_models/default", mmap='r')
    wv_1 = KeyedVectors.load("W2V_models/word2vec_vectors", mmap='r')
    mapping_list = []
    for i in range(num_clusters):
        range_list_1 = []
        range_list_2 = []
        label_list_1 = []
        label_list_2 = []

        index_list_1 = indices(prediction[0], i)
        index_list_2 = indices(prediction[1], i)

        ''' Print classification '''
        print("\n")
        print("Class %u" % (i + 1))
        print_class(index_list_1, test_tag_1, label_list_1, test_doc_1, range_n, range_list_1)
        print_class(index_list_2, test_tag_2, label_list_2, test_doc_2, range_n, range_list_2)

        ''' Construct score list '''
        score_list = get_score_3(label_list_1, label_list_2, wv, wv_1)

        ''' Find and print mapping'''
        mapping_list = mapping_list + get_mapping(score_list, label_list_1, label_list_2)
    return mapping_list


def train_model(num_iter, data_paths_train, num_clusters):
    """ Training method """
    ''' Fetch data and extract features '''
    info = []
    new_words = {}
    get_info(info, new_words, num_iter, data_paths_train)

    ''' Features for training and clustering '''
    extraction = extract_features(info)
    features = extraction[0]
    label_list = extraction[1]

    ''' Hierarchical clustering '''
    out_classes = hierarchical_clustering(features, label_list, num_clusters, plot=0)

    ''' Train '''
    # Neural Network
    # prediction = nn(features, out, test_feature_1, test_feature_2, num_clusters, len(features), num_feat)

    # RF
    # prediction = rf(features, out_classes, test_feature_1, test_feature_2, num_clusters, num_feat)

    # XG Boost
    xg_train(features, out_classes, num_clusters)
    return


def test_model(num_iter, data_paths_test, num_clusters):
    """ Testing method"""
    ''' Test data set '''
    test_info = []
    test_new_words = {}
    get_info(test_info, test_new_words, num_iter, data_paths_test)

    test_feature_1 = test_info[0]['features_tag'][1]
    test_tag_1 = test_info[0]['features_tag'][2]
    test_doc_1 = test_info[0]['doc']

    test_feature_2 = test_info[1]['features_tag'][1]
    test_tag_2 = test_info[1]['features_tag'][2]
    test_doc_2 = test_info[1]['doc']

    ''' Prediction '''
    prediction = predict(test_feature_1, test_feature_2, test_tag_1, test_tag_2)

    ''' Match schemas '''
    mapping_list = mapping(num_clusters, prediction, test_tag_1, test_tag_2, test_doc_1, test_doc_2)
    return mapping_list


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    data_paths_train = "Datasets/courses_schemas_train.txt"
    data_paths_test = "Datasets/courses_schemas_test.txt"
    num_instance = 10  # Number of instances that are considered for get category from categorical classifier
    num_clusters = 13

    ''' Training '''
    # train_model(num_instance, data_paths_train, num_clusters)

    ''' Testing '''
    mapping_list = test_model(num_instance, data_paths_test, num_clusters)
    return mapping_list


if __name__ == "__main__":
    main()

import xml.dom.minidom
from scipy.stats import variation
import statistics
import math
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
# import pylab
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cluster import AgglomerativeClustering
import xml.etree.ElementTree as ET
import scipy.optimize
# from operator import add


import cmath as math
import operator
import sys


def get_elem(tag_name, doc):
    list_val = []
    tag_property = doc.getElementsByTagName(tag_name)
    for skill in tag_property:
        try:
            list_val.append(skill.firstChild.nodeValue)
        except AttributeError:
            continue
    return list_val


def get_features(tag_list, doc):
    new_tag_list = []
    obj = {}
    example_list = []
    for tag_name in tag_list:
        list_val = []
        tag_property = doc.getElementsByTagName(tag_name)
        # print ("%d %s:" % (tag_property.length, tag_name))
        for skill in tag_property:
            try:
                list_val.append(skill.firstChild.nodeValue)
            except AttributeError:
                continue
        obj['list_' + tag_name] = list_val
        # print (list_val)
    final_feature_list = []
    final_feature_list_concat = []
    for tag_name in tag_list:
        name = 'list_' + tag_name
        orig_list = obj[name]
        try:
            example_list.append(tag_name + '_' + orig_list[0])
        except (TypeError, IndexError):
            example_list.append(tag_name + '_' + 'NONE')

        processed_list = []
        digit_list = []
        alpha_list = []
        chara_list = []
        space = 0
        distinct_val = {}
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
                    # print ("No")
                    new_line += 1
                else:
                    if item in distinct_val.keys():
                        distinct_val[item] += 1
                    else:
                        distinct_val[item] = 1
            except AttributeError:
                continue

        if processed_list == []:
            continue
        elif len(distinct_val)==0:
            # print ("DISTINCT")
            continue
        else:
            new_tag_list.append(tag_name)

            total_items = sum([distinct_val[key] for key in distinct_val.keys()])
            distr_list = [distinct_val[key]/total_items for key in distinct_val.keys()]
            distr_list_mod = [(i*math.log(i, 10)).real for i in distr_list]

            total_char = len(digit_list) + len(alpha_list) + len(chara_list)
            try:
                stat_feat = [max(processed_list), min(processed_list), sum(processed_list)/len(processed_list),
                             variation(processed_list), statistics.stdev(processed_list)]
            except ValueError:
                stat_feat = [0, 0, 0, 0, 0]
            try:
                comp_feat = [len(digit_list)/total_char, len(alpha_list)/total_char, len(chara_list)/total_char]
            except ZeroDivisionError:
                comp_feat = [0, 0, 0]
            dist_feat = [len(distinct_val.keys()), -sum(distr_list_mod)]

            norm_stat_feat = []
            for val in stat_feat:
                norm_stat_feat.append(2 * (1 / (1 + 1.01 ** (val * -1)) - 0.5))

            norm_dist_feat = []
            for val in dist_feat:
                norm_dist_feat.append(2 * (1 / (1 + 1.01 ** (val * -1)) - 0.5))

            # feature_map = stat_feat + comp_feat + dist_feat
            # print ("START")
            # print (norm_stat_feat)
            # print (comp_feat)
            # print (dist_feat)
            norm_feature_map = norm_stat_feat + comp_feat + norm_dist_feat

            final_feature_list.append(norm_feature_map)
            final_feature_list_concat = final_feature_list_concat + norm_feature_map
            # print (final_feature_list)
            # print (norm_feature_map)
    # print (example_list)
    return [final_feature_list_concat, final_feature_list, new_tag_list]


def nn(batch_x, batch_y, batch_test_x_1, batch_test_x_2, num_class, num_vec):
    # Python optimisation variables
    learning_rate = 0.9  # 0.01  # 0.5
    epochs = 1000  # 5000
    # batch_size = 6   # 100

    x = tf.compat.v1.placeholder(tf.float32, [None, 10])
    y = tf.compat.v1.placeholder(tf.float32, [None, num_class])

    w1 = tf.Variable(tf.random.normal([10, 15], stddev=0.03), name='w1')
    b1 = tf.Variable(tf.random.normal([15]), name='b1')
    w2 = tf.Variable(tf.random.normal([15, num_class], stddev=0.03), name='w2')
    b2 = tf.Variable(tf.random.normal([num_class]), name='b2')

    hidden_out = tf.add(tf.matmul(x, w1), b1)
    # hidden_out = tf.nn.relu(hidden_out)
    hidden_out = tf.math.sigmoid(hidden_out)

    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))
    # y_ = tf.math.log_sigmoid(tf.add(tf.matmul(hidden_out, w2), b2))

    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.math.log(y_clipped)
                                                  + (1 - y) * tf.math.log(1 - y_clipped), axis=1))

    optimiser = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    init_op = tf.compat.v1.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cost_list = []
    epoch_list = []
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        total_batch = num_vec
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                _, c = sess.run([optimiser, cross_entropy],
                                feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
            cost_list.append(avg_cost)
            epoch_list.append(epoch+1)
        # print(sess.run(accuracy, feed_dict={x: batch_test_x, y: batch_test_y}))
        out_list_1 = sess.run(y_, feed_dict={x: batch_test_x_1})
        class_list_1 = []
        for li in out_list_1:
            li_round = ['%.2f' % elem for elem in li]
            # print (li_round)
            class_list_1.append(li_round.index(max(li_round)) + 1)
        out_list_2 = sess.run(y_, feed_dict={x: batch_test_x_2})
        class_list_2 = []
        for li in out_list_2:
            li_round = ['%.2f' % elem for elem in li]
            # print (li_round)
            class_list_2.append(li_round.index(max(li_round)) + 1)
        # print (class_list)
        plt.plot(epoch_list, cost_list)
        # plt.show()
    return [class_list_1, class_list_2]


def indices(list_in, value):
    new_list = []
    for i in range(len(list_in)):
        if list_in[i]==value:
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


def main():

    ''' Fetch data and extract features '''

    # f = open("Datasets/phone_schema.txt", "r")
    f = open("Datasets/courses_schemas.txt", "r")
    # f = open("Datasets/real_es_schema.txt", "r")
    # f = open("Datasets/schemas.txt", "r")
    fl = f.readlines()
    info = []
    for line in fl:
        line = line[:-1]
        print (line)
        info_dict = {}
        xml_tree = ET.parse(line)  # ### Define line
        elem_list = []
        for elem in xml_tree.iter():
            elem_list.append(elem.tag)

        elem_list = list(set(elem_list))
        doc = xml.dom.minidom.parse(line)
        features_tag = get_features(elem_list, doc)

        info_dict['elem_list'] = elem_list
        info_dict['doc'] = doc
        info_dict['features_tag'] = features_tag

        info.append(info_dict)

    print (len(info))

    ''' Features for training and clustering '''
    features = []
    label_list = []

    for i in range(len(info)):
        features = features + info[i]['features_tag'][1]
        label_list = label_list + info[i]['features_tag'][2]

    ''' Hierarchical clustering '''
    plt.figure(figsize=(10, 7))
    plt.title("Clusters")
    get_link = shc.linkage(features, method='ward')
    dend = shc.dendrogram(get_link, leaf_font_size=8., leaf_rotation=90., labels=label_list)
    plt.axhline(y=1, color='r', linestyle='--')
    # pylab.show()

    ''' Ground truth for training '''
    num_clusters = 13  # 6  # 8 #13
    cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    out_classes = cluster.fit_predict(features)
    # print (label_list)
    # print (out_classes)

    out = []
    for class_name in out_classes:
        one_hot_vec = [0 * i for i in range(class_name)] + [1] + [0 * i for i in range(num_clusters - class_name - 1)]
        # print (one_hot_vec)
        out.append(one_hot_vec)

    ''' Test data set '''
    test_set_1 = 3
    test_set_2 = 4
    test_feature_1 = info[test_set_1-1]['features_tag'][1]  # features_tag_1[1]
    test_tag_1 = info[test_set_1-1]['features_tag'][2]  # features_tag_1[2]
    test_doc_1 = info[test_set_1-1]['doc']  # doc1

    test_feature_2 = info[test_set_2-1]['features_tag'][1]  # features_tag_4[1]
    test_tag_2 = info[test_set_2-1]['features_tag'][2]  # features_tag_4[2]
    test_doc_2 = info[test_set_2-1]['doc']  # doc4

    ''' Prediction '''
    prediction = nn(features, out, test_feature_1, test_feature_2, num_clusters, len(features))

    print ("\n")
    print ("Schema 1:")

    for i in range(len(test_feature_1)):
        print ("%s  : %u" % (test_tag_1[i], prediction[0][i]))

    print ("\n")
    print ("Schema 2:")

    for i in range(len(test_feature_2)):
        print ("%s  : %u" % (test_tag_2[i], prediction[1][i]))

    ''' Match schemas '''
    print ('\n')
    range_n = 5

    for i in range(num_clusters):
        range_list_1 = []
        range_list_2 = []
        label_list_1 = []
        label_list_2 = []
        index_list_1 = indices(prediction[0], i+1)
        index_list_2 = indices(prediction[1], i+1)

        print ("\n")
        print ("Class %u" % (i + 1))
        for item in index_list_1:
            label = test_tag_1[item]
            print (label)
            label_list_1.append(label)
            temp = range_extract(get_elem(label, test_doc_1))
            # print (temp)
            if len(temp)>range_n:
                range_list_1.append(temp[:range_n])
            else:
                range_list_1.append(temp)
        print ("___________")
        for item in index_list_2:
            label = test_tag_2[item]
            print (label)
            label_list_2.append(label)
            temp = range_extract(get_elem(label, test_doc_2))
            # print (temp)
            if len(temp)>range_n:
                range_list_2.append(temp[:range_n])
            else:
                range_list_2.append(temp)
        print ("_______________________________")
        # print (range_list_1)
        # print ("\n")
        # print (range_list_2)
        # print ("\n")

        score_list = []

        for i_attr in range_list_1:
            # print ('\n')
            score_row = []
            i_attributes = [x[0] for x in i_attr]
            for j_attr in range_list_2:
                j_attributes = [y[0] for y in j_attr]
                score = 0
                for i_letter in i_attributes:
                    if i_letter in j_attributes:
                        score += 1
                # print (score)
                # print (len(i_attributes))
                score_row.append(5 - score*range_n/len(i_attributes))
            score_list.append(score_row)
        # print (score_list)

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
                # print (score)
                # print (len(i_attributes))
                cost_row.append(cost)
            cost_list.append(cost_row)
        # print (cost_list)

        try:
            matching_pairs = scipy.optimize.linear_sum_assignment(score_list)
        except ValueError:
            continue

        try:
            matching_pairs_2 = scipy.optimize.linear_sum_assignment(cost_list)
        except ValueError:
            continue

        # print ("\nPAIR")
        # print (matching_pairs)
        # print ("\n")
        # print ("SCORE PAIRS")
        # for ind in range(len(matching_pairs[0])):
        #     print ('%s : %s' % (label_list_1[matching_pairs[0][ind]], label_list_2[matching_pairs[1][ind]]))
        #
        # print ("COST PAIRS")
        for ind in range(len(matching_pairs_2[0])):
            print ('%s : %s' % (label_list_1[matching_pairs_2[0][ind]], label_list_2[matching_pairs_2[1][ind]]))

    # plt.show()


if __name__ == "__main__":
    main()

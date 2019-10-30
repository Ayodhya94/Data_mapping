import re
import xgboost as xgb
import joblib
from keras.models import Sequential
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import chars2vec
from keras import optimizers
from keras.models import load_model
# fix random seed for reproducibility
np.random.seed(7)


def train_categories_stat(batch_x, batch_y, num_class):
    d_train = xgb.DMatrix(batch_x, label=batch_y)
    param = {
        'eta': 0.3,
        'max_depth': 3,
        'objective': 'multi:softprob',
        'num_class': num_class}
    steps = 1000  # The number of training iterations

    model = xgb.train(param, d_train, steps)
    joblib.dump(model, "Categorical_classifier_model")
    return


def train_categories_char2vec(x, y, num_class, tag_list):
    x_train = np.asarray(x)
    y_train = np.asarray(y)

    '''create the model'''
    model = Sequential()
    model.add(LSTM(50))
    model.add(Dense(num_class, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    '''Train model'''
    print("Start training...")
    model.fit(np.asarray(x_train), y_train, epochs=150, batch_size=128, shuffle=True,)
    print("Trained")

    '''Save model'''
    model.save("Categorical_classifier_models/Categorical_classifier_test.h5")
    print("Model saved to Categorical_classifier_test.h5")
    return


def train_categories_embedding(x, y, num_class, tag_list):
    x_train = np.asarray(x)
    y_train = np.asarray(y)

    '''create the model'''
    model = Sequential()
    model.add(Embedding(93, 50, input_length=None))
    model.add(LSTM(10))
    model.add(Dense(num_class, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    '''Train model'''
    print("Start training...")
    model.fit(np.asarray(x_train), y_train, epochs=150, batch_size=32, shuffle=True,)
    print("Trained")

    '''Save model'''
    model.save("Categorical_classifier_models/Categorical_classifier_embedd_test.h5")
    print("Model saved to Categorical_classifier_test.h5")
    return


def get_features_stat(word):
    char_list = list(word)
    num_char = len(char_list)
    num_alpha = len(re.findall("[A-Za-z]", word))
    num_nume = len(re.findall("[0-9]", word))
    num_space = len(re.findall(" ", word))
    num_at = len(re.findall("@", word))
    num_col = len(re.findall(":", word))
    num_per = len(re.findall(",", word))
    num_full = len(re.findall("[.]", word))
    num_special = len(re.findall("\W", word))

    feature_list = [num_char/1000, num_alpha/num_char, num_nume/num_char, num_space/num_char, num_at/num_char,
                    num_col/num_char, num_per/num_char, num_full/num_char, num_special/num_char]
    return feature_list


def get_features_char2vec(word, max_review_length, c2v_model):
    chars = list(word)
    features = []

    for char in chars:
        char2vec = c2v_model.vectorize_words([char])[0]
        norm_char2vec_list = []
        for val in char2vec:
            try:
                norm_char2vec_list.append(10 * (val - min(char2vec)) / (max(char2vec) - min(char2vec)))
            except ZeroDivisionError:
                norm_char2vec_list.append(0)
        features.append(norm_char2vec_list)

    # Zero padding
    features_2 = []
    if len(features) < max_review_length:
        for num in range(max_review_length - len(features)):
            features_2.append([0 * i for i in range(50)])
    features = features_2 + features
    return features


def get_features_embedding(word, max_review_length, char_dic):
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


def predict_char2vec(word, model, c2v_model, max_review_length):
    feature = []
    feature.append(get_features_char2vec(word, max_review_length, c2v_model))

    features = np.asarray(feature)
    prediction = model.predict(features)
    return prediction


def predict_embedding(word, embedding_model, char_dic, max_review_length):
    feature = []
    feature.append(get_features_embedding(word, max_review_length, char_dic))

    features = np.asarray(feature)
    prediction = embedding_model.predict(features)
    return prediction


def create_dic():
    alphanum = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=`~[]{}|;:,.<>/? ')
    char_dic = {}
    i = 1
    for char in alphanum:
        char_dic[char] = i
        i += 1
    print(char_dic)
    return char_dic


def test_model(embedding_model, char_dic, max_review_length):
    f = open("Datasets/category_test_month.txt", "r")
    classes = ["Email", "Phone", "Name", "Country", "street", "Time", "Month", "Day", "Gender", "Date", "Currency"]
    fl = f.readlines()
    i = 0
    for line in fl:
        i += 1
        print(i)
        prediction = predict_embedding(line, embedding_model, char_dic, max_review_length).tolist()[0]
        print("%s is a %s" % (line, classes[prediction.index(max(prediction))]))
    return


def train_model(f_train_in, f_train_out, max_review_length, c2v_model, num_class):
    x_batch = []
    y_batch = []

    fl = f_train_in.readlines()
    i = 0
    tag_list = []
    for ln in fl:
        i = i + 1
        # print(i)
        tag_list.append(ln)
        x_batch.append(get_features_char2vec(ln, max_review_length, c2v_model))
    print(len(x_batch))

    # Output data set
    fl = f_train_out.readlines()
    for ln in fl:
        out = [0*i for i in range(num_class)]
        out[int(ln)] = 1
        y_batch.append(out)
    print(len(y_batch))

    # Train model
    train_categories_char2vec(x_batch, y_batch, num_class, tag_list)
    return


def main():
    max_review_length = 50
    c2v_model = chars2vec.load_model('eng_50')
    num_class = 9
    char_dic = create_dic()
    f_train_in = open("Datasets/categ.txt", "r")
    f_train_out = open("Datasets/outs.txt", "r")

    ''' Training model'''
    train_model(f_train_in, f_train_out, max_review_length, c2v_model, num_class)

    '''Predict'''
    embedding_model = load_model("Categorical_classifier_models/Categorical_classifier_embedd.h5")
    test_model(embedding_model, char_dic, max_review_length)
    return


if __name__ == "__main__":
    main()

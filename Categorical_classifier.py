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
    # print("Feature Importance:")
    # print("Weight:")
    # print(model.get_score(importance_type='weight'))
    # print("Gain:")
    # print(model.get_score(importance_type='gain'))
    # print("Cover:")
    # print(model.get_score(importance_type='cover'))
    # print("Total gain:")
    # print(model.get_score(importance_type='total_gain'))
    # print("Total cover:")
    # print(model.get_score(importance_type='total_cover'))
    joblib.dump(model, "Categorical_classifier_model")
    return


def train_categories_char2vec(x, y):
    x_train = np.asarray(x)
    y_train = np.asarray(y)

    max_review_length = 40  # Length of sequence

    # x_train_padded = []

    # for entry in x_train:
    #     if len(entry) < max_review_length:
    #         for num in range(max_review_length-len(entry)):
    #             entry = np.append(entry, [[0*i for i in range(50)]], axis=0)
    #     x_train_padded.append(entry)

    # create the model
    model = Sequential()
    model.add(LSTM(10))
    model.add(Dense(2, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # # Print model summary
    # model.build((None, 40, 50))
    # print(model.summary())

    # Train model
    print("Start training...")
    model.fit(np.asarray(x_train), y_train, epochs=20, batch_size=32, shuffle=True,)
    print("Trained")

    # # Final evaluation of the model
    # scores = model.evaluate(np.asarray(x_test_padded), y_test, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1] * 100))

    # Save model
    model.save("Categorical_classifier.h5")
    print("Model saved to Categorical_classifier.h5")
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
    # c2v_model = chars2vec.load_model('eng_50')
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
    if len(features) < max_review_length:
        for num in range(max_review_length - len(features)):
            features.append([0 * i for i in range(50)])
    return features


def predict_char2vec(word, model, c2v_model, max_review_length):
    feature = []
    feature.append(get_features_char2vec(word, max_review_length, c2v_model))

    features = np.asarray(feature)
    prediction = model.predict(features)
    return prediction


def main():
    """ Classes
    0 - Name
    1 - Address
    2 - Phone
    3 - Email
    4 - Country
    5 - ID
    6 - Code

    Name
    City
    Country
    Phone
    Email


    """

    # Training data preparation
    x_batch = []
    y_batch = []

    max_review_length = 40

    f = open("Datasets/categ.txt", "r")
    fl = f.readlines()

    # Get features
    c2v_model = chars2vec.load_model('eng_50')
    i = 0
    for ln in fl:
        i = i + 1
        # print(i)
        x_batch.append(get_features_char2vec(ln, max_review_length, c2v_model))

    # Output data set
    f = open("Datasets/outs.txt", "r")
    fl = f.readlines()
    for ln in fl:
        if ln == "0\n":
            y_batch.append([1, 0])
            # y_batch.append([0])
        else:
            y_batch.append([0, 1])
            # y_batch.append([1])

    # Train model
    train_categories_char2vec(x_batch, y_batch)

    # Predict
    model = load_model("Categorical_classifier.h5")
    test = "ayodhyakinkini@gmaIL.COM"
    prediction = predict_char2vec(test, model, c2v_model, max_review_length).tolist()[0]
    print(prediction)
    print(max(prediction))
    classes = ["Email", "Phone"]
    print("%s is a %s" % (test, classes[prediction.index(max(prediction))]))

    return


if __name__ == "__main__":
    main()

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import load_files
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
import pandas as pd, numpy as np, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import csv
import datetime
import pickle

# 由于 train_data.txt 与 test.txt 的文件内容组织形式不一样，所以我们需要分别将其转化为统一的 csv 格式
def txt_to_csv_train(filePathSrc, filePathDst):
    list_data = []
    with open(filePathSrc, 'r', encoding='utf-8') as input_file, open(filePathDst, 'w', newline='') as output_file:
        for data in input_file:
            data = eval(data)
            list_data.append(data)
        keys = list_data[0].keys()
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_data)

def txt_to_csv_test(filePathSrc, filePathDst):
    with open(filePathSrc, 'r', encoding='utf-8') as input_file, open(filePathDst, 'w', newline='') as output_file:
        stripped = (line.strip('\n') for line in input_file)
        # 这里只需要用第一个逗号进行分割，因为一个句子中可能有很多个逗号
        lines = (line.split(', ', 1) for line in stripped if line)
        writer = csv.writer(output_file)
        writer.writerows(lines)


if __name__ == '__main__':
    txt_to_csv_train('train_data.txt', 'train_data.csv')
    txt_to_csv_test('test.txt', 'test.csv')

    dataset_train = pd.read_csv('train_data.csv')
    X = dataset_train.drop('label', axis=1)
    y = dataset_train['label']
    # X = X.to_numpy()
    X = X.values.tolist()
    # print(X)
    list1 = []
    for index_i in range(len(X)):
        for index_j in range(len(X[index_i])):
            list1.append(X[index_i][index_j])
    # print(list1)
    tv = TfidfVectorizer(stop_words='english')
    X_fit = tv.fit_transform(list1).toarray()
    print(X_fit.shape)

    # 数据集划分，进行模型的训练以及预测
    X_train, X_val, y_train, y_val = train_test_split(X_fit, y, test_size=0.2, random_state=42)
    # 使用支持向量机进行模型的训练
    # model = SVC(kernel='linear', C=1.0, random_state=42)
    # 使用 Logistic Regression 进行模型训练
    model = LogisticRegression(penalty='l2', C=10, random_state=42, max_iter=200)
    print('this model is running')
    starttime = datetime.datetime.now()
    model.fit(X_train, y_train)
    endtime = datetime.datetime.now()
    print('this model finishes running, running time:', (endtime - starttime).seconds, 'seconds.')
    score = model.score(X_val, y_val)
    print("Model score:", score)


    # 保存该 model, save the model as a pickle object in Python

    with open('text_classifier', 'wb') as picklefile:
        pickle.dump(model, picklefile)

    # 对测试集进行预测
    dataset_test = pd.read_csv('test.csv')
    X_test = dataset_test['text']
    X_test = X_test.values.tolist()
    # print(len(X_test))
    # list2 = []
    # for index_i in range(len(X_test)):
    #     list2.append(X_test[index_i])
    # tv_test = TfidfVectorizer(stop_words='english')

    X_test_fit = tv.transform(X_test).toarray()
    print(X_test_fit.shape)

    # 从本地加载训练好的模型
    # with open('text_classifier', 'rb') as training_model:
    #     model = pickle.load(training_model)
    y_pred = model.predict(X_test_fit)
    y_pred = np.expand_dims(y_pred, axis=1)
    print(y_pred)
    print(y_pred.shape)
    # print(type(model.predict(X_test_fit)))
    # X_test_np = np.array(X_test)
    X_test_id = dataset_test['id']
    X_test_id = X_test_id.to_numpy()
    X_test_id = np.expand_dims(X_test_id, axis=1)
    print(type(X_test_id))
    print(X_test_id.shape)
    result = np.concatenate((X_test_id, y_pred), axis=1)
    label = np.array([['label', 'pred']])
    result = np.concatenate((label, result), axis=0)
    # print(result.shape)
    np.savetxt('results_LR.txt', result, delimiter=', ', fmt='%s')

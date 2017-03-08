import numpy as np
from sklearn import svm
from sklearn import decomposition
from sklearn import neighbors
from sklearn.externals import joblib

import preprocessing


def unpack_data(data):
    src_names = np.array(map(lambda n: n[0], data))
    features = np.array(map(lambda n: n[1], data))
    labels = np.array(map(lambda n: n[2], data))
    return src_names, features, labels

def train_and_test(data, method, cv_fold=10):
    # extract features of every images
    fold_unit = len(data) / cv_fold
    np.random.shuffle(data)
    accu_rates = []
    models = []
    for fold in xrange(cv_fold):              ### only one trial for now
        print 'start fold:', fold
        train_data = data[:fold_unit*fold] + data[fold_unit*(fold+1):]
        test_data = data[fold_unit*fold:fold_unit*(fold+1)]
        model = train(train_data, method)
        print 'training done. start testing...'
        accu_rate = test(model, test_data, method)
        accu_rates.append(accu_rate)
        models.append(model)
    print accu_rates
    print 'average: ', np.average(accu_rates)
    # cache the best model
    best = models[np.argmax(accu_rates)]
    save_model(best)
    return models, np.average(accu_rates)


def train(data, method):
    src_names, features, labels = unpack_data(data)
    print 'train feature vector dim:', features.shape

    # initialize models (not all used)
    params = method[0]
    pca = decomposition.PCA(n_components=params['pca_n'])
    svc = svm.LinearSVC()
    knn = neighbors.KNeighborsClassifier(n_neighbors=params['knn_k'], metric=params['knn_metric'])

    if 'pca' in method:
        features = pca.fit_transform(features)

    if 'svc' in method:
        svc.fit(features, labels)

    if 'knn' in method:
        knn.fit(features, labels)

    return pca, svc, knn


def predict(model, features, method):
    pca, svc, knn = model
    params = method[0]
    if 'pca' in method:
        features = pca.transform(features)
    if 'svc' in method:
        predicted = svc.predict(features)
        return predicted
    if 'knn' in method:
        predicted = knn.predict(features)
        return predicted
    print 'error: no classification method specified'
    return []


def test(model, data, method):
    src_names, features, labels = unpack_data(data)
    predicted = predict(model, features, method)

    # get stats for accuracy
    test_size = src_names.shape[0]
    accuracy = (predicted == labels)
    accu_rate = np.sum(accuracy) / float(test_size)
    print np.sum(accuracy), 'correct out of', test_size
    print 'accuracy rate: ', accu_rate

    # write out all the wrongly-classified samples
    wrongs = np.array([src_names, labels, predicted])
    wrongs = np.transpose(wrongs)[np.invert(accuracy)]
    with open('last_wrong.txt', 'w') as log:
        for w in wrongs:
            log.write('{} truly {} classified {}\n'.format(w[0], w[1], w[2]))
    return accu_rate


def save_model(model):
    pca, svc, knn = model
    joblib.dump(svc, 'last_svc.model')
    return

def load_model(params):
    svc = joblib.load('last_svc.model')

    pca = decomposition.PCA(n_components=params['pca_n'])
    knn = neighbors.KNeighborsClassifier(n_neighbors=params['knn_k'], metric=params['knn_metric'])

    return pca, svc, knn


def main():
    data = preprocessing.feature_extract()
    print 'processed data.'
    model_params = {
        'pca_n': 10,
        'knn_k': 5,
        'knn_metric': 'minkowski'
    }
    # train_and_test(data, [model_params, 'svc'])
    model = load_model(model_params)
    test(model, data, [model_params, 'svc'])

if __name__ == '__main__':
    main()

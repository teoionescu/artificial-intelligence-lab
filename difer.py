
from inputReader import readTrainInput, readTestInput
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def balance(x):
    features = len(x[0])
    for j in range(features):
        mean = x[x[:, j] != None, j].mean()
        x[x[:, j] == None, j] = mean
    return np.array(x.tolist())

def split_nans(x, y):
    features = len(x[0])
    nones = x[:, 0] == None
    for j in range(features):
        nones |= x[:, j] == None
    x_ffeat = np.delete(x, np.where(nones), axis=0)
    y_ffeat = np.delete(y, np.where(nones), axis=0)
    x_nones = np.delete(x, np.where(nones == False), axis=0)
    y_nones = np.delete(y, np.where(nones == False), axis=0)
    x_ffeat = np.array(x_ffeat.tolist())
    x_nones = np.array(x_nones.tolist())
    assert x_ffeat.shape[0] + x_nones.shape[0] == x.shape[0]
    assert y_ffeat.shape[0] + y_nones.shape[0] == y.shape[0]
    return x_ffeat, x_nones, y_ffeat, y_nones, nones

def train_dirty(X, y, rnd_seed = 0):
    assert (X == None).sum() == 0
    mlp = MLPClassifier(
        hidden_layer_sizes = (60, 60),
        activation = 'tanh', #tanh
        solver = 'sgd',
        learning_rate = 'adaptive',
        random_state = 0 + rnd_seed,
        max_iter = 100000)
    mlp.fit(X, y[:,0])
    return mlp

def train_clean(X, y, rnd_seed = 0):
    assert (X == None).sum() == 0
    mlp = MLPClassifier(
        hidden_layer_sizes = (80),
        activation = 'relu', #relu
        solver = 'sgd',
        learning_rate = 'adaptive',
        random_state = 1 + rnd_seed,
        max_iter = 100000 )
    mlp.fit(X, y[:,0])
    return mlp

def my_predict(mlp_dirty, mlp_clean, X, nones):
    samples = X.shape[0]
    result = []
    for i in range(samples):
        if nones[i] == False:
            prediction = mlp_clean.predict([X[i]])[0]
        else:
            prediction = mlp_dirty.predict([X[i]])[0]
        result.append(prediction)
    return np.array(result)

def score(y, predict):
    print(classification_report(y, predict, digits=3).split('\n')[-2])

def main(rnd_seed = 0):
    print(rnd_seed)
    xl, yl = readTrainInput()
    x = np.array(xl)
    y = np.array(yl)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.95)
    # differential training
    X_train_ffeat, X_train_nones, y_train_ffeat, y_train_nones, _ = split_nans(X_train, y_train)
    X_test_ffeat, X_test_nones, y_test_ffeat, y_test_nones, nones_test = split_nans(X_test, y_test)
    X_train = balance(X_train)
    X_test = balance(X_test)
    X_train_nones = balance(X_train_nones)
    X_test_nones = balance(X_test_nones)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_ffeat = scaler.transform(X_train_ffeat)
    X_train_nones = scaler.transform(X_train_nones)
    X_test_ffeat = scaler.transform(X_test_ffeat)
    X_test_nones = scaler.transform(X_test_nones)

    mlp_dirty = train_dirty(X_train, y_train, rnd_seed)
    #score(y_test_nones, mlp_dirty.predict(X_test_nones))

    mlp_clean = train_clean(X_train_ffeat, y_train_ffeat, rnd_seed)
    #score(y_test_ffeat, mlp_clean.predict(X_test_ffeat))

    test_pred = my_predict(mlp_dirty, mlp_clean, X_test, nones_test)
    score(y_test, test_pred)

    XL = readTestInput()
    XT = np.array(XL)
    _, _, _, _, NT = split_nans(XT, XT)
    XT = balance(XT)
    XT = scaler.transform(XT)

    predictions = my_predict(mlp_dirty, mlp_clean, XT, NT)

    return test_pred, predictions

if __name__ == '__main__':
    main()
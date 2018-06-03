
import random
import numpy as np
from sklearn.model_selection import train_test_split
from inputReader import readTrainInput, readTestInput
from sklearn.metrics import classification_report, confusion_matrix
from difer import score
from difer import main

def inputSizes():
    xl, yl = readTrainInput()
    x = np.array(xl)
    y = np.array(yl)
    XL = readTestInput()
    XT = np.array(XL)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.95)
    return X_test.shape[0], XT.shape[0]

def relative_y():
    xl, yl = readTrainInput()
    x = np.array(xl)
    y = np.array(yl)
    XL = readTestInput()
    XT = np.array(XL)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.95)
    return y_test

def save_predictions(test_a, a):
    np.save('./test_a.npy', test_a)
    np.save('./a.npy', a)

def reset_predictions():
    test_size, size = inputSizes()
    xa = np.zeros((test_size, 10)).astype(int)
    xb = np.zeros((size, 10)).astype(int)
    save_predictions(xa, xb)

def load_predictions():
    test_a = np.load('./test_a.npy')
    a = np.load('./a.npy')
    return test_a, a

def evaluate_sample():
    test_a, a = load_predictions()
    y = relative_y()
    pred = test_a.argmax(1)
    score(y, pred)

def to_onehot(a):
    b = np.zeros((a.shape[0], 10))
    b[np.arange(a.shape[0]), a] = 1
    return b.astype(int)

def results():
    test_a, a = load_predictions()
    predictions = a.argmax(1)
    for i in range(10):
        print(i, ' ', sum(predictions == i))
    import csv
    with open('submission.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Id', 'Prediction'])
        for i in range(a.shape[0]):
            writer.writerow([str(i+1), str(predictions[i])])

def train(rnd_seed = 0):
    test_p, p = main(rnd_seed)
    test_po = to_onehot(test_p)
    po = to_onehot(p)
    test_a, a = load_predictions()
    test_po = np.matrix(test_po)
    po = np.matrix(po)
    test_a = np.matrix(test_a)
    a = np.matrix(a)
    test_a = np.array(test_a + test_po)
    a = np.array(a + po)
    save_predictions(test_a, a)

def training():
    badg = [474131]
    cdone = [526727, 207463, 170946] # chosen seeds
    clist = [63484, 945057, 420993]
    for i in cdone:
        train(i)
        print('Evaluation')
        evaluate_sample()

def raport():
    test_a, a = load_predictions()
    print(confusion_matrix(relative_y(), test_a.argmax(1)))
    rep = np.concatenate((test_a, relative_y()), axis = 1)
    np.set_printoptions(threshold=np.inf)
    #print(rep)

if __name__ == '__main__':
    #reset_predictions()
    #results()
    #training()
    #raport()
    evaluate_sample()
    pass

from sklearn import svm
import numpy as np
import csv


def paiwise_transform(x,y):
    print('transform pairwise to classify')
    x2 = []
    y2 = []

    for i in range(len(x)):
        for k in range(len(x)):
            if i == k or y[i, 0] == y[k, 0] or y[i, 1] != y[k, 1]:
                continue
            x2.append(x[i] - x[k])
            y2.append(np.sign(y[i, 0] - y[k, 0]))

    return np.asarray(x2), np.asarray(y2)
def sort(s):
   r=[]
   for i in range(len(s)):
      r.append((i+1,s[i]))
   r.sort(key=lambda tup:tup[1],reverse=True)
   r=np.asarray(r)
   return r

class RankSVM(svm.LinearSVC):
    def fit(self, X,y):
       X_trans,y_trans = paiwise_transform(X,y)
       super(RankSVM, self).fit(X_trans, y_trans)
       return self

    def predict(self, X):
        s = []
        for i in range(len(X)):
            t = []
            for k in range(len(X)):
                if i != k:
                    t.append(X[i] - X[k])
            s.append(sum(super(RankSVM, self).predict(np.asarray(t))))
        return sort(np.asarray(s))

def PredictionLayer():
    # data
    train = np.arange(92)
    test = np.arange(88)

    x = []
    for t in csv.reader(open('x.csv', 'r')):
        x.append(t)
    x = np.asarray(x, 'f')
    print(x.shape)
    # x=pca(x)

    y = []
    for t in csv.reader(open('y.csv', 'r')):
        y.append(t)
    y = np.asarray(y, 'd')

    # train
    rsvm = RankSVM().fit(x[train], y[train])
    # rank
    r = rsvm.predict(x[test])
    print(r)
if __name__ == '__main__':
    PredictionLayer()




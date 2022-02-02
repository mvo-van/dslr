import pandas as pd
import numpy as np
import json
import sys

class MyLogisticRegression():
    def __init__(self, theta, alpha=0.001, n_cycle=1000):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.theta = theta

    def sigmoid_(self, x):

        new = np.array(x)
        new = new.astype('float64')
        if len(x.shape) != 0:
            for index, elem in enumerate(new):
                new[index] = 1/(1+np.exp(-elem))
        else:
            new = np.array(1/(1+np.exp(-x)))
        return new

    def add(self, x):    
        b = np.array([1 for i in range(x.shape[0])])
        new_table = np.c_[b,x]
        return new_table

    def ligistic_predict_(self, x):
        return self.sigmoid_(self.add(x)@self.theta)

    def vec_log_gradient(self, x, y):
        return (self.add(x).T @ (self.sigmoid_((self.add(x) @ self.theta))-y))/x.shape[0]

    def fit_(self, x, y):
        cycle = self.max_iter
        save = None
        while cycle > 0 or save.all() != self.theta.all():
            save = self.theta
            grad = self.vec_log_gradient(x,y)
            self.theta = self.theta - (self.alpha * (grad))
            cycle -= 1
    
    def vec_log_loss_(self, y, y_hat, eps=1e-15):
        vec_one = np.ones(y.shape[0]).reshape((-1,1))
        res = y.flatten() @ np.log(y_hat + eps).flatten() + (vec_one - y).flatten() @ np.log(1 - y_hat + eps).flatten()
        return res/-y.shape[0]

def confusion_matrix_(y, y_hat, labels=None):
    conc = np.array([y,y_hat]).T
    if labels:
        new = []
        for row in conc:
            
            true = 1
            for cel in row:
                if(cel not in labels):
                    true = 0
            if true == 1:
                new += [row]
        conc = np.array(new)
    key = np.unique(conc)
    dicoVal = {}
    for index, val in enumerate(key):
        dicoVal[val] = index
    res = np.zeros((len(key),len(key)))
    for row in conc:
        res[dicoVal[row[0]]][dicoVal[row[1]]] += 1
    return res



lstFeature = ["Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Charms","Flying"]

def countTest(y, y_hat, true = 1):
    tp, fp, tn, fn = 0,0,0,0
    for valY, valHat in zip(y,y_hat):
        if valY == valHat:
            tp += 1
        else:
            fp += 1
    return tp, fp, tn, fn

def precision_score_(y, y_hat, pos_label=1):
    tp, fp, tn, fn = countTest(y, y_hat, true = pos_label)
    return tp / (tp + fp)

def oneVsAll(index, dataSet):
    lstOneOn = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    data = dataSet['Hogwarts House'].replace(["Hufflepuff","Ravenclaw","Slytherin","Gryffindor"],lstOneOn[index])
    
    mlRP = MyLogisticRegression([[0], [0], [0], [0],[0], [0], [0], [0],[0], [0], [0]],alpha=0.1,n_cycle=1000)
    Y = np.array(data).reshape((data.shape[0],1))
    X = np.array(dataSet[lstFeature])

    mlRP.fit_(X, Y)
    return mlRP.theta

def normalizeFeat(dataSet, feat, dicoinfo):
    min = dicoinfo[feat][0]
    max = dicoinfo[feat][1]
    dataSet[feat] = (dataSet[feat] - min) / (max - min)

def makePrediction(csv, dicoInfo):
    lstFeature = ["Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Charms","Flying"]
    lstHouse = ["Hufflepuff","Ravenclaw","Slytherin","Gryffindor"]
    dataSet = pd.read_csv(csv, usecols=lstFeature)

    resPredict = {}
    for feat in lstFeature:
        normalizeFeat(dataSet, feat, dicoInfo)
    dataSet = dataSet.fillna(dataSet.mean())
    for house in lstHouse:
        resPredict[house] = MyLogisticRegression(dicoInfo[house],alpha=0.01,n_cycle=6000).ligistic_predict_(dataSet)
    res = np.concatenate((resPredict["Hufflepuff"],resPredict["Ravenclaw"],resPredict["Slytherin"],resPredict["Gryffindor"]),axis=1)
    res = np.argmax(res, axis=1)

    resHouse = []
    for index, val  in enumerate(res):
        resHouse += [lstHouse[int(val)]]
    resHouse = np.array(resHouse)
    df = pd.DataFrame({'Index':range(len(resHouse)),'Hogwarts House': resHouse})
    return df.to_csv(index=False)



if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as test:
            with open("infoPred.json", 'r') as file:
                dicoInfo = json.loads(file.read())
                res = makePrediction(sys.argv[1], dicoInfo)
            with open("houses.csv", 'w') as file:
                file.write(res)


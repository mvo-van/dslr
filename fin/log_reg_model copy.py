import pandas as pd
import numpy as np
import math

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

dataSet = pd.read_csv("datasets/dataset_train.csv", usecols=["Hogwarts House"]+lstFeature)
dataSet = dataSet.dropna(axis=0)
#print(dataSet)

hufflepuff = dataSet['Hogwarts House'].replace(["Hufflepuff","Ravenclaw","Slytherin","Gryffindor"],[1,0,0,0])
ravenclaw = dataSet['Hogwarts House'].replace(["Hufflepuff","Ravenclaw","Slytherin","Gryffindor"],[0,1,0,0])
slytherin = dataSet['Hogwarts House'].replace(["Hufflepuff","Ravenclaw","Slytherin","Gryffindor"],[0,0,1,0])
gryffindor = dataSet['Hogwarts House'].replace(["Hufflepuff","Ravenclaw","Slytherin","Gryffindor"],[0,0,0,1])



mlRPHufflepuff = MyLogisticRegression([[0], [0], [0], [0],[0], [0], [0], [0],[0], [0], [0]],alpha=0.0001,n_cycle=6000)
Y0 = np.array(hufflepuff).reshape((hufflepuff.shape[0],1))
X = np.array(dataSet[lstFeature])
print(mlRPHufflepuff.vec_log_loss_(Y0,mlRPHufflepuff.ligistic_predict_(X)))

mlRPHufflepuff.fit_(X, Y0)
print(mlRPHufflepuff.theta)
predictHufflepuff = mlRPHufflepuff.ligistic_predict_(X)
print(mlRPHufflepuff.vec_log_loss_(Y0,mlRPHufflepuff.ligistic_predict_(X)))

mlRPRavenclaw = MyLogisticRegression([[0], [0], [0], [0],[0], [0], [0], [0],[0], [0], [0]],alpha=0.0001,n_cycle=6000)
Y1 = np.array(ravenclaw).reshape((ravenclaw.shape[0],1))
X = np.array(dataSet[lstFeature])
print(mlRPRavenclaw.vec_log_loss_(Y1,mlRPRavenclaw.ligistic_predict_(X)))

mlRPRavenclaw.fit_(X, Y1)
print(mlRPRavenclaw.theta)
predictRavenclaw = mlRPRavenclaw.ligistic_predict_(X)
print(mlRPRavenclaw.vec_log_loss_(Y1,mlRPRavenclaw.ligistic_predict_(X)))




mlRPSlytherin = MyLogisticRegression([[0], [0], [0], [0],[0], [0], [0], [0],[0], [0], [0]],alpha=0.0001,n_cycle=6000)
Y2 = np.array(slytherin).reshape((slytherin.shape[0],1))
X = np.array(dataSet[lstFeature])
print(mlRPSlytherin.vec_log_loss_(Y2,mlRPSlytherin.ligistic_predict_(X)))

mlRPSlytherin.fit_(X, Y2)
print(mlRPSlytherin.theta)
predictSlytherin = mlRPSlytherin.ligistic_predict_(X)
print(mlRPSlytherin.vec_log_loss_(Y2,mlRPSlytherin.ligistic_predict_(X)))

mlRPGryffindor = MyLogisticRegression([[0], [0], [0], [0],[0], [0], [0], [0],[0], [0], [0]],alpha=0.0001,n_cycle=6000)
Y3 = np.array(gryffindor).reshape((gryffindor.shape[0],1))
X = np.array(dataSet[lstFeature])
print(mlRPGryffindor.vec_log_loss_(Y3,mlRPGryffindor.ligistic_predict_(X)))

mlRPGryffindor.fit_(X, Y3)
print(mlRPGryffindor.theta)
predictGryffindor = mlRPGryffindor.ligistic_predict_(X)
print(mlRPGryffindor.vec_log_loss_(Y3,mlRPGryffindor.ligistic_predict_(X)))

res = np.concatenate((predictHufflepuff,predictRavenclaw,predictSlytherin,predictGryffindor),axis=1)
res = np.argmax(res, axis=1)

lstHouse = ["Hufflepuff","Ravenclaw","Slytherin","Gryffindor"]
resHouse = []
for index, val  in enumerate(res):
    resHouse += [lstHouse[int(val)]]
resHouse = np.array(resHouse)
resHouse = resHouse.reshape((resHouse.shape[0],1))

print(list(dataSet['Hogwarts House']), resHouse.reshape(res.shape[0]))
print(confusion_matrix_(list(dataSet['Hogwarts House']),resHouse.reshape(resHouse.shape[0]), labels=None))

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

print(precision_score_(list(dataSet['Hogwarts House']), resHouse.reshape(res.shape[0])))

dataTest = pd.read_csv("datasets/dataset_test.csv")
dataSet = dataSet.dropna(axis=0)
X = dataTest[lstFeature]
X = X.fillna(X.mean())

predictHufflepuff = mlRPHufflepuff.ligistic_predict_(X)
predictRavenclaw = mlRPRavenclaw.ligistic_predict_(X)
predictSlytherin = mlRPSlytherin.ligistic_predict_(X)
predictGryffindor = mlRPGryffindor.ligistic_predict_(X)

res = np.concatenate((predictHufflepuff,predictRavenclaw,predictSlytherin,predictGryffindor),axis=1)
res = np.argmax(res, axis=1)

lstHouse = ["Hufflepuff","Ravenclaw","Slytherin","Gryffindor"]
resHouse = []
for index, val  in enumerate(res):
    resHouse += [lstHouse[int(val)]]
resHouse = np.array(resHouse)
resHouse = resHouse.reshape((resHouse.shape[0],1))

voulu = pd.read_csv("datasets/dataset_truth.csv")
print(list(dataSet['Hogwarts House']), resHouse.reshape(res.shape[0]))
print(confusion_matrix_(list(voulu['Hogwarts House']),resHouse.reshape(resHouse.shape[0]), labels=None))


print(precision_score_(list(voulu['Hogwarts House']), resHouse.reshape(res.shape[0])))

def oneVsAll(house, index, dataSet):
    lstOneOn = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    data = dataSet['Hogwarts House'].replace(["Hufflepuff","Ravenclaw","Slytherin","Gryffindor"],lstOneOn[index])
    
    pass

def makePrediction(csv):
    lstFeature = ["Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Charms","Flying"]

    dataSet = pd.read_csv(csv, usecols=["Hogwarts House"]+lstFeature)
    dataSet = dataSet.dropna(axis=0)

    for index, house in enumerate(["Hufflepuff","Ravenclaw","Slytherin","Gryffindor"]):
        oneVsAll(house, index, dataSet)
    pass


def main(argv):
    if len(argv) > 1:
        with open(argv[1], 'r') as file:
            makePrediction(argv[1])

main()
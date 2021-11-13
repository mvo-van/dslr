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
                #print(elem)
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
        # print(add(x))
        # print("dkljslfkjldkfjlsdjsd",(sigmoid_((add(x) @ theta))-y))
        return (self.add(x).T @ (self.sigmoid_((self.add(x) @ self.theta))-y))/x.shape[0]

    def fit_(self, x, y):
        cycle = self.max_iter
        # print(self.thetas)
        # print()
        save = None
        while cycle > 0 or save.all() != self.theta.all():
            save = self.theta
            grad = self.vec_log_gradient(x,y)
            #print(grad)
            # print(self.alpha)
            # print("grad",grad)
            self.theta = self.theta - (self.alpha * (grad))
            cycle -= 1
            # print(cycle)
    
    def vec_log_loss_(self, y, y_hat, eps=1e-15):
        vec_one = np.ones(y.shape[0]).reshape((-1,1))

        # print(y.flatten()@ (y_hat + eps).flatten())
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
# lstFeature = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures"]
# lstFeature = ["Charms","Flying"]
# lstFeature = ["Arithmancy","Astronomy"]

dataSet = pd.read_csv("datasets/dataset_train.csv", usecols=["Hogwarts House"]+lstFeature)
# dataSet.fillna(0,inplace=True)
dataSet = dataSet.dropna(axis=0)
print(dataSet)




data_census = pd.read_csv("./day08/ex10/solar_system_census.csv", index_col=0)
data_census_planets = pd.read_csv('day08/ex10/solar_system_census_planets.csv', index_col=0)
# print(data_census, data_census_planets)

data_census_planets0 = data_census_planets['Origin'].replace([0.0,1.0,2.0,3.0],[1,0,0,0])
data_census_planets1 = data_census_planets['Origin'].replace([0.0,1.0,2.0,3.0],[0,1,0,0])
data_census_planets2 = data_census_planets['Origin'].replace([0.0,1.0,2.0,3.0],[0,0,1,0])
data_census_planets3 = data_census_planets['Origin'].replace([0.0,1.0,2.0,3.0],[0,0,0,1])



mlRP0 = MyLogisticRegression([[ 4.90348242], [-0.02999681], [-0.03250215], [ 2.40047782]],alpha=0.0001,n_cycle=10000)
Y0 = np.array(data_census_planets0).reshape((data_census_planets0.shape[0],1))
X = np.array(data_census)
print(mlRP0.vec_log_loss_(Y0,mlRP0.ligistic_predict_(X)))

# mlRP0.fit_(X, Y0)
print(mlRP0.theta)
predict0 = mlRP0.ligistic_predict_(X)
print(mlRP0.vec_log_loss_(Y0,mlRP0.ligistic_predict_(X)))
# print(Y)




mlRP1 = MyLogisticRegression([[ 1.28802845], [-0.06179008], [ 0.01894334], [ 8.00000601]],alpha=0.0001,n_cycle=10000)
Y1 = np.array(data_census_planets1).reshape((data_census_planets1.shape[0],1))
X = np.array(data_census)
print(mlRP1.vec_log_loss_(Y1,mlRP1.ligistic_predict_(X)))

# mlRP1.fit_(X, Y1)
print(mlRP1.theta)
predict1 = mlRP1.ligistic_predict_(X)
print(mlRP1.vec_log_loss_(Y1,mlRP1.ligistic_predict_(X)))




mlRP2 = MyLogisticRegression([[-4.75798975], [-0.00574743], [ 0.09731946], [-4.55362614]],alpha=0.0001,n_cycle=10000)
Y2 = np.array(data_census_planets2).reshape((data_census_planets2.shape[0],1))
X = np.array(data_census)
print(mlRP2.vec_log_loss_(Y2,mlRP2.ligistic_predict_(X)))

# mlRP2.fit_(X, Y2)
print(mlRP2.theta)
predict2 = mlRP2.ligistic_predict_(X)
print(mlRP2.vec_log_loss_(Y2,mlRP2.ligistic_predict_(X)))






mlRP3 = MyLogisticRegression([[-2.20593027], [ 0.08724529], [-0.09877385], [-8.59898021]],alpha=0.0001,n_cycle=10000)
Y3 = np.array(data_census_planets3).reshape((data_census_planets3.shape[0],1))
X = np.array(data_census)
print(mlRP3.vec_log_loss_(Y3,mlRP3.ligistic_predict_(X)))

# mlRP3.fit_(X, Y3)
print(mlRP3.theta)
predict3 = mlRP3.ligistic_predict_(X)
print(mlRP3.vec_log_loss_(Y3,mlRP3.ligistic_predict_(X)))


res = np.concatenate((predict0,predict1,predict2,predict3),axis=1)
res = np.argmax(res, axis=1)
res = res.reshape((res.shape[0],1))

#print(np.concatenate((data_census_planets, res), axis=1))
print(list(data_census_planets['Origin']), res.reshape(res.shape[0]))
print(confusion_matrix_(list(data_census_planets['Origin']),res.reshape(res.shape[0]), labels=None))

def countTest(y, y_hat, true = 1):
    tp, fp, tn, fn = 0,0,0,0
    for valY, valHat in zip(y,y_hat):
        if valY == true:
            if valHat == true:
                tp += 1
            else:
                fn += 1
        else:
            if valHat == true:
                fp += 1
            else:
                tn += 1
    return tp, fp, tn, fn

def precision_score_(y, y_hat, pos_label=1):
    tp, fp, tn, fn = countTest(y, y_hat, true = pos_label)
    return tp / (tp + fp)

print(precision_score_(list(data_census_planets['Origin']), res.reshape(res.shape[0])))
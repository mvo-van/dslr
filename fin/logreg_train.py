import pandas as pd
import numpy as np
import json
import sys
from log_reg_model import MyLogisticRegression

lstFeature = ["Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Charms","Flying"]

def oneVsAll(index, dataSet):
    lstOneOn = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    data = dataSet['Hogwarts House'].replace(["Hufflepuff","Ravenclaw","Slytherin","Gryffindor"],lstOneOn[index])
    
    mlRP = MyLogisticRegression([[0], [0], [0], [0],[0], [0], [0], [0],[0], [0], [0]],alpha=0.1,n_cycle=1000)
    Y = np.array(data).reshape((data.shape[0],1))
    X = np.array(dataSet[lstFeature])

    mlRP.fit_(X, Y)
    return mlRP.theta

def normalizeFeat(dataSet, feat):
    min = dataSet[feat].min()
    max = dataSet[feat].max()
    dataSet[feat] = (dataSet[feat] - min) / (max - min)
    return [min, max]

def makeTrain(csv):
    lstFeature = ["Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Charms","Flying"]
    dicoInfo = {}
    dataSet = pd.read_csv(csv, usecols=["Hogwarts House"]+lstFeature)
    dataSet = dataSet.dropna(axis=0)

    for feat in lstFeature:
        dicoInfo[feat] = normalizeFeat(dataSet, feat)
    for index, house in enumerate(["Hufflepuff","Ravenclaw","Slytherin","Gryffindor"]):
        dicoInfo[house] = oneVsAll(index, dataSet).tolist()
    return dicoInfo

if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            dicoInfo = makeTrain(sys.argv[1])
            with open("infoPred.json", 'w') as file:
                file.write(json.dumps(dicoInfo))

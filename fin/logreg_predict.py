import pandas as pd
import numpy as np
import json
import sys
from log_reg_model import MyLogisticRegression

lstFeature = ["Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Charms","Flying"]

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


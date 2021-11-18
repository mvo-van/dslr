import pandas
import matplotlib.pyplot as plt
import numpy as np

lstFeature = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
lstAnalyse = ["","count","mean","std"]
lstNameHouse = ["Ravenclaw","Slytherin","Gryffindor","Hufflepuff"]
lstHouse = {"Ravenclaw":{},"Slytherin":{},"Gryffindor":{},"Hufflepuff":{}}
infoHouse = {}

dataSet = pandas.read_csv("datasets/dataset_train.csv", usecols=["Hogwarts House"]+lstFeature)
dataSet.fillna("",inplace=True)


for house,elem in lstHouse.items():
    for feat in lstFeature:
        lstHouse[house][feat] = []

for index,house in enumerate(dataSet["Hogwarts House"]):
    for feat in lstFeature :
        lstHouse[house][feat] += [dataSet[feat][index]]

    
for house,dataHouse in lstHouse.items() :
    dictFeature = {}
    for key,feature in dataHouse.items():
        somme = 0
        info = {"count":0}
        ordoner = []

        for note in feature:
            if note:
                ordoner += [float(note)]
        ordoner.sort()
        info["count"] = len(ordoner)

        for note in ordoner:
            somme += note
        info["mean"] = somme / info["count"]

        somme = 0
        for note in ordoner:
            somme += (note - info["mean"])**2
        info["std"] = (somme / info["count"])**0.5

        for index, cel in enumerate(lstHouse[house][key]) :
            if cel == "":
                lstHouse[house][key][index] = info["mean"]

        dictFeature[key] = info
    infoHouse[house] = dictFeature



for house,info in infoHouse.items():
    for id,line in enumerate(lstAnalyse):
        string = "%-10s"%line
        if id == 0: 
            for col in lstFeature:
                string += "%13s"%col[:13]
        else:
            for col in lstFeature:
                string += "%13.6f"%info[col][line]


compFeat = {}

def ft_abs(nb):
    if nb < 0:
        nb = -nb
    return nb

dictFeature[key]

def comparFeat(compar, lstFeat):
    result = {}
    for featComp in lstFeat:
        sommeCov = 0
        sommeVarX = 0
        sommeVarY = 0
        count = 0
        for house in lstNameHouse:
            for index, note in enumerate(lstHouse[house][compar]):
                sommeCov += (lstHouse[house][featComp][index] - dictFeature[featComp]['mean'])*(lstHouse[house][compar][index] - dictFeature[compar]['mean'])
                sommeVarX += (lstHouse[house][featComp][index] - dictFeature[featComp]['mean'])**2
                sommeVarY += (lstHouse[house][compar][index] - dictFeature[compar]['mean'])**2
                count += 1
        cov = sommeCov / count
        varX = sommeVarX / count
        varY = sommeVarY / count

        coefCorrelation = cov / ((varX*varY)**0.5)

        result[featComp] = coefCorrelation
    return result



compFeat = {}
for i in range(12):
    compFeat[lstFeature[i]] = comparFeat(lstFeature[i],lstFeature[i+1:])
    print(lstFeature[i])
    print(compFeat[lstFeature[i]])
    


min = 0

for feat1,lstComp in compFeat.items():
    for feat2,result in lstComp.items():
        if abs(result) > min:
            min = abs(result)
            featMin1 = feat1
            featMin2 = feat2

print(featMin1, featMin2)

max_nbins = 10
data1=lstHouse["Ravenclaw"][featMin1]
data2=lstHouse["Slytherin"][featMin1]
data3=lstHouse["Gryffindor"][featMin1]
data4=lstHouse["Hufflepuff"][featMin1]
data5=lstHouse["Ravenclaw"][featMin2]
data6=lstHouse["Slytherin"][featMin2]
data7=lstHouse["Gryffindor"][featMin2]
data8=lstHouse["Hufflepuff"][featMin2]
data_range = [0,500]
binwidth=(data_range[1]-data_range[0])/max_nbins


plt.scatter(data1[:20],data5[:20],c='#AE0001',alpha=1)
plt.scatter(data2[:20],data6[:20],c='#222F5B',alpha=1)
plt.scatter(data3[:20],data7[:20],c='#F0C75E',alpha=1)
plt.scatter(data4[:20],data8[:20],c='#2A623D',alpha=1)
plt.ylabel(featMin2)
plt.xlabel(featMin1)
plt.title(featMin1)
plt.show()

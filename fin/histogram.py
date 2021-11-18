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
    for feat in lstFeature :
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

def comparHouse(compar, lstHouse, feature):
    result = []
    for house in lstHouse:
        mean1 = infoHouse[house][feature]["mean"]
        mean2 = infoHouse[compar][feature]["mean"]
        std1 = infoHouse[house][feature]["std"]
        std2 = infoHouse[compar][feature]["std"]
        count1 = infoHouse[house][feature]["count"]
        count2 = infoHouse[compar][feature]["count"]
        result += [ft_abs(mean1-mean2)/(((std1**2/count1)+(std2**2/count2))**0.5)]
    return result

for feat in lstFeature:
    print(feat)
    compFeat[feat] = []
    for i in range(3):
        compFeat[feat] += comparHouse(lstNameHouse[i],lstNameHouse[i+1:],feat)
    print(compFeat[feat])
    
    somme = 0
    for nb in compFeat[feat]:
        somme += nb
    compFeat[feat] = somme/len(compFeat[feat])


min = 100
featMin = ""
for feat,nb in compFeat.items():
    if nb < min:
        min = nb
        featMin = feat

max_nbins = 10
data1=lstHouse["Ravenclaw"][featMin]
data2=lstHouse["Slytherin"][featMin]
data3=lstHouse["Gryffindor"][featMin]
data4=lstHouse["Hufflepuff"][featMin]
data_range = [0,500]
binwidth=(data_range[1]-data_range[0])/max_nbins

for dt in [data1,data2,data3,data4]:
    while "" in dt:
        dt.remove("")

mini = int(data2[0])
maxi = int(data3[-1])

bins = [x for x in range(mini,maxi,10000)]

plt.hist(data4,bins = bins, color = '#F0C75E', alpha = 0.25, label = 'data4_name')
plt.hist(data2,bins = bins, color = '#2A623D', alpha = 0.25, label = 'data2_name')
plt.hist(data1,bins = bins, color = '#222F5B', alpha = 0.25, label = 'data1_name')
plt.hist(data3,bins = bins, color = '#AE0001', alpha = 0.25, label = 'data3_name')

plt.ylabel('y_label')
plt.xlabel('x_label')
plt.title(featMin)
plt.show()
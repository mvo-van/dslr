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
    #print(house)
    for id,line in enumerate(lstAnalyse):
        string = "%-10s"%line
        if id == 0: 
            for col in lstFeature:
                string += "%13s"%col[:13]
        else:
            for col in lstFeature:
                string += "%13.6f"%info[col][line]
        #print(string)

compFeat = {}

def ft_abs(nb):
    if nb < 0:
        nb = -nb
    return nb

def comparFeat(compar, lstFeat):
    result = {}
    for featComp in lstFeat:
        somme = 0
        count = 0
        for house in lstNameHouse:
            for index, note in enumerate(lstHouse[house][compar]):
                somme += note - lstHouse[house][featComp][index]
                count += 1
        moyDesDifferences = somme / count
        somme = 0
        count = 0
        for house in lstNameHouse:
            for index, note in enumerate(lstHouse[house][compar]):
                somme += (note - lstHouse[house][featComp][index])**2 
                count += 1
        ecartType = ((somme/ count )- (moyDesDifferences)**2)**0.5
        sd = ((count/(count-1))*ecartType)**0.5
        # Tobs = (ft_abs(moyDesDifferences))/(sd/(count**0.5))
        Tobs = (ft_abs(moyDesDifferences))/(sd/(count**0.5))
        result[featComp] = Tobs
        #print(count, moyDesDifferences, Tobs, ecartType)
    return result



compFeat = {}
for i in range(12):
    compFeat[lstFeature[i]] = comparFeat(lstFeature[i],lstFeature[i+1:])
    print(lstFeature[i])
    print(compFeat[lstFeature[i]])
    


min = 100

for feat1,lstComp in compFeat.items():
    for feat2,result in lstComp.items():
        if result < 100:
            print(feat1,feat2)
        if result < min:
            # max_nbins = 10
            # data1=lstHouse["Ravenclaw"][feat1]
            # data2=lstHouse["Slytherin"][feat1]
            # data3=lstHouse["Gryffindor"][feat1]
            # data4=lstHouse["Hufflepuff"][feat1]
            # data5=lstHouse["Ravenclaw"][feat2]
            # data6=lstHouse["Slytherin"][feat2]
            # data7=lstHouse["Gryffindor"][feat2]
            # data8=lstHouse["Hufflepuff"][feat2]
            # data_range = [0,500]
            # binwidth=(data_range[1]-data_range[0])/max_nbins

            # plt.scatter(data1[:20],data5[:20],c='#AE0001',alpha=1)
            # plt.scatter(data2[:20],data6[:20],c='#222F5B',alpha=1)
            # plt.scatter(data3[:20],data7[:20],c='#F0C75E',alpha=1)
            # plt.scatter(data4[:20],data8[:20],c='#2A623D',alpha=1)
            # plt.ylabel(feat2)
            # plt.xlabel(feat1)
            # plt.title(feat1)
            # plt.show()

            min = result
            featMin1 = feat1
            featMin2 = feat2

print(featMin1, featMin2)

# bins = [x + 1 for x in range(0, 1000, 100)]
# for house in lstNameHouse:
#     plt.hist(lstHouse[house][featMin], bins = bins)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.title('title')
# plt.show()

featMin1 = "Astronomy"
featMin2 = "Defense Against the Dark Arts"

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


# while "" in data1:
#     data1.remove("")

# data1 =sorted(data1)
# while "" in data2:
#     data2.remove("")


# print("2")

# while "" in data3:
#     data3.remove("")

# print("3")

# while "" in data4:
#     data4.remove("")

# print("4")


# while "" in data5:
#     data5.remove("")

# while "" in data6:
#     data6.remove("")

# while "" in data7:
#     data7.remove("")

# while "" in data8:
#     data8.remove("")




plt.scatter(data1[:20],data5[:20],c='#AE0001',alpha=1)
plt.scatter(data2[:20],data6[:20],c='#222F5B',alpha=1)
plt.scatter(data3[:20],data7[:20],c='#F0C75E',alpha=1)
plt.scatter(data4[:20],data8[:20],c='#2A623D',alpha=1)
plt.ylabel(featMin2)
plt.xlabel(featMin1)
plt.title(featMin1)
plt.show()

print((4)**0.5)

#ax.legend(loc='best')
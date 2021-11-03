import pandas

lstFeature = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
lstAnalyse = ["","count","mean","std","min","25%","50%","75%","max"]


dataSet = pandas.read_csv("datasets/dataset_train.csv", usecols=lstFeature)
dataSet.fillna("",inplace=True)
dictFeature = {}


for key,feature in dataSet.items():
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

    info["min"]=ordoner[0]
    info["max"]=ordoner[-1]
    info["25%"]=ordoner[int(info["count"]/4 + 0.9)]
    info["50%"]=ordoner[int(info["count"]/2 + 0.9)]
    info["75%"]=ordoner[int((info["count"]/4)*3 + 0.9)]

    dictFeature[key] = info


for id,line in enumerate(lstAnalyse):
    string = "%-10s"%line
    if id == 0: 
        for col in lstFeature:
            string += "%13s"%col[:13]
    else:
        for col in lstFeature:
            string += "%13.6f"%dictFeature[col][line]
    print(string)



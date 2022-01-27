import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

lstFeature = ["Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Charms","Flying"]
dataSet = pandas.read_csv("datasets/dataset_train.csv", usecols=["Hogwarts House"]+lstFeature)
pairplot = sns.pairplot(dataSet,vars=lstFeature, hue = "Hogwarts House", corner=True)
plt.show()
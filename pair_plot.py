import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

lstFeature = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]

dataSet = pandas.read_csv("datasets/dataset_train.csv", usecols=["Hogwarts House"]+lstFeature)
dataSet.fillna("",inplace=True)

sns_plot = sns.pairplot(dataSet, hue = "Hogwarts House")
sns_plot = 
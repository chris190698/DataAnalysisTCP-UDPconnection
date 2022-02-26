from functions import * #importo il file che contiene tutte le funzioni
import numpy as np
from operator import itemgetter
from random import randrange

trainpath = "trainDdosLabelNumeric.csv" #percorso del training set che useremo per l'addestramento
dataSet = loadDataSet(trainpath)
target = 'Label' #la nostra classe
seed = 42
np.random.seed(seed)

#info del dataset
print("________SHAPE:________")
print(dataSet.shape) #stampa delle righeXcolonne del dataset
print("\n")
print("________HEAD:________")
print(dataSet.head()) #stampa le prime 5 righe del dataset
print("\n")
print("________COLUMNS:________")
print(dataSet.columns) #stampa le colonne presenti nel dataset
print("\n")

#pre elaborazione dei nostri dati, ci serve per capire quali attributi sono utili per la fase di training
columns = list(dataSet.columns.values) #prendiamo i nomi di ogni colonna con i relativi valori,creandone una lista
preElaborationData(dataSet, columns, target)
newDataSet, removedColumns = removeColumns(dataSet, columns)
print("\n________REMOVED COLUMNS________")
print(removedColumns) #stampo le colonne rimosse
print("\n")
preElaborationClass(newDataSet, target)

#utilizzo mutual info per rankare le feature
independentList = list(newDataSet.columns.values)
independentList.remove(target)
rankingFeature = mutualInfoRank(newDataSet, independentList, target, seed)
print("\n________MUTUAL INFO________")
print(rankingFeature) #stampo il ranking delle mie feature

#selezioniamo le prime n feature in base al mutual info
n = 10
topList = topFeatureSelect(rankingFeature, n)
topList.append(target)
print("\n________TOP N LIST ________")
print(topList)
#costruisco un nuovo dataframe che contiene solo gli attributi che ho ricavato da topFeatureSelect + la classe
selectedMIData = newDataSet.loc[:, topList]
print("\n________ SELECTED MI DATA SHAPE________")
print(selectedMIData.shape)
print("\n________ SELECTED MI DATA HEAD________")
print(selectedMIData.head())
print("\n________ SELECTED MI DATA COLUMNS________")
print(selectedMIData.columns)

#applichiamo lo scaling al nostro dataset di 66 colonne e successivamente applichiamo una mutual info ranking per notare le differenze
newScaledDataSet = newDataSet.copy(deep=True) #mi copio il mio dataset di partenza
newScaledDataSet = scalingDataMinMax(newScaledDataSet, independentList)
rankingFeatureScaled = mutualInfoRank(newScaledDataSet, independentList, target, seed)
print("\n________MUTUAL INFO SCALED________")
print(rankingFeatureScaled) #stampo il ranking delle mie feature, notiamo come i valori calcolati sono leggermente diversi prima del seed

#procediamo col il calcolo della PCA per indentificare le componenti principali
X = newDataSet.loc[:, independentList]
pca, pcalist = pca(X)
pcaData = applyPCA(X, pca, pcalist)
pcaData.insert(loc=len(independentList), column=target, value=newDataSet[target], allow_duplicates=True) # aggiunta della label come ultima colonna
print("\n________PCA________")
print(pcaData)
print("\n________PCA HEAD________")
print(pcaData.head())
print("\n________PCA SHAPE________")
print(pcaData.shape)

#procediamo con il calcolo della PCA prendendo n componenti principali
pcaDataSelected = selectedPCAData(X, n)
independentListPca = list(pcaDataSelected.columns.values)
pcaDataSelected.insert(loc=len(independentListPca), column=target, value=newDataSet[target], allow_duplicates=True) # aggiunta della label come ultima colonna
print("\n________PCA SELECTED________")
print(pcaDataSelected)
print("\n________PCA SELECTED HEAD________")
print(pcaDataSelected.head())
print("\n________PCA SELECTED SHAPE________")
print(pcaDataSelected.shape)

#stratified k-fold cross validation ci può servire per capire come settare i parametri per poi applicare l'algoritmo
#sceglieremo la miglior configurazione
x = newDataSet.loc[:, independentList]
y = newDataSet[target]
folds = 5 #nClassi
ListXTrain, ListYTrain, ListXTest, ListYTest = stratifiedKfold(x, y, folds, seed)
print("\n________LIST X TRAIN________\n")
print(ListXTrain)
print("\n________LIST Y TRAIN________\n")
print(ListYTrain)
print("\n________LIST X TEST________\n")
print(ListXTest)
print("\n________LIST Y TEST________\n")
print(ListYTest)

#calcolo infogain
rankingFeatureInfoGain = giRank(newDataSet, independentList, target)
print("\n________RANKED FEATURE INFO GAIN________")
print(rankingFeatureInfoGain)

#costruisco un albero decisionale per prova
T = decisionTreeLearner(x, y, "entropy", 500) #scegliamo il criterio dell'entropia per la bontà degli split
print("\n________SHOW TREE________")
showTree(T)

#ricerco la miglior configurazione per costruire l'albero basandomi su ranking di feature calcolate in maniera diversa
#1) MUTUAL INFO
bestCMI, bestNMI, bestF1MI = determineDecisionTreekFoldConfiguration(ListXTrain, ListYTrain, ListXTest, ListYTest, rankingFeature)
print("\n________BEST CONFIGURATION WITH MUTUAL INFO________")
print("bestC: ", bestCMI, "\nbestN: ", bestNMI, "\nbestF1: ", bestF1MI)

#2) INFO GAIN
bestCIG, bestNIG, bestF1IG = determineDecisionTreekFoldConfiguration(ListXTrain, ListYTrain, ListXTest, ListYTest, rankingFeatureInfoGain)
print("\n________BEST CONFIGURATION WITH INFO GAIN________")
print("bestC: ", bestCIG, "\nbestN: ", bestNIG, "\nbestF1: ", bestF1IG)

#3) PCA
#passo i miei dati trasformati al calolo della cross validation
xPCA = pcaData.loc[:, pcalist]
yPCA = pcaData[target]
ListXPCATrain, ListYPCATrain, ListXPCATest, ListYPCATest = stratifiedKfold(xPCA, yPCA, folds, seed)
list = []
for c in pcalist:
    list.append(1.0)

res = dict(zip(pcalist, list))
rankPCA = sorted(res.items(), key=lambda kv: kv[1], reverse=True)

bestCPCA, bestNPCA, bestF1PCA = determineDecisionTreekFoldConfiguration(ListXPCATrain, ListYPCATrain, ListXPCATest, ListYPCATest, rankPCA)
print("\n________BEST CONFIGURATION WITH PCA________")
print("bestC: ", bestCPCA, "\nbestN: ", bestNPCA, "\nbestF1: ", bestF1PCA)

#determino la miglior configurazione
listConfiguration = []
tuplaMutualInfo = (bestCMI, bestNMI, bestF1MI, "mi")
tuplaInfoGain = (bestCIG, bestNIG, bestF1IG, "ig")
tuplaPCA = (bestCPCA, bestNPCA, bestF1PCA, "pca")
listConfiguration.append(tuplaMutualInfo)
listConfiguration.append(tuplaInfoGain)
listConfiguration.append(tuplaPCA)

max_value = max(listConfiguration, key=itemgetter(2))[2] #fscore
min_value = min(listConfiguration, key=itemgetter(1))[1] #bestn
max_list = []
min_list = []
bestConfiguration = ()
for c in listConfiguration:
    if (c[2] == max_value):
        max_list.append(c)

if (len(max_list) == 1):
    bestConfiguration = max_list
else:
    for m in max_list:
        if (m[1] == min_value):
            min_list.append(m)

if (len(min_list) == 1):
    bestConfiguration = min_list
else:
    bestConfiguration = min_list[randrange(len(min_list))]

print("\n________BEST CONFIGURATION EVER________")
print(bestConfiguration)

#mi carico il mio testing set e procedo con le stesse operazioni
pathTest = "testDdosLabelNumeric.csv"
testSet = loadDataSet(pathTest)

xBest = []
yBest = []
xTestBest = []
yTestBest = []

if(bestConfiguration[0][3] == "mi"):
    topList = topFeatureSelect(rankingFeature, bestConfiguration[0][1])
    xBest = newDataSet.loc[:, topList]
    yBest = newDataSet[target]
    xTestBest = testSet.loc[:, topList]
    yTestBest = testSet[target]
elif(bestConfiguration[0][3] == "ig"):
    topList = topFeatureSelect(rankingFeatureInfoGain, bestConfiguration[0][1])
    xBest = newDataSet.loc[:, topList]
    yBest = newDataSet[target]
    xTestBest = testSet.loc[:, topList]
    yTestBest = testSet[target]
else:
    topList = topFeatureSelect(rankPCA, bestConfiguration[0][1])
    xBest = pcaData.loc[:, topList]
    yBest = pcaData[target]
    xTestBest = testSet.loc[:, topList]
    yTestBest = testSet[target]

tBest = decisionTreeLearner(xBest, yBest, bestConfiguration[0][0], 500)
print("\n________TREE BEST CONFIGURATION________")
showTree(tBest)

#calcoliamo la matrice di consufusione inerente la miglior configurazione
confusionMatrix(yBest, tBest.predict(xBest), tBest.classes_)
print("\n________REPORT TRAINING________")
printReport(yBest, tBest.predict(xBest))

#calcoliamo la matrice di consufusione inerente la miglior configurazione
confusionMatrix(yTestBest, tBest.predict(xTestBest), tBest.classes_)
print("\n________REPORT TEST________")
printReport(yTestBest, tBest.predict(xTestBest))

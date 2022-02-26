import math
import pandas as pandas
import matplotlib.pyplot as plt
import numpy as np

"""
funzione che mi permette di caricare un dataset da un file csv e convertirlo in una struttura data frame
PARAM path: percorso del file csv
RETURN: dataframe
"""
def loadDataSet(path):
    return pandas.read_csv(path)

"""
funzione che stampa una descrizione dei miei attributi in termini di metriche statistiche
PARAM dataSet: dataSet contenete i dati
PARAM columns: lista degli attributi presenti nel nostro dataSet
RETURN: void
"""
def preElaborationData(dataSet, columns,label):
    print("________DATA DESCRIBE________")
    for col in columns:
        print(dataSet[col].describe())
        print("\n")

        #stampiamo il boxplot con la distribuzione dei valori rispetto alla classe
        dataSet.boxplot(column=col, by=label)
        plt.show()

"""
funzione che rimuove tutte le colonne non utili per la classificazione e restituisce il nuovo dataset e le colonne rimosse
PARAM dataset: dataset contenete tutti gli esempi
PARAM columns: attributi del nostro dataset sul quale itereremo
RETURN newDataSet: nuovo dataset meno le colonne non utili alla classificazione
RETURN removedColumns: lista delle colonne rimosse
"""
def removeColumns(dataSet, columns):
    removedColumns = list()
    for col in columns:
        if(dataSet[col].min() == dataSet[col].max()): #scartiamo gli attributi che hanno lo stess valore max e min
            removedColumns.append(col)

    # passiamo alla funzione la lista delle colonne da rimuovere,
    # axis=1 indice che stiamo passando delle colonne,
    # inplace=False perchè ci interessa creare una copia
    newDataSet = dataSet.drop(removedColumns, axis=1, inplace=False)
    return newDataSet, removedColumns

"""
funzione che visualizza il trend dei dati in base alla classe, visualizzeremo un istogramma
PARAM dataSet: dataSet dal quale ricavare i valori della classe
PARAM label: la classe che vogliamo classificare
RETURN void
"""
def preElaborationClass(dataSet, label):
    split = dataSet.groupby(label).count() #splittiamo i dati in base alla label e tiriamo fuori il numero di occorrenze per ogni classe
    counts = split.iloc[:, 1:2] #accedimo a tutti i valori dello split di prima distiguendo valore e totale occorrenze
    print("________LABEL COUNTS________")
    print(counts)

    dataSet[label].plot(x=label, title=label, kind="hist")
    plt.show() #stampiamo l'istogramma

"""
funzione che ci permette di calcolare una metrica tra due diverse variabili, che rappresentano l'attributo da valutare e la classe.
Assume valori che indicano quanto può essere rilevante l'attributo.Più è alto più è rilevante
Alternativa all'infoGain perchè già implementata in sklearn
Calcola solo il singolo attributo, non è esaustivo
PARAM dataSet: il dataset sul quale vogliamo calcolare le nostre metriche
PARAM independetList: la lista delle variabili indipendenti
PARAM label: la classe da tenere in considerazione
RETURN sorted_x: dizionario, una sorta di ranking in base al valore di mutual info con ordinamento DESC
"""
def mutualInfoRank(dataSet, independentList, label, seed):
    from sklearn.feature_selection import mutual_info_classif

    #creo un dizionario chiave,valore dove avrò il nome della variabile indipendente con il suo valore di mutual info calcolato
    res = dict(
            zip(independentList,
                mutual_info_classif( #resituisce la lista dei valori mutual info calcolati per ciascuna variabile indipendente
                    dataSet[independentList], #rappresenta il nostro x, la matrice che contiene le nostre feature
                    dataSet[label], #rappresenta il nostro y, ovvero la classe che ci serve per la valutazione della feature
                    discrete_features=False, #consideriamo solo valori di tipo numerico
                    random_state=seed #ci permette di calcolare alla stessa maniera i dati, riprendendo il confronto con min max scaler in quel caso non è più necessario utilizzazlo
                    )
                )
    )
    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_x

"""
funzione che ci permette di produrre una lista dei nomi di n elementi ordinati in base al mutual info, prendiamo solo l'intestazione della colonna
PARAM rankMI: dizionario contenente(attributo,valore) ordinato per mutual info DESC
PARAM N: quante feature vogliamo
RETURN topList: lista di n valori ordinati per mutual info
"""
def topFeatureSelect(rankMI, N):
    topList = list()
    for i in range(0, N):
        topList.append(rankMI[i][0])

    return topList

"""
funzione che ci permette di applicare uno scaling,tecnica di traformazione. In questo caso i valori saranno 
trasformati in una scale che va da zero ad 1
PARAM dataSet: dataset da trasformare
PARAM indipendentList: lista degli attributi indipendenti sul quale calcolare il nuovo valore
RETURN newDataSet: nuovo dataset trasformato
"""
def scalingDataMinMax(dataSet, independentList):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    newDataSet = dataSet
    newDataSet[independentList] = scaler.fit_transform(dataSet[independentList]) #fitta e trasforma
    return newDataSet

"""
funzione che ci permette di costruire il modello e la lista delle componenti 
PARAM dataSet: dataSet sul quale operare
RETURN pca: modello PCA che abbiamo calcolato
RETURN pcaList: lista contentente il nome delle colonne che stiamo costruendo
"""
def pca(dataSet):
    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(dataSet)
    pcaList = []
    for c in range(len(dataSet.columns.values)):
        name = "pc_" + str(c + 1)
        pcaList.append(name)

    return pca, pcaList

"""
funzione che calcola le componenti principali
PARAM dataSet: dataframe sul quale effettuare la trasformazione
PARAM pca: modello pca calcolato precedentemente
PARAM pcaList: lista contentente il nome delle colonne che stiamo costruendo
RETURN pcaData: dataFrme trasformato secondo la tecnica PCA
"""
def applyPCA(dataSet, pca, pcaList):
    principalComponents = pca.transform(dataSet) #restituisce una matrice
    pcaData = pandas.DataFrame(data=principalComponents, columns=pcaList)
    return pcaData

"""
funzione che ci permette di trasformare il nostro dataset con la tecnica pca prendendo n componenti principali
PARAM dataSet: dataframe da trasformare
PARAM N: numero di componenti principali
RETURN pcaDataSet: datatframe trasformato
"""
def selectedPCAData(dataSet, N):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=N)
    pca.fit(dataSet)
    pcaList = []
    for i in range(0, N):
        name = "pc_" + str(i + 1)
        pcaList.append(name)

    pcaDataSet = applyPCA(dataSet, pca, pcaList)
    return pcaDataSet

"""
funzione che ci permette di applicare la cross validation e quindi dividere in n fold il dataset e prendere il campione migliore
PARAM x: dataset sul quale effettuare le operazioni meno la classe
PARAM y: porzione del dataset che riguarda la classe
PARAM folds: numero di strati in cui vogliamo dividere il nostro dataset
PARAM seed: seme per ottenere gli stessi risultati(trucco tecnico)
RETURN listXTrain: lista di tutti i training set proiettati rispetto ad x
RETURN listYTrain: lista di tutti i training set proiettati rispetto ad y
RETURN listXTest: lista di tutti i testing set proiettati rispetto ad x
RETURN listYTest: lista di tutti i testing set proiettati rispetto ad y
"""
def stratifiedKfold(x, y, folds, seed):
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=folds)
    skf.get_n_splits(x, y)
    StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True) #impostiamo shuffle=True perchè vogliamo essere sicuri che i dati siano stati opportunamente mescolati
    listXTrain = []
    listYTrain = []
    listXTest = []
    listYTest = []
    for train_index, test_index in skf.split(x, y):
        #uso iloc per prendere l'esempio che corrisponde a quell'indice
        listXTrain.append(x.iloc[train_index])
        listYTrain.append(y.iloc[train_index])
        listXTest.append(x.iloc[test_index])
        listYTest.append(y.iloc[test_index])

    return listXTrain, listYTrain, listXTest, listYTest

"""
funzione che calcola l'entropia
PARAM classDistribution: distribuzione della classe
PARAM nClass: numerodi classi
RETURN entropy: entropia calcolata
"""
def entropy(classDistribution, nClass):
    entropy = 0
    for cD in classDistribution:
        entropy = entropy - cD * math.log(cD, nClass)

    return entropy

"""
funzione che calcola l'ingogain
PARAM: col: attributo inerente al calcolo
PARAM label: classe a prendere in considerazione
RETURN infoGain: info gain dell'attributto calcolato
"""
def infogain(col, label):
    class_distribution = (label.value_counts().to_numpy()) / label.shape[0]  # conteggio dei valori unici della classe/ numero di esempi = probabilità della classe
    nClass = class_distribution.size #numero di classi
    entropyClass = entropy(class_distribution, nClass) #entropia della classe
    values = col.unique() #mi prendo i valori distinti che assume la feature
    average = 0
    for v in values:
        exampleCount = col[col == v].shape[0] #prendo gli esempi per quel valore
        classCount = label[col == v].groupby(label).count().to_numpy() #prendo le classi in corrispondenza del valore
        probability = classCount / exampleCount
        average = average + (exampleCount * entropy(probability, nClass)) #calolo la media pesata

    average = average / col.shape[0] #calcolo la probabilità condizionata della classe, media pesata
    infogain = entropyClass - average
    return infogain



"""
funzione che ci permette di richiamare il calcolo dell'infogain per ogni attributo
PARAM dataSet: dataSet contenente gli attributi
PARAM label: classe in base a cui calcolare la classe
RETURN info: lista conte
"""
def giClassif(data,label):
    cols = list(data.columns.values)
    info = []
    for c in cols:
        info.append(infogain(data[c], label))

    return info

"""
funzione che costruisce un dizionario dove come chiave abbiamo l'attributto e come valore l'info gain calcolato,
ordinando il tutto in maniera decrescente, più è alto l'infogain più l'attributo è utile
PARAM dataSet: dataset contenete gi attributi e gli esempi
PARAM independentList: lista delle variabili indipendenti
PARAM label: la classe che ci servierà per il calcolo
RETURN sorted_x: dizionario ordinato per infogain (DESC)
"""
def giRank(dataSet, independentList, label):
    res = dict(zip(independentList, giClassif(dataSet[independentList], dataSet[label])))
    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_x

"""
funzione che costruisce un albero di decisione, lo addestriamo con 500 esempi a split per evitare l'overfitting
PARAM x: training set delle variabili indipendenti per costruire l'albero
PARAM y: training set per l'attributo classe
PARAM c: criterio per la costruzione dell'albero
PARAM n: numero di esempi da prendere per splittare
RETURN t: albero decisionale addestrato sul nostro training set
"""
def decisionTreeLearner(x, y, c, n):
    from sklearn.tree import DecisionTreeClassifier

    t = DecisionTreeClassifier(criterion=c, #criterio per misurare la bontà degli split
                               random_state=0, #serve per avere sempre lo stesso risultato
                               min_samples_split=n #numero minimo di esempi per eseguire uno split, altrimenti problema di overfitting
                               )
    t.fit(x, y)

    return t

"""
funzione che permette di stampare il decision tree costruito in precedenza
PARAM t: albero decisionale costrutito precedentemente
RETURN void
"""
def showTree(t):
    from sklearn import tree

    print('#Nodi: ', t.tree_.node_count, '#Foglie:', t.tree_.n_leaves)
    plt.figure(figsize=(20, 20))
    tree.plot_tree(t, filled=True, #colora i nodi per indicare la classe maggioritaria
                   fontsize=8, #size del testo
                   proportion=True) #serve per visualizzare i campioni e i valori in maniera proporzionale
    plt.show()

"""
funzione che calcola l'F1 score (2 * (precision * recall) / (precision + recall) ) metrica di valutazione
PARAM xTest: testing set contenente le variabili indipendenti
PARAM yTest: testing set contenente le variabili dipendenti
PARAM t: decision tree addestrato in precedenza
RETURN f1: score calcolato sul testing set
"""
def decisionTreeF1(xTest, yTest, t):
    from sklearn.metrics import f1_score

    yPred = t.predict(xTest)
    f1 = f1_score(yTest, yPred, average="weighted")
    return f1

"""
funzione che ci permette di determinare la miglior configurazione in termini dei parametri da passare alla funzione 
per la costruzione di un albero decisionale
PARAM listXTrain, listYTrain, listXTest, listYTest: training set e testing set generati dalla cross validation
PARAM rankedList: lista delle migliori feature
RETURN bestC: miglior criterio per la bontà degli split
RETURN bestN: numero di feature selezionate
RETURN bestF1: f1 score della miglior configurazione
"""
def determineDecisionTreekFoldConfiguration(listXTrain, listYTrain, listXTest, listYTest, rankedList):
    bestF1 = -1
    bestN = -1
    bestC = 'none'
    crit = ['gini', 'entropy'] # lista criteri possibili
    list = np.arange(5, len(rankedList), 5)
    list = np.insert(list, len(list), len(rankedList))
    for N in list:
        for c in crit:
            f1 = []
            for xTrain, yTrain, xTest, yTest in zip(listXTrain, listYTrain, listXTest, listYTest):
                topList = topFeatureSelect(rankedList, N) # selezioniamo le migliori top n feature
                xTrainSelected = xTrain.loc[:, topList] # proiettiamo le feature selezionate al passo prima sul nostro training set
                t = decisionTreeLearner(xTrainSelected, yTrain, c, 500) #apprendiamo/addestriamo l'albero di decisione
                XTestSelected = xTest.loc[:, topList] #selezioniamo le top n migliori feature
                f1.append(decisionTreeF1(XTestSelected, yTest, t)) #calcolo l'f1 score sull'albero addestrato

            avg_F1 = np.mean(f1) #calcolo la media dell'f1 score

            if avg_F1 > bestF1:
                bestF1 = avg_F1
                bestC = c
                bestN = N

    return bestC, bestN, bestF1

"""
funzione che ci permette di generare una matrice di confusione, printandola a video
PARAM yTrue: valori corretti della classe
PARAM yPred: valori predetti dal nostro classificatore
PARAM label: Elenco di etichette per indicizzare la matrice
RETURN void
"""
def confusionMatrix(yTrue, yPred, label):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(yTrue, yPred, labels=label) #List of labels to index the matrix. This may be used to reorder or select a subset of labels.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=label)
    disp.plot()
    plt.show()

"""
funzione che ci permette di generare/stampare un report delle metriche completo
PARAM yTrue: valori corretti della classe
PARAM yPred: valori predetti dal nostro classificatore
PARAM label: Elenco di etichette per indicizzare la matrice
RETURN void
"""
def printReport(yTrue, yPred):
    from sklearn.metrics import classification_report

    print(classification_report(yTrue, yPred))

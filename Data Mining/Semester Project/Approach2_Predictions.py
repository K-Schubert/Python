from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import pandas as pd
import sklearn
import scipy.stats
import matplotlib
import math

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 

#Datasets are loaded after having been preprocessed with the importfunction.py code and importfiles.py
all_data=None
CP=pd.read_csv('/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/btk_mac_os/CP.csv',sep=',')
#CP['Category']="CP"
FD=pd.read_csv('/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/btk_mac_os/FD.csv',sep=',')
#FD['Category']="FD"
ITW=pd.read_csv('/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/btk_mac_os/ITW.csv',sep=',')
#ITW['Category']="ITW"
all_data=pd.concat([CP,FD,ITW],axis=0,sort=False)
#all_data.to_csv('/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/btk_mac_os/CP_FD_ITW.csv',sep=',')
#all_data=pd.concat([FD,ITW],axis=0,sort=False)
ncol=all_data.shape[1]
nrow=all_data.shape[0]
#print(all_data)
#all_data.isna().sum().sum()

#Randomize the training set over the patients for the training and test set split (80,20)
def Split(all_data):
    "Split the data 2/3, 1/3"
    #Split the datas X,Y
    #666-345-123
    np.random.seed(666)
    l=all_data.shape[1]
    listpatient=pd.unique(all_data.loc[:,'Patient'])
    index=np.random.permutation(len(listpatient))
    train_ind=index[:(np.int(0.8*len(listpatient)))]#get index for the training set
    test_ind=np.delete(index,train_ind) #get index for the test set
    listtrain=pd.DataFrame(listpatient[train_ind],columns=['Indic'])#get the patient index for train set
    listtest=pd.DataFrame(listpatient[test_ind],columns=['Indic'])#get the patient index for test set
    trainset=all_data.merge(listtrain.iloc[:,0],how='left',right_on=listtrain.iloc[:,0],left_on=all_data.loc[:,'Patient'])
    trainset=trainset.dropna(axis=0,subset=trainset.columns[l:l+2])#Delete the empty rows(test set)
    trainset=trainset.iloc[:,1:l+1]
    testset=all_data.merge(listtest.iloc[:,0],how='left',right_on=listtest.iloc[:,0],left_on=all_data.loc[:,'Patient'])
    testset=testset.dropna(axis=0,subset=testset.columns[l:l+2])#Delete the empty rows(training set)
    testset=testset.iloc[:,1:l+1]
    return(trainset,testset)

#Crossvalidation function with 5 folds. Return a graph with mean and standard error
def CrossValid(traincross,algo,nver,g):
    mean=np.zeros(len(nver))
    sd=np.zeros(len(nver))
    b=0
    r=5
    l=traincross.shape[1]
    listpatient=pd.unique(traincross.loc[:,'Patient'])
    pat=pd.DataFrame(listpatient)
    for i in nver:
        np.random.seed(r)
        index=np.array(list(pat.index))
        np.random.shuffle(index)
        list_of_fold=np.array_split(index,5,axis=0)
        acc=np.zeros(5)
        print(index)
        k=0
        for fold in list_of_fold:
            listvalid=pd.DataFrame(listpatient[fold])
            listtrain=pd.DataFrame(np.delete(listpatient,fold))            
            #listtrain=pd.DataFrame(listpatient[train_ind],columns=['Indic'])#get the patient index for train set
            #listvalid=pd.DataFrame(listpatient[test_ind],columns=['Indic'])#get the patient index for test set
            trainset=traincross.merge(listtrain.iloc[:,0],how='left',right_on=listtrain.iloc[:,0],left_on=traincross.loc[:,'Patient'])
            trainset=trainset.dropna(axis=0,subset=trainset.columns[l:l+2])#Delete the empty rows(test set)
            trainset=trainset.iloc[:,1:l+1]
            validset=traincross.merge(listvalid.iloc[:,0],how='left',right_on=listvalid.iloc[:,0],left_on=traincross.loc[:,'Patient'])
            validset=validset.dropna(axis=0,subset=validset.columns[l:l+2])#Delete the empty rows(training set)
            validset=validset.iloc[:,1:l+1]
            xtrain, ytrain, conttrain = prep(trainset,rando=1,classe=1)
            xvalid, yvalid, contvalid = prep(validset,rando=0,classe=1)
            xtrain = Scale(xtrain)
            xvalid = Scale(xvalid)
            acc[k]=algo(xtrain,ytrain,xvalid,yvalid,i,g)
            k+=1
        mean[b]=np.mean(acc)
        sd[b]=np.std(acc)
        b+=1
        r+=1
    plt.xscale('log')
    plt.errorbar(nver,mean,yerr=sd,marker='s', mfc='red',mec='green', ms=20, mew=4)
    plt.show
    return(mean,sd)

#Preparation of dataset train, valid and test. For train, the missing values rows are ignored,
#for the test and valid set the missing values are replaced by the mean
#The context is a variable with the index to use the info on the patient later
def prep(data,rando,classe):
    allssna=pd.DataFrame()
    if (rando==1):
        allssna=data.dropna(axis=0,subset=data.columns[2:88])
        allssna=allssna.sample(n=len(allssna),axis=0)
    else:
        allssna=data
        allssna=allssna.fillna(allssna.mean())
    context=pd.DataFrame(allssna.index)
    if (classe==1):
        di=allssna.shape[1]-1
    else:
        di=allssna.shape[1]-2
    allssna=pd.concat([allssna.iloc[:,2:88],allssna.iloc[:,di]],axis=1) #keep only the x's and the y's in the dataset
    #allssna=allssna.round(2) #Round the data 
    X_values=allssna.iloc[:,:86]
    y_values=allssna.iloc[:,-1]
    return(X_values,y_values, context)

#Datasets xtrain xvalid and xtest are scaled
def Scale(X_data):
    namedata=X_data.columns
    scaler = StandardScaler()  
    # Don't cheat - fit only on training data
    scaler.fit(X_data)  
    X_data = pd.DataFrame(scaler.transform(X_data))
    X_data.columns=namedata
    return(X_data) 

#Functions that define different methods to compute predictions
#(Naive,kNN,Random Forest, Gradient Boosting, Neural Network)
def Naivemodel(X_train,y_train,X_test,y_test,k,nod):
    gnb = GaussianNB()
    pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %f" % (y_test.shape[0],(y_test != pred).sum()*100/y_test.shape[0]))
    acc=(y_test != pred).sum()*100/y_test.shape[0]
    #return(acc)
    return(pred)

def knnmodel(X_train,y_train,X_test,y_test,k,nod):
    error = []
    #for u in range(k):
    classifier = KNeighborsClassifier(n_neighbors=k)  
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    #print(confusion_matrix(y_test, y_pred))  
    #print(classification_report(y_test, y_pred))
    print("Number of mislabeled points out of a total %d points : %f" % (y_test.shape[0],(y_test != pred).sum()*100/y_test.shape[0]))
    acc=(y_test != pred).sum()*100/y_test.shape[0]
    #return(acc)
    return(pred)

def RF(X_train,y_train,X_test,y_test,k,g):
    clf = RandomForestClassifier(n_estimators=100,max_features=30,random_state=0)
    clf.fit(X_train, y_train)
    pred=clf.predict(X_test)
    print("Number of mislabeled points out of a total %d points : %f" % (y_test.shape[0],(y_test != pred).sum()*100/y_test.shape[0]))
    acc=(y_test != pred).sum()*100/y_test.shape[0]
    #return(acc)
    return(pred)

def GB(X_train,y_train,X_test,y_test,k,nod):
    #learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
    #for learning_rate in learning_rates:
    #    gb = GradientBoostingClassifier(n_estimators=200, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    #    gb.fit(X_train, y_train)
    #    print("Learning rate: ", learning_rate)
    #    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    #    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
    #    print()
    gb = GradientBoostingClassifier(n_estimators=40, learning_rate = 0.1, max_features=10, max_depth = 4, random_state = 0)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    #print("Confusion Matrix:")
    #print(confusion_matrix(y_test, predictions))
    #print()
    #print("Classification Report")
    #print(classification_report(y_test, predictions))
    print("Number of mislabeled points out of a total %d points : %f" % (y_test.shape[0],(y_test != pred).sum()*100/y_test.shape[0]))
    acc=(y_test != pred).sum()*100/y_test.shape[0]
    #return(acc)
    return(pred)

#(500,200,100,60,50)
def NN(X_train,y_train,X_test,y_test,k,nod):
    clf2 = MLPClassifier(solver='sgd', alpha=1, hidden_layer_sizes=(500,200,10), random_state=1)
    clf2.fit(X_train, y_train) 
    MLPClassifier(activation='relu', alpha=1, batch_size=100,
              beta_1=0.9, beta_2=0.999, early_stopping=True,
              epsilon=1e-08, hidden_layer_sizes=(500,200,10),
              learning_rate='adaptive', learning_rate_init=0.0001,
              max_iter=500, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5,  random_state=1,
              shuffle=True, solver='sgd', tol=0.0001,
              validation_fraction=0.2, verbose=False, warm_start=True)
    pred=clf2.predict(X_test)
    print("Number of mislabeled points out of a total %d points : %f" % (y_test.shape[0],(y_test != pred).sum()*100/y_test.shape[0]))
    acc=(y_test != pred).sum()*100/y_test.shape[0]
    return(pred)
    #return(acc)

def scor(predModel):
    #Merge the contextual information with the results
    # in case we want to update the events in the c3d files
    pred=pd.DataFrame()
    pred=pd.concat([conttest,predModel],axis=1,sort=False)
    pred.columns=['Context','Classe1']
    datafin=testdata.merge(pred,how='left',left_index=True,right_on=pred.iloc[:,0])

    #  function that calculates the accuracy on the events only
    ##Construct a table with the real events and the predictions to compare
    Final=datafin['Patient'].astype(str) + ' ' + datafin['Trial'].astype(str)
    Final=pd.concat([Final,datafin['Classe1']],axis=1)
    Final=pd.concat([Final,(datafin['Label'] + ' ' + datafin['Context_x'])],axis=1)
    Final=pd.concat([Final,datafin['Frame']],axis=1)
    Final.columns=['Patient','Classe1','Realevent','RealFrame']
    listpat=pd.unique(Final['Patient'])
    all_events=[]
    events=[]
    for pat in listpat:
        Finalsub=Final.loc[Final['Patient']==pat,:]
        events.append('na')
        for m in range(1,len(Finalsub)):
            prec=Finalsub.iloc[m-1,1]
            on=Finalsub.iloc[m,1]
            if prec != on:
                if on=='R': events.append('Foot Off Left')
                elif on=='LR': events.append('Foot Strike Left')
                elif on=='L': events.append('Foot Off Right')
                elif on=='RL':events.append('Foot Strike Right')
                else: events.append('na')
            else: events.append('na')
    all_events=pd.DataFrame(events)
    Final2=pd.concat([Final,all_events], axis=1)
    Final2.columns=['Patient','Classe1','Realevent','RealFrame','Predictions']

    ##Calculate the score for the predictions on the test set
    Nevent=len(Final2)-Final2['Realevent'].isnull().sum()
    Neventdetected=len(Final2)-(Final2['Predictions']=='na').sum()
    score=0
    k=0
    p=0
    scorerate=0
    for n in range(0,len(Final2)):
        if (n>10 and n<(len(Final2)-10)):
            if pd.isna(Final2.iloc[n,2])== False :
                k=k+1        
                for tr in range(-10,10):
                    if Final2.iloc[n+tr,4]==Final2.iloc[n,2]:
                        score=score+abs(tr)
                        p=p+1
    scorerate=score/p
    results=[score,k,p,scorerate,Neventdetected]
    return(results)

#Main calls function that prepares the datasets and calls the different methods
#either with the test set or the crossvalidation
traindata, testdata = Split(all_data)
xtrain, ytrain, conttrain = prep(traindata,rando=1,classe=1)
xtest, ytest, conttest = prep(testdata, rando=0,classe=1)
xtrain = Scale(xtrain)
xtest = Scale(xtest)

#NaiveBayes
#Crossvalid
nver=[1]
nodes=[500]
#pred=CrossValid(traindata,Naivemodel,nver,nodes)

#Testset
predNaive =(Naivemodel(xtrain,ytrain,xtest,ytest,nver,nodes))
predNaive=pd.DataFrame(predNaive)
resNB=scor(predNaive)
print(resNB)

#kNN
#Crossvalid
#nver=[10]
#nodes=[500]
#pred=CrossValid(traindata,knnmodel,nver,nodes)

#Testset
predknn1=(knnmodel(xtrain,ytrain,xtest,ytest,50,nodes))
predknn1=pd.DataFrame(predknn1)
resknn1=scor(predknn1)
print(resknn1)

#RandomForest
#Crossvalid
#nver=[10]
#nodes=[500]
#pred=CrossValid(traindata,RF,nver,nodes)

#Testset
predRF=RF(xtrain,ytrain,xtest,ytest,nver,nodes)
predRF=pd.DataFrame(predRF)
resRF=scor(predRF)
print(resRF)

#Gradient boosting
#Crossvalid
#nver=[2,4,6,8,10]
#nodes=[500]
#pred=CrossValid(traindata,GB,nver,nodes)

#Testset
predGB=(GB(xtrain,ytrain,xtest,ytest,nver,nodes))
predGB=pd.DataFrame(predGB)
resGB=scor(predGB)
print(resGB)

#NeuralNet
#Crossvalid
#nver=[1.e-1,1,1.e1]
#nver=[500]
#nodes=[500]
#pred=CrossValid(traindata,NN,nver,nodes)

#Testset
predNN=NN(xtrain,ytrain,xtest,ytest,500,1)
predNN=pd.DataFrame(predNN)
resNN=scor(predNN)
print(resNN)

#Export predictions
#Final2.to_csv('/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/btk_mac_os/score2.csv',sep=',')
#datafin.to_csv('/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/btk_mac_os/pred.csv',sep=',')
#ytest.to_csv('/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/btk_mac_os/test.csv',sep=',')

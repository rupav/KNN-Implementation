# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:22:45 2017

@author: MARK
"""

'''
to be done:
    1) ListedColormap(colors=['',''])
    2) plt.xlim,plt.ylim :"x-axis and y-axis limits"
    3) alpha in plt.hist  : "In computer graphics, alpha compositing is the process of combining an image with a background to create the appearance of partial or full transparency"
    4) np.meshgrid
'''

'''
KNN implementation
'''
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score, train_test_split,KFold
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots

from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF' ])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])





def euclid_cal(TestData,TestTrain):
    return ((((TestData-TestTrain)**2).sum())**0.5)

def majority(top_labels,KNN):
    count_0 = 0
    count_1 = 0
    
    for i in range(KNN):
        if (top_labels[i] == 1):
            count_1 += 1
        else:
            count_0 += 1
    #print(count_0,count_1)        
    
    if(count_0 > count_1):
        return 0
    else:
        return 1

    
def predictKNN(x_train,y_train,x_test,y_test,KNN):
    y_pred = []
    for k in range (x_test.shape[0]):               #for every point
        
        distances = []
        TestData = np.array(x_test.iloc[k,:])       # whole col.
        
        for i in range(x_train.shape[0]): ##for every row
                
            TrainData = np.array(x_train.iloc[i,:])
            dist  = euclid_cal(TestData,TrainData)
            distances.append((dist, i))                
        
        distances.sort()
        top_indexes =[]
        top_labels = []
        
        
            
        for m in range(KNN):
            top_indexes.append(distances[:KNN][m][1])               
            top_labels.append(y_train.iloc[top_indexes[m],0])
            #print (top_labels[m])
            
        y_pred.append(majority(top_labels,KNN))
    y_pred = pd.DataFrame(y_pred)
    #print("Confusion Matrix :",confusion_matrix(y_test,y_pred))
    #print("Classification Report :\n",classification_report(y_test,y_pred))
        
        
    return y_pred

def run():
    dataset = datasets.load_breast_cancer()
    print(dataset.data.shape)

    x_train,x_test,y_train,y_test = cross_validation.train_test_split(dataset.data,dataset.target,test_size=0.2)
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    x_test = pd.DataFrame(x_test)
    y_test = pd.DataFrame(y_test)

    for i in range(1,16,2):                                                     #to find best K
        y_pred = predictKNN(x_train,y_train,x_test,y_test,i)

        TN_TP = (((y_pred^y_test)==0).sum())     ## XOR: if 0 then correct, if 1 then wrong Prediction

        accuracy = (TN_TP)/(y_pred.shape[0])
        print("Accuracy for k = ",i,"  is:\n",accuracy)

        TP = (((y_pred==1))&(y_test==1)).sum()
        print("TP:",TP)

        FN = (((y_pred==0))&(y_test==1)).sum()
        print("FN:",FN)

        FP = (((y_pred==1))&(y_test==0)).sum()
        print("FP:",FP)

        TN = ((y_pred==0)&(y_test==0)).sum()        
        print("TN:",TN)

    return 

#run()

'''
dataset = datasets.load_breast_cancer()
print(dataset.DESCR)
print(dataset.data)
print(dataset.target.shape)
'''

df = pd.read_csv('BreastCancer.csv',header = 0)   #got the url from dataset.descr
print(df.columns)
print(df.shape)

df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
print(df['diagnosis'])

df.drop('id',axis = 1,inplace = True)
print(df.head())
print(df.shape)
outputDF = df['diagnosis']
print(df.dtypes)             #all flotat64 except diagnosis is int64   
print(pd.isnull(df).sum())   #not a single entry is missing
inputDF= df.iloc[:,1:11]
print(inputDF.head())

#data columns is not so sparse, but still scaling

def scale(inputDF):
    
    for i in range(inputDF.shape[1]):
        inputDF.iloc[:,i] = (inputDF.iloc[:,i].mean()-inputDF.iloc[:,i])/(inputDF.iloc[:,i].mean())
        
    return 

scale(inputDF)
print (inputDF.head())

def plotting(inputDF,outputDF):
    
    dfM = inputDF[outputDF==1]   ##maligned inputDF
    dfB = inputDF[outputDF==0]    ##benign inputDF
    print("dfM shape: ",dfM.shape)
    print("dfB shape: ",dfB.shape)
    
    plt.rcParams.update({'font.size':8})
    fig, axes = plt.subplots(nrows=5,ncols=2,figsize=(8,10))
    #print(fig)
    axes = axes.ravel()
    #print(axes)
    
    for index,ax in enumerate(axes):
        ax.figure
        binwidth = (inputDF.iloc[:,index].max()-inputDF.iloc[:,index].min())/50
        bins = (np.arange(start=inputDF.iloc[:,index].min(),stop=inputDF.iloc[:,index].max()+binwidth,step=binwidth))
        
        
        ax.hist(
                [dfM.iloc[:,index],
                dfB.iloc[:,index]],
                bins = bins, 
                alpha = 0.5,
                stacked = True,
                normed = True,
                label = ['M','B'],
                color = ['r','g'] 
                )
        
        ax.legend(loc='upper right')
        ax.set_title(inputDF.columns[index])
        
    plt.tight_layout()
    plt.show()
    
    return
                    
plotting(inputDF,outputDF)

# from plotting results
#radius_mean,area_mean,perimeter_mean,compactness_mean,concavity and concave points are important for classification

inputDF.drop(['texture_mean','smoothness_mean','symmetry_mean','fractal_dimension_mean'],axis=1,inplace = True)
print(inputDF.head())

#import seaborn as sns
#cmap = sns.cubehelix_palette(n_colors=7,start=0,rot=0.5,dark=0,light=0.9,as_cmap=True)

def KNNplot(inputDF,outputDF):
    ### we need 2 columns for plotting KNN ,radius,perimeter and area are interrealted, so 1 can be chosen from them i.e. radius
    ### taking other feature as concavity for KNN plotting
    inputDF = inputDF.filter(items = ['radius_mean','concavity_mean'], axis = 1)
    print(inputDF.shape)
    
    #sns.pairplot(inputDF)
    #plt.show()
    
    
    x_train,x_test,y_train,y_test = cross_validation.train_test_split(inputDF,outputDF,test_size=0.2)
    
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    x_train.reset_index(drop=True,inplace = True)
    y_train.reset_index(drop=True,inplace = True)
    
    x_min, x_max = x_train.iloc[:,0].min() , x_train.iloc[:,0].max()+1
    y_min ,y_max = x_train.iloc[:,1].min() , x_train.iloc[:,1].max()+1
    h1 = (x_max-x_min)/10
    h2 = (y_max-y_min)/10
    
    xx , yy = np.meshgrid(np.arange(start = x_min,stop = x_max,step = h1),
                          np.arange(start = y_min,stop = y_max,step = h2))
    
    
    plottingInput = pd.DataFrame(np.c_[xx.ravel(),yy.ravel()])
    plottingInput.columns = ['radius_mean','concavity_mean']
    print(plottingInput.shape)
    print(plottingInput.head())
    
    plottingResult = predictKNN(x_train,y_train,plottingInput,y_test,5)
    print(plottingResult.shape)
    #color plot
    Z = np.array(plottingResult)
    print(type(Z))
    print(Z.shape)
    Z = Z.reshape(xx.shape)
    print(Z.shape)
    plt.pcolormesh(xx,yy,Z,cmap = cmap_light)
    
    #training data plot
    plt.scatter(x_train.iloc[:,0],x_train.iloc[:,1],c=y_train,cmap = cmap_bold)
    plt.xlim(xx.min(), xx.max())   
    plt.ylim(yy.min(), yy.max())
    plt.show()
    return     
            
KNNplot(inputDF,outputDF)    
    



    










                
                    
                    
        
                
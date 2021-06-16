import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from seaborn import countplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def TitanicLogistic():
    print("Inside Logistic function")
    #Step 1:load data
    titanic_Data=pd.read_csv("MarvellousTitanicDataset.csv")
    print("First Five record of data set")
    print(titanic_Data.head())

    print("Total number of record are:",len(titanic_Data))#titanic_Data.shap :dimension

    #step 2:analyze data
    print("Visualisation:Survived and non survied passangers")
    figure()
    countplot(data=titanic_Data,x="Survived").set_title("Survived vs Non-Survived")
    show()

    print("Visualisation according to gender")
    figure()
    countplot(data=titanic_Data,x="Survived",hue="Sex").set_title("Visualisation according to gender")
    show()

    print("Visualisation according to Pclass")
    figure()
    countplot(data=titanic_Data,x="Survived",hue="Pclass").set_title("Survived vs non survived according to Pclass")
    show()

    print("Visualisation according to age")
    figure()
    titanic_Data["Age"].plot.hist().set_title("Visualisation according to age")
    show()

    #Step 3:Data Clearing
    titanic_Data.drop("zero",axis=1,inplace=True)#axis 1 =col and 0 =row
    print("Data after coloum removel")
    print(titanic_Data.head())

    Sex=pd.get_dummies(titanic_Data["Sex"])
    print(Sex)

    Sex=pd.get_dummies(titanic_Data["Sex"],drop_first=True)
    print("Sex coloum after updation")
    print(Sex)

    Pclass=pd.get_dummies(titanic_Data["Pclass"],drop_first=True)
    print(Pclass.head())

    #concate sex and pclass field in our dataset
    titanic_Data=pd.concat([titanic_Data,Sex,Pclass],axis=1)
    print(titanic_Data.head(5))

    #removing no needed fields
    titanic_Data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
    print(titanic_Data.head(5))

    #divied the dataset into x and y
    x=titanic_Data.drop("Survived",axis=1)#if there is no 'inplace' then it will not remove servived from titanic_data
    y=titanic_Data["Survived"]

    #split the data for traing and testing purpose
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5)

    logmodel=LogisticRegression(max_iter=1000)#error solve for max iteration
    #step 4:Data training
    logmodel.fit(xtrain,ytrain)
    #step 5:Data testing
    prediction=logmodel.predict(xtest)

    print("accuracy of given data is:")
    print((accuracy_score(ytest,prediction))*100)

    print("Confusion matrix is:")
    print((confusion_matrix(ytest,prediction)))

def main():
    print("Logistic casestudy")

    TitanicLogistic()
if __name__ == '__main__':
    main()
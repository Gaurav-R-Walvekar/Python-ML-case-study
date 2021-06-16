import  pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def Predictor(path):

    data=pd.read_csv(path)
    print("Dataset loaded succefully with the size:",len(data))

    Wether=data.Wether
    Temperature=data.Temperature
    Play=data.Play

    lobj=preprocessing.LabelEncoder()

    WetherX=lobj.fit_transform(Wether)
    TemperatureX=lobj.fit_transform(Temperature)
    Label=lobj.fit_transform(Play)

    print("Encoded wether is:")
    print(WetherX)

    print("Encoded Temperature is:")
    print(TemperatureX)

    feture=list(zip(WetherX,TemperatureX))#it is list of list
    print(feture)

    obj=KNeighborsClassifier(n_neighbors=3)#check nearest 3 value.i.e:k=3

    obj.fit(feture,Label)

    ret=obj.predict([[0,2]])#we have to take list of list because there are two feture

    if ret==1:
        print("play")
    else:
        print("dont play")

def main():
    print("_________Play Predictor-----------")
    print("Enter the path of the file which contains dataset:")
    path="C:\\Users\\ME\\Desktop\\assignment\\Play.csv"

    Predictor(path)



if __name__ == '__main__':
    main()
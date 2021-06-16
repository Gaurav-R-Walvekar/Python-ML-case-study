import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def CheckAccuracy(ret,Target_test):
    acc=accuracy_score(Target_test,ret)
    print("Accuracy is:",acc*100,"%")

def main():
    Data=pd.read_csv("C:\\Users\\ME\\Desktop\\assignment\\WinePredictor.csv")
    Data.columns=Data.columns.str.replace(" ","_")#use to replay spaces in words#replace ' ' with _
    #Data.to_csv("Wine.csv")#to save data
    #print(Data)
    feture=list(zip(Data.Alcohol,Data.Malic_acid,Data.Ash,Data.Alcalinity_of_ash,Data.Magnesium,Data.Total_phenols
                    ,Data.Flavanoids,Data.Nonflavanoid_phenols,Data.Proanthocyanins,Data.Color_intensity,Data.Hue,Data.OD280_OD315_of_diluted_wines,
                    Data.Proline))
    #feture=pd.DataFrame(feture)#to make dataframe
    #print(feture)
    target=Data.Class
    #print(target)

    obj=KNeighborsClassifier(n_neighbors=3)

    Data_train,Data_test,Target_train,Target_test=train_test_split(feture,target,test_size=0.3)
    #print(len(Data_test),len(Target_test))#to see testind data count

    obj.fit(Data_train,Target_train)

    ret=obj.predict(Data_test)
    print(ret)
    print(list(Target_test))

    CheckAccuracy(ret,Target_test)

if __name__ == '__main__':
    main()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def HeadBrain(Name):
    dataset=pd.read_csv(Name)
    print("size of dataset",dataset.shape)

    X=dataset["Head Size(cm^3)"].values
    Y=dataset["Brain Weight(grams)"].values
    X=X.reshape((-1,1))
    obj=LinearRegression()
    obj.fit(X,Y)

    output=obj.predict(X)
    #Dataset=pd.read_csv("Test.csv")
    #X_new=dataset["Head_size"].values
    #output=obj.prefict(X_new)
    #print("Expected Result is:",output)

    rsquare=obj.score(X,Y)

    print("Value of R Square is:",rsquare)

def main():
    print("Enter file name of dataset")
    name="C:\\Users\\ME\\Desktop\\Today1\\HeadBrain.csv"
    HeadBrain(name)

if __name__ == '__main__':
    main()
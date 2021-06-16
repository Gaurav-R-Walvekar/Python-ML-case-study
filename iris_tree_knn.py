from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def TreeDecision(data_train,data_test,target_train,target_test):
    
    cobj=tree.DecisionTreeClassifier()
    
    cobj.fit(data_train,target_train)
    
    output=cobj.predict(data_test)
    
    Accuracy=accuracy_score(target_test,output) #use to see accuracy or check you data tested:accuracy_score(ANS,check tested ANS)
    #print(target_train)
    return Accuracy

def KNN(data_train,data_test,target_train,target_test):
    
    cobj=KNeighborsClassifier()#another algorithm like tree but will give diffrent accuracy
    
    cobj.fit(data_train,target_train)
    
    output=cobj.predict(data_test)
    
    Accuracy=accuracy_score(target_test,output) #use to see accuracy or check you data tested:accuracy_score(ANS,check tested ANS) 
    #print(target_train)    #to check the data is same in all function
    return Accuracy
    
def main():
    dataset=load_iris()
    
    data = dataset.data
    target = dataset.target
    
    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)
    #data will go same to both function
    #print(target_train)    #to check the data is same in all function 
    ret=TreeDecision(data_train,data_test,target_train,target_test)
    print("Accuracy of decision tree algorithm is",ret*100,"%")
    
    ret=KNN(data_train,data_test,target_train,target_test)
    print("Accuracy of KNN algorithm is",ret*100,"%")
    
if __name__=="__main__":
    main()
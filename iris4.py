from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

def main():
    dataset = load_iris()
    
    print("Featuers of datasets")
    print(dataset.feature_names)
    
    print("Target name of dataset")
    print(dataset.target_names)
    
    index=[1,2,3,4,5,6,7,8,9,10,51,52,53,54,55,56,57,58,59,60,
            101,102,103,104,105,106,107,108,109,110]
    test_target=dataset.target[index]
    test_feature=dataset.data[index]
    
    train_target=np.delete(dataset.target,index)
    train_feature=np.delete(dataset.data,index,axis=0)
    
    obj=tree.DecisionTreeClassifier()
    
    obj.fit(train_feature,train_target)
    result=obj.predict(test_feature)
    
    print("Result prediction by ML\n",result)
    print("result expected\n",test_target)
    
if __name__=="__main__":
    main()
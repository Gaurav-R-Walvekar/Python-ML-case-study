from sklearn import tree
#Rough 1  #Smooth 0  #Tennis 1  #Cricket 2
def ML(weight,surface): #ML(91,0)
    #step 1 and 2
    Features=[[34,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],
              [35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]#list fo list
    
    Labels=[1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]
    #step 3
    dobj=tree.DecisionTreeClassifier()
    #step 4
    dobj.fit(Features,Labels)#training
    #step 5
    result=dobj.predict([[weight,surface]])#Testing :machine predict values.it will return 1 or 2 label by perdicting.
    #result=dobj.perdict([[91,0],[85,0],[40,1]]) we can test multipal values
    #result->[2,2,1]    need to use for loop to display
    
    if result==1:
        print("Your object looks like tennis ball")
    else:
        print("Your object looks like cirket ball")


def main():
    print("------Supervised Machine Learning-------")
    print("Enter weight of object")
    weight=int(input())
    print("Enter  surface type of object")
    surface=input()
    
    if surface.lower()=="rough":#.lower() :will make input in lower case 
        surface=1                   #eg:Rough  ROUGH rough ->rough
    elif surface.lower()=="smooth":
        surface=0
    else:
        print("Error:wrong input")
        exit()
    
    ML(weight,surface)# MarvellousML(91,0)
    
if __name__=="__main__":
    main()
    
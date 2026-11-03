import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

df=pd.read_csv("Iris - Iris.csv.xls")

df=df.drop('Id', axis=1)

x=df.drop ("Species",axis=1)
y=df["Species"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

IrispreModel=LogisticRegression()
IrispreModel.fit(x_train,y_train)
pred=IrispreModel.predict(x_test)

print('Prediction of first Sample ;' ,pred[0])
accuracy=accuracy_score(pred,y_test)

print("accuracy is:" , accuracy)

with open('Iris_Model.pkl','wb') as file:
    pickle.dump(IrispreModel,file)

print("Model is saved as Iris_Model.pkl")
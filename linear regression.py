import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle 

data=pd.read_csv("student-mat.csv", sep=";")

#print(data.head())
#print("Columns in dataset:", data.columns.tolist())


data = data[["G1","G2", "G3","studytime","failures","absences"]]

#print(data.head())
predict="G3"

x=np.array(data.drop([predict],axis=1))
y=np.array(data[predict])

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)
linear=linear_model.LinearRegression()

linear.fit(x_train,y_train)
acc=linear.score(x_test,y_test)
print(acc)

predictions=linear.predict(x_test)

for x in range(len(predictions)):
  print(predictions[x],x_test[x],y_test[x])
  
  

# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

step 1:Start the program

step 2: Import the libraries and read the data frame using pandas.

step 3: Calculate the null values present in the dataset and apply label encoder.

step 4: Determine test and training data set and apply decison tree regression in dataset.

step 5:calculate Mean square error,data prediction and r2.

step 6:End the program


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Ligneshwar K
RegisterNumber: 212223230113
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
### MSE value
![{AB54BE21-1B30-4ACD-904D-74D880CD8171}](https://github.com/user-attachments/assets/5250bb8c-2c58-43b7-827f-0be0c690b48a)


### r2 Value
![{06567775-EF0A-42D2-A7D3-FC0E8BC4A238}](https://github.com/user-attachments/assets/0f04f6c3-1cd6-454a-bd13-700bf7f1aeae)


### Data Prediction
![{4F46A6B3-0DD9-4B24-A599-9A9719B9C21B}](https://github.com/user-attachments/assets/d5a077ec-6592-4d2e-a8e0-aebed3293a93)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

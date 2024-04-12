# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/c0be1560-2e7d-46ff-a497-aef29aa8a80e)

```
data.isnull().sum()
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/b6615fa6-bca0-439e-93d9-e9bac897a698)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/21a5fac7-9da9-4b74-8d55-f48d3947ef3b)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/e4be12ce-a8b7-495c-ac98-88439c6b1614)
```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/6ed747a8-5ccd-47f4-8677-2e59b8aa450a)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/4817bd9c-ebf6-4850-873f-d3b4ec85b57a)

```
data2
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/fbf36cc7-93d2-47de-b665-8ec09b3cd540)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/70618804-02ea-4244-971e-c0c2d35aba7f)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/ed5ee91b-8bf1-4e8c-a8cf-3d9d96a5589d)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/5ddd2e17-6819-4f8b-acba-90cf4897797a)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/ba4ffd91-efc5-4987-8486-49b620e17f41)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/e0f0561b-97cf-4176-bbb5-19614b52c408)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/b9d1634e-87a4-42f6-89f9-2d05628018c5)
```

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/3f22074f-9d4d-4758-962c-57f12b70146b)
```

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

# RESULT:
       # INCLUDE YOUR RESULT HERE

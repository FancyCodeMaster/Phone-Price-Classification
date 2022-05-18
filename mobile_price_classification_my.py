import numpy as np
import pandas as pd
import joblib

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")


# dataset.info() provides column name along with dtype and null values or not

# Checking whether there are null values in the columns of the train dataset
null_columns = []
for i in train.columns:
    for j in train[i].isnull().value_counts().index.values:
        if j== True:
            null_columns.append(i)
# null columns seem to be empty , no missing  values

# Checking whether categorical values present or not
categorical_columns = []
for i in train.columns:
    if not(train[i].dtype == "float64" or train[i].dtype == "int64"):
        categorical_columns.append(i)
# there are no categorical columns too

# Now applying feature scaling:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train.iloc[: , 0:-1] = scaler.fit_transform(train.iloc[: , 0:-1])

# Now for test dataset applying all the preprocessing
test_null_columns = []
for i in test.columns:
    for j in test[i].isnull().value_counts().index.values:
        if j == True:
            test_null_columns.append(i)
# no null containing columns in the test csv  too

test_categorical_columns = []
for i in test.columns:
    if not(test[i].dtype == "float64" or test[i].dtype == "int64"):
        test_categorical_columns.append(i)

# Removing the id column from the test dataset
test = test.drop(["id"] , axis = 1)

test_scaler = StandardScaler()
test.iloc[: , 0::] = test_scaler.fit_transform(test.iloc[: , 0::])

        
# now converting the data into training and test set
X_train = train.iloc[: , 0:-1]
y_train = train.iloc[: , -1]

# from sklearn.model_selection import train_test_split
# X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.1 , random_state = 0)

# Performing feature selection using mutual info
# from sklearn.feature_selection import mutual_info_classif
# mutual_info = mutual_info_classif(X_train, y_train)
#
#
# mutual_info = pd.Series(mutual_info)
# mutual_info.index = X_train.columns
# mutual_info = mutual_info.sort_values(ascending=False)
#
# from sklearn.feature_selection import SelectKBest
# sel_five_cols = SelectKBest(mutual_info_classif , k = 5)
# sel_five_cols.fit(X_train , y_train)
# X_train.columns[sel_five_cols.get_support()]
#
# X = train[['battery_power' , 'px_height' , 'px_width' , 'ram' , 'sc_w']]

# from sklearn.model_selection import train_test_split
# X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.1 , random_state = 0)


X_test = test.iloc[: , 0::].values


# Using cross validation to try different classification models
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


svc_score = cross_val_score(SVC() , X_train , y_train , cv = 10)
dt_score = cross_val_score(DecisionTreeClassifier() , X_train , y_train , cv = 10)
rf_score = cross_val_score(RandomForestClassifier() , X_train , y_train , cv = 10)
log_score = cross_val_score(LogisticRegression() , X_train , y_train , cv = 10)
knn_score = cross_val_score(KNeighborsClassifier() ,  X_train , y_train , cv = 10)

# Logistic regression shines in this problem set with the accuracy of 96 % 
# But lets check whether some unnncecessary columns can be elmination(Feature Selection) so
# that accuracy would increase

log_reg = LogisticRegression()
log_reg.fit(X_train , y_train)
y_pred = log_reg.predict(X_test)

joblib.dump(log_reg , 'logisticRegression.pkl')

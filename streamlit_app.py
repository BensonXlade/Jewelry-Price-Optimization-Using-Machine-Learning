
##import libraries

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")
##load dataset

df = pd.read_csv("/Users/xlade/Desktop/Amdari/Internship/Project_2_Jewelry_Price_Optimization/Jewelry_Dataset.csv")
#df.head(5)
## Assign column names to the dataset

df.columns = [
    "Order_Datetime",
    "Order_ID",
    "Product_ID",
    'SKU_Quantity',
    'Category_ID',
    'Category',
    'Brand_ID',
    "Price_USD",
    "User_ID",
    "Target_Gender",
    "Main_Color", 
    "Main_metal",
    "Main_Gem"
]
df.head(5)
## check for missing values
df.isnull().sum()
#know the data shape

df.shape
## check fro duplicated rows
df.duplicated().sum()
## check nunique values
df.nunique()
df.info()
df.describe()
## drop duplicated rows
df = df.drop_duplicates()
#check the isnull values again
df.isnull().sum()
## Look through category feature
df["Category"].unique()
##filter out the real categories

real_categories = df["Category"].unique().tolist()
real_categories = [c for c in real_categories if isinstance(c, str) and "jewelry" in c ]
real_categories
df_category = df.loc[df["Category"].isin(real_categories)]
df_category
## haven filtered out the wrong cats check  isnull again
df_category.isnull().sum()
## check Target gender
df_category["Target_Gender"].value_counts()
df_category["Target_Gender"] = df_category["Target_Gender"].fillna(df_category["Target_Gender"].mode()[0])
df_category.head(5)
df_category.isnull().sum()
df_category.shape
## drop the rest of thr null rows

df_category = df_category.dropna()
## confirm the isnull is gone
df_category.isnull().sum()
## confirm the new data size

df_category.shape
## Observe the unique values again

df_category.nunique()
df_category.head(5)
#Univariate Analysis
## category count plot

plt.figure(figsize = (15, 8))

sns.countplot(data = df_category, x = "Category")

plt.xlabel("Jewelry categories")
plt.ylabel("Category frequency")

plt.show(); plt.close()
## visualize target gender

plt.figure(figsize = (10, 5))

sns.countplot(data = df_category, x = "Target_Gender")

plt.xlabel("Gender")
plt.ylabel("Gender Frequency")

plt.tight_layout()
plt.show()
## visualize target gender

plt.figure(figsize = (10, 5))

sns.countplot(data = df_category, x = "Main_Color")

plt.xlabel("Main Colour sold")
plt.ylabel("Main colour Frequency")

plt.tight_layout()
plt.show()
## visualize target gender

plt.figure(figsize = (20, 5))

sns.countplot(data = df_category, x = "Main_Gem")

plt.xlabel("Main Gem sold")
plt.ylabel("Main Gem Frequency")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
#Bivariate Analysis
## category count plot

plt.figure(figsize = (15, 8))

sns.countplot(data = df_category, x = "Category", hue="Target_Gender")

plt.xlabel("Jewelry categories")
plt.ylabel("Category frequency")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show(); plt.close()
#Feature Engineering
## Label Encoding

from sklearn.preprocessing import LabelEncoder
## Encode the categorical variables

le = LabelEncoder()

df_category["Category"] = le.fit_transform(df_category["Category"])
df_category["Target_Gender"] = le.fit_transform(df_category["Target_Gender"])
df_category["Main_Color"] = le.fit_transform(df_category["Main_Color"])
df_category["Main_metal"] = le.fit_transform(df_category["Main_metal"])
df_category["Main_Gem"] = le.fit_transform(df_category["Main_Gem"])
##confirm that the features are now encoded

df_category.head(5)
## feature correlation
correlation = df_category.drop(labels=["Order_Datetime", "SKU_Quantity", "Main_metal"], axis = 1).corr(method = "spearman")
correlation
## correlation heatmap

plt.figure(figsize = (10,10))

sns.heatmap(correlation, annot = True, center = .3)

plt.show(); plt.close()
invariant_column = df_category.nunique()[df_category.nunique() == 1].index.tolist()
invariant_column
## Drop columns with nunique == 1

df_category.drop(labels=invariant_column, axis = 1, inplace=True)
columns_to_eliminate = ["Order_Datetime", "Order_ID", "Product_ID", "Category_ID", "User_ID"]
df_category.drop(labels=columns_to_eliminate, axis = 1, inplace=True)
df_category.head(5)
### Seperate label from the rest of the data

df1 = df_category[["Category", "Brand_ID", "Target_Gender", "Main_Color", "Main_Gem"]]
label = df_category[["Price_USD"]]
df1.head(5)
label.head(5)
#Machine Learning
## import 
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.impute import SimpleImputer


#Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# for data preprocessing
from sklearn.model_selection import train_test_split

# Classifier Libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor

from sklearn.preprocessing import PowerTransformer
import mlflow
import mlflow.sklearn  # Use the relevant MLflow module for your model type
import os
# Set the tracking URI to the new folder inside Jewelry_Price_Optimization
mlflow.set_tracking_uri("file:///Users/xlade/Desktop/Amdari/Internship/Jewelry_Price_Optimization/mlruns_storage")
mlflow.set_experiment("Test_Experiment")
# split the dataset into training and test sets x = questions, Y = Answers

X_train, X_test, y_train, y_test = train_test_split(df1, label, test_size = 0.1, random_state = 42)
# module building

#logistic regression

linreg = LinearRegression()

linreg.fit(X_train, y_train)

lin_y_pred = linreg.predict(X_test)

print("Linear Regression")

print("Mean Absolute Error:", mean_absolute_error(y_test, lin_y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, lin_y_pred))
print("R2 Score:", r2_score(y_test, lin_y_pred))
#print("Accuracy:", accuracy_score(y_test, lin_y_pred))
#print("Precision:", precision_score(y_test, lin_y_pred))
#print("Recall:", recall_score(y_test, lin_y_pred))
#print("F1-score:", f1_score(y_test, lin_y_pred))
#print("AUC-ROC:", roc_auc_score(y_test, lin_y_pred))
# module building

#Adaboost regression

adaboost = AdaBoostRegressor(loss = "exponential", n_estimators=1000, learning_rate=0.1)

adaboost.fit(X_train, y_train)

adaboost_y_pred = adaboost.predict(X_test)

print("Adaboost Regression")

print("Mean Absolute Error:", mean_absolute_error(y_test, adaboost_y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, adaboost_y_pred))
print("R2 Score:", r2_score(y_test, adaboost_y_pred))
# module building

#Catboost regression

catboost = CatBoostRegressor(loss_function = "RMSE", iterations=10000, learning_rate=0.1, one_hot_max_size = 2, depth=6, verbose=0)

catboost.fit(X_train, y_train)

catboost_y_pred = catboost.predict(X_test)

print("Catboost Regression")

print("Mean Absolute Error:", mean_absolute_error(y_test, catboost_y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, catboost_y_pred))
print("R2 Score:", r2_score(y_test, catboost_y_pred))
# module building

#Extra Trees regression

ext = ExtraTreesRegressor(criterion="friedman_mse", n_estimators=1000, bootstrap = True, max_depth=10)

ext.fit(X_train, y_train)

ext_y_pred = ext.predict(X_test)

print("Extra Tree Regressor")

print("Mean Absolute Error:", mean_absolute_error(y_test, ext_y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, ext_y_pred))
print("R2 Score:", r2_score(y_test, ext_y_pred))
 
 
 
 

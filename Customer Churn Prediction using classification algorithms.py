#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

#load dataset
df = pd.read_csv('d:/python_ka_chilla/AI Projects/Customer Churn Prediction using classification algorithms/Churn_Modelling.csv')

# display first few rows of the dataset
print("Customer Churn DataSet:\n", df.head())
#basic data preprocessing
df = df.dropna()  # remove missing values if any

# convert categorical variables to numerical using one-hot encoding
df_encoded = pd.get_dummies(df,  columns=['Geography', 'Gender'], drop_first=True)
print("\nEncoded DataSet:\n", df_encoded.head())

#define features and labels
X = df_encoded.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = df_encoded['Exited']  # target variable

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
#make predictions on the test set
y_pred = model.predict(X_test)
#evaluate the model
print("Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Output:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def data_processing_helper(data_path):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = pd.read_csv(data_path)
    colums = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    df = data[colums] 
    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean) 
    df['Sex'] =df['Sex'].map({'female':0, 'male':1}).astype(int)
    df = pd.get_dummies(data = df, columns=['Embarked'])
    arr = np.array(df)
    arr = scaler.fit_transform(arr)
    return arr

def data_processing_helper_test(data_path):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = pd.read_csv(data_path)
    colums = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    df = data[colums] 
    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean) 
    df['Sex'] =df['Sex'].map({'female':0, 'male':1}).astype(int)
    fare_mean = df['Fare'].mean()
    df['Fare'] = df['Fare'].fillna(fare_mean)
    df = pd.get_dummies(data = df, columns=['Embarked'])
    arr = np.array(df)
    arr = scaler.fit_transform(arr)
    return arr

import pandas as pd

#讀取訓練用CSV
train_data_path = 'dataset/train.csv'
def data_processing_helper(train_data_path):
    #讀取訓練用CSV
    train_data = pd.read_csv(train_data_path)
    #要取用的特徵
    colums = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    train_df = train_data[colums] 
    #取年紀平均數來填補空缺
    age_mean = train_df['Age'].mean()
    train_df['Age'] = train_df['Age'].fillna(age_mean)
    #將性別用'0','1'表示 
    train_df['Sex'] = train_df['Sex'].map({'female':0, 'male':1}).astype(int)
    #將登船口改成數字表示
    train_df = pd.get_dummies(data=train_df, columns=['Embarked'])
    return train_df
def main():
    data_processing_helper(train_data_path)
if __name__ == '__main__':
    main()
    print('資料處理完成')

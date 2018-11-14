import tensorflow as tf
from dataprocessing import data_processing_helper_test
from keras.models import load_model 
import pandas as pd
import numpy as np
test_data_path = 'dataset/test.csv'

test_arr = data_processing_helper_test(test_data_path)
test_data = test_arr[:,:]


model = load_model('model_save')
predic = model.predict_classes(test_data)
prediction = pd.DataFrame(predic,columns = ['Survived'])
submit = pd.concat([test_data,prediction],axis=1, join_axes=[test_df.index])
submit_col = ['PassengerId','Survived']
submit = submit[submit_col]
submit.to_csv('dataset/submit.csv',index=False)
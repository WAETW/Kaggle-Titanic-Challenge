import tensorflow as tf
from dataprocessing import data_processing_helper
from keras.models import Sequential  
from keras.layers import Dense,Dropout 

train_data_path = 'dataset/train.csv'

train_arr = data_processing_helper(train_data_path)
train_data = train_arr[:,1:]
train_label = train_arr[:,0]  
model = Sequential()  
model.add(Dense(units=40, input_dim=9, kernel_initializer='uniform', activation='relu'))  
model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))  
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))  
model.summary()  

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
train = model.fit(x=train_data, y=train_label, validation_split=0.2, epochs=500, batch_size=30, verbose=2)

model.save('model_save')
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder



data=pd.read_csv("churn.csv")
data=shuffle(data)
data.isna().sum


X=data.drop(['RowNumber','CustomerId','Surname','Exited'],axis=1)
y=data['Exited']


X = pd.get_dummies(X , prefix='Geography' , columns= ['Geography'] , drop_first=True )
X = pd.get_dummies(X , prefix='Gender' , columns= ['Gender'] , drop_first=True )



scalar=StandardScaler()
X=scalar.fit_transform(X)


X_train , X_test , y_train , y_test = train_test_split(X, y , train_size=0.3 , random_state=0 ,shuffle=True )




model = tf.keras.models.Sequential([Dense(256, activation= 'relu' , input_shape = (None,3000,11)) , 
                                    Dense(128,activation='relu') , Dense(64,activation='relu') ,
                                    Dense(32,activation='sigmoid') , Dense(1,activation='sigmoid')])
model.build()  


model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'] ) 
model.fit(X_train, y_train, verbose = 1, epochs = 25, batch_size = 32, validation_data = (X_test , y_test))
















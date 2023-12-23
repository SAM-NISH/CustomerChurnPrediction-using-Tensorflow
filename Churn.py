import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
# ... (rest of your imports)

data = pd.read_csv("churn.csv")
data = shuffle(data)

X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = data['Exited']

X = pd.get_dummies(X, prefix='Geography', columns=['Geography'], drop_first=True)
X = pd.get_dummies(X, prefix='Gender', columns=['Gender'], drop_first=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0, shuffle=True)

model = tf.keras.models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
model.fit(X_train, y_train, verbose=1, epochs=25, batch_size=32, validation_data=(X_test, y_test))

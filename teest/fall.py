import numpy as np
from sklearn.model_selection import train_test_split
import keras

from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense

x=[]
y=[]
file_ADL='ADL_CNN.txt' # contains all the ADLs data windows (length of the window 401*3)
file_FALL='FALL_CNN.txt' # contains all the Falls data windows (length of the window 401*3)
data_ADL = np.loadtxt(file_ADL, dtype=float, delimiter=',')
data_FALL = np.loadtxt(file_FALL, dtype=float, delimiter=',')
for i in range(len(data_FALL),len(data_ADL)):
        x.append(data_ADL[i])
        y.append(0)
for i in range(len(data_FALL)):
    x.append(data_ADL[i])
    y.append(0)
    x.append(data_FALL[i])
    y.append(1)


#%%

labels=['ADL','FALL'] # labels : ADL=0; Fall=1

#%%

X= np.array(x).reshape(len(x), 401, 3)
Y = np.array(y)

print(X.shape)
print(Y.shape)


#split the data to 60% training, 20% test and 20% validation
X_train, X_t, y_train, y_t = train_test_split(X, Y, test_size = 0.4, random_state =42, stratify = Y)
X_test, X_val, y_test, y_val = train_test_split(X_t, y_t, test_size=0.5, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
print(y_test.shape)

model = keras.Sequential()
#1st convolution layer
model.add(Conv1D(16, 3,activation='relu',padding='same',input_shape=(401,3)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3,padding='same'))

#2nd convolution layer
model.add(Conv1D(32, 3,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3,padding='same'))

#3rd convolution layer
model.add(Conv1D(64,3,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3,padding='same'))


#4th convolution layer
model.add(Conv1D(128,3,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3,padding='same'))
model.add(Dropout(0.5))

#fully connected layer
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))


#%%

print(model.summary())
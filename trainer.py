import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.utils import to_categorical

def load_data():
		dataframe = pd.read_csv('emotions_df.csv')
		return dataframe
def train_data(dataframe):

		x_train = dataframe.iloc[:,:-1].values
		x_train = x_train.reshape(35000,48,48,1)
		y_train = dataframe.iloc[:,-1].values
		y_train = to_categorical(y_train)

		model = Sequential() 
 
		#1st convolution layer
		model.add(Convolution2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
		model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
 
		#2nd convolution layer
		model.add(Convolution2D(64, (3, 3), activation='relu'))
		model.add(Convolution2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
 
		#3rd convolution layer
		model.add(Convolution2D(128, (3, 3), activation='relu'))
		model.add(Convolution2D(128, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
 
		model.add(Flatten())
 
		#fully connected neural networks
		model.add(Dense(1024, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(1024, activation='relu'))
		model.add(Dropout(0.2))
 
		model.add(Dense(7, activation='softmax'))

		model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

		model.fit(x_train,y_train,epochs=30,batch_size=500,validation_split=0.2)

		model.save('emotion_trainer.h5')


dataframe = load_data()
train_data(dataframe)
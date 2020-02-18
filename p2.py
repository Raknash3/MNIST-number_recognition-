from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
a=tf.keras.datasets.mnist
(img_train,label_train),(img_test,label_test)=a.load_data()
img_train,img_test=img_train/255.0,img_test/255.0
model=keras.Sequential([        
        keras.layers.Flatten(input_shape=(28,28)), #flatten layer
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),# three hidden layer with 128 nodes and relu as activation function
        keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2), #20% of the nodes are dropped in each rotation
        keras.layers.Dense(10,activation='softmax')# final layer with ten nodes coressponding to the labels
        ])
#specify optimiser, loss function and metric to track
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#train the model
model.fit(img_train,label_train,epochs=15) #default batch size=32
#evaluate the accuracy
test_loss,test_acc= model.evaluate(img_test,label_test)
print(test_acc)
#make predictions
p=model.predict(img_test)
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation,Flatten,Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn import datasets
from keras import backend as k

k.set_image_dim_ordering("th")

num_classes=10
img_depth=1
img_height=28
img_width=28

model = Sequential()

# add conv layer 1
model.add(Conv2D(20,(5,5),border_mode="same",input_shape=(img_depth,img_height,img_width)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


# add conv layer2
model.add(Conv2D(50,(5,5),border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


# add Fully Connected Layer
model.add(Flatten())
model.add(Dense(500))

model.add(Activation("relu"))

# add SoftMax Layer
model.add(Dense(num_classes))
model.add(Activation("softmax"))
mnist = datasets.fetch_mldata('MNIST original',data_home='./')


# MNIST data is 784 size
# should transform into 28x28 image to transfer conv layers
mnist.data = mnist.data.reshape((mnist.data.shape[0],28,28))

mnist.data = mnist.data[:,np.newaxis,:,:]
mnist.data = mnist.data/255.0

train_data,test_data,train_label,test_label = train_test_split(mnist.data,mnist.target,test_size=0.25)

train_label = np_utils.to_categorical(train_label,10)
test_label = np_utils.to_categorical(test_label,10)

model.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.0001),metrics=["accuracy"])

# train model
model.fit(train_data,train_label,batch_size=32,epochs=30,verbose=1)

# test model
loss,accuracy = model.evaluate(test_data,test_label,batch_size=64,verbose=1)
print("Accuracy: %",format(accuracy*100))


import numpy as np
import tensorflow as tf
from tensorflow import keras
import skimage as ski
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

IMG_SIZE = 224

def processed_data (X, y):
    
    for i in range(len(X)):
        X[i] = ski.exposure.equalize_adapthist(X[i])
        X[i]=tf.image.resize(X[i],[IMG_SIZE, IMG_SIZE])

    resize_and_rescale = tf.keras.Sequential([
        keras.layers.Rescaling(1./255)
    ])
    data_augmentation = tf.keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
    ])
    X=resize_and_rescale(X)
    X=data_augmentation(X)
    X=np.array(X)
    y=np.eye(8)[y]

    return X,y

def load_data(filex,filey):
    count = 1
    X=[]
    while(count<=408):  #!408 is the number of training data
        X.append(plt.imread(filex+'/%d.jpg'%count))
        count+=1
    y=np.load(filey)
    return X,y

def load_and_process_data(file_x, file_y):
    X,y = load_data(file_x,file_y)
    X,y= processed_data(X,y)
    return X,y

from sklearn.model_selection import train_test_split


train_x, train_y = load_and_process_data('Face_recognition/train_img','Face_recognition/train_name.npy')
# X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size = 0.2)

IMG_shape = [IMG_SIZE,IMG_SIZE,3]
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_shape,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = True
fine_tune=100
for layer in base_model.layers[:fine_tune]:
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,  # 1
    tf.keras.layers.Conv2D(32, 3, activation='relu'),  # 2
    tf.keras.layers.Dropout(0.2),  # 3
    tf.keras.layers.GlobalAveragePooling2D(),  # 4
    tf.keras.layers.Dense(8, activation='softmax')  # 5
])

model.compile(optimizer=tf.keras.optimizers.Adam(),  # 1
              loss='categorical_crossentropy',  # 2
              metrics=['accuracy'])  # 3

model.summary()

# Train the model
# We will do it in 10 Iterations
epochs = 10

# Fitting / Training the model
history = model.fit(x=train_x,
                    y=train_y,
                    batch_size=10,
                    epochs=epochs,
                    validation_split=0.1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


        

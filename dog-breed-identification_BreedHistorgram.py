#TechVidvan load all required libraries
import cv2
import numpy as np 
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout,BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet50V2,preprocess_input
import torch
from torch import nn

import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import classification_report
import playsound
from datetime import datetime
import time
start_time = time.time()
currTime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
currTime2 = datetime.now().strftime("%Y%m%d")

f2=open(f"LogTest_{currTime2}.txt", "a", encoding='utf8')        #open txt file

f2.write("<--------------"+currTime+"------------------>\n")

#specify number
#num_breeds = 120
im_size = 224      #original 224
batch_size = 32                   #original 64                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
encoder = LabelEncoder()

#read the csv file
df_labels = pd.read_csv("labels.csv")
#store training and testing images folder location
train_file = 'train/'
test_file = 'test/'

#check the total number of unique breed in our dataset file
print("Total number of unique Dog Breeds :",len(df_labels.breed.unique()))
num_breeds = len(df_labels.breed.unique())

#get only 60 unique breeds record
breed_dict = list(df_labels['breed'].value_counts().keys()) 
#print (breed_dict)
#new_list = sorted(breed_dict,reverse=True)[:num_breeds*2+1:2]
new_list = sorted(breed_dict,reverse=True)
#change the dataset to have only those 60 unique breed records
df_labels = df_labels.query('breed in @new_list')
#create new column which will contain image name with the image extension
df_labels['img_file'] = df_labels['id'].apply(lambda x: x + ".jpg")

#create a numpy array of the shape
#(number of dataset records, image size , image size, 3 for rgb channel ayer)
#this will be input for model
#train_x = np.zeros((len(df_labels), im_size, im_size, 3), dtype='float32')
train_x = np.zeros((len(df_labels),im_size, im_size,3), dtype='float32')

#iterate over img_file column of our dataset
for i, img_id in enumerate(df_labels['img_file']):
  #read the image file and convert into numeric format
  #resize all images to one dimension i.e. 224x224
  #we will get array with the shape of
  #(224,224,3) where 3 is the RGB channels layers
  img = cv2.resize(cv2.imread(train_file+img_id,cv2.IMREAD_COLOR),((im_size,im_size)))    #cv2.IMREAD_GRAYSCALE , cv2.IMREAD_COLOR!!!!!!!!!!!!!!!!!!!!!!
  #scale array into the range of -1 to 1.
  #preprocess the array and expand its dimension on the axis 0 
  img_array = preprocess_input(np.expand_dims(np.array(img[...,::-1].astype(np.float32)).copy(), axis=0))
  #update the train_x variable with new element
  train_x[i] = img_array

#this will be target for model.
#convert breed names into numerical format
train_y = encoder.fit_transform(df_labels["breed"].values)

#from sklearn.preprocessing import LabelEncoder
plt.hist(train_y, 120)
plt.show()


#split the dataset in the ratio of 80:20. 
#80% for training and 20% for testing purpose
#x_train, x_test, y_train, y_test = train_test_split(train_x,train_y,test_size=0.2,random_state=42)
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)       #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

## this function plot the breeds distribution in train data 
"""def dist_breed(labels):
    encoder = LabelEncoder()
    breeds_encoded = encoder.fit_transform(labels)
    n_classes = len(encoder.classes_)
    
    breeds = pd.DataFrame(np.array(breeds_encoded), columns=["breed"]).reset_index(drop=True)
    breeds['freq'] = breeds.groupby('breed')['breed'].transform('count')
    avg = breeds.freq.mean()
    
    title = 'Distribution of Dog Breeds in training Dataset\n (%3.0f samples per class on average)' % avg
    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xticks([])
    
    ax.hlines(avg, 0, n_classes - 1, color='white')
    ax.set_title(title, fontsize=18)
    _ = ax.hist(breeds_encoded, bins=n_classes)
    
    return(breeds["freq"].describe())
dist_breed(y_train)"""

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42) # 0.25 x 0.8 = 0.2   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#print("y_test label")
#print(y_test)
#print(x_train[0].shape)
#print(type(x_train[0]))

"""train_set=[]
test_set=[]
for i in range(len(x_train)):
    numpy2tensor=torch.tensor(x_train[i])
    train_set.append([numpy2tensor, y_train[i]])
for i in range(len(x_test)):
    numpy2tensor=torch.tensor(x_test[i])
    test_set.append([numpy2tensor, y_test[i]])
#train_set=np.array(list(zip(x_train, y_train)))
#test_set=np.array(list(zip(x_test, y_test)))

conv_layer = nn.Conv2d(in_channels=1, out_channels=5, stride=2, kernel_size=3)
conv_layer.weight.shape
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size, 
                                          shuffle=False)"""
#print(type(train_loader))
#print(x_train.shape())

num_classes = num_breeds
input_shape = (224, 224, 1)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

print(model.summary())
#batch_size = 64
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])




"""#building the model using ResNet50V2 with input shape of our image array
#weights for our network will be from of imagenet dataset
#we will not include the first Dense layer
resnet = ResNet50V2(input_shape = [im_size,im_size,3], weights='imagenet', include_top=False)
#freeze all trainable layers and train only top layers 
for layer in resnet.layers:
    layer.trainable = False

#add global average pooling layer and Batch Normalization layer
x = resnet.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
#add fully connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

#add output layer having the shape equal to number of breeds
predictions = Dense(num_breeds, activation='softmax')(x)

#create model class with inputs and outputs
model = Model(inputs=resnet.input, outputs=predictions)
#model.summary()

#epochs for model training and learning rate for optimizer
epochs = 20 ##original 20         #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
learning_rate = 1e-3            #original 1e-3  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#using RMSprop optimizer compile or build the model
optimizer = RMSprop(learning_rate=learning_rate,rho=0.9)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

#fit the training generator data and train the model
model.fit(train_generator,
                 steps_per_epoch= x_train.shape[0] // batch_size,
                 epochs= epochs,
                 validation_data= val_generator,
                 validation_steps= x_val.shape[0] // batch_size)

#Save the model for prediction
model.save("model")

#load the model
model = load_model("model")

all_preds = []
for test_img in x_test:
    
  #feed the model with the image array for prediction
  test_img = test_img.reshape(1,224,224,3)
  pred_val = model.predict(np.array(test_img,dtype="float32"))

  #display the predicted breed of dog
  pred_breed = sorted(new_list)[np.argmax(pred_val)]
  #print("Predicted Breed for this Dog is :",pred_breed)
  #print(f"breed number: {np.argmax(pred_val)}")
  all_preds .append(np.argmax(pred_val))
#print (all_preds)"""


"""k=[]
for i in range (0,len(list(encoder.classes_))):
    k.append(i)
classes = list(encoder.inverse_transform(k))
breed_dict2 = dict(zip(k, classes))
#runtime=str(datetime.timedelta(seconds = (time.time() - start_time)))
runtime=time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
print(f"length(number) of label in train set: {len(list(encoder.classes_))}")
#print (breed_dict2)
print("I haven't use original test data, only split train set data to train, validation and test.")
print(f"length(number) of train set data: {len(y_train)}")
#print(f"length(number) of validation set data: {len(y_val)}")
print(f"length(number) of test set data: {len(y_test)}")
print(f"Epoch= {num_epochs}, Learning rate= {learning_rate}, batch_size: {batch_size}")
#print(type(y_test))
ytestlist=y_test.tolist()

Y_true =[]
Y_pred_classes =[]
for i in range(len(all_labels)):
  Y_true .append(all_labels[i].cpu().numpy())
for i in range(len(all_preds)):
  Y_pred_classes .append(all_preds[i].cpu().numpy())
#print (all_labels)
#print (Y_true )
#print (all_preds)
#print (Y_pred_classes )
#classes = ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
print(classification_report(Y_true, Y_pred_classes, target_names = classes))

#print(classification_report(ytestlist, all_preds, target_names = classes))
print(f"---runtime= {runtime} ---")
f2.write(f"This is result of using HW4a model\n")
f2.write(f"length(number) of label in train set: {len(list(encoder.classes_))}\n")
#f2.write(str(breed_dict2))
f2.write("\nI haven't use original test data, only split train set data to train, validation and test.\n")
f2.write(f"\nlength(number) of train set data: {len(y_train)}\n")
#f2.write(f"length(number) of validation set data: {len(y_val)}\n")
f2.write(f"length(number) of test set data: {len(y_test)}\n")
f2.write(f"Epoch= {num_epochs}, Learning rate= {learning_rate}, batch_size: {batch_size}\n")
#f2.write(str(classification_report(y_test, all_preds, target_names = classes)))
f2.write(str(classification_report(Y_true, Y_pred_classes, target_names = classes)))
f2.write(f"\n---runtime= {runtime} ---\n")
f2.write("\n<------------------------End------------------------>\n\n\n")
f2.close() 

time.sleep(5)
playsound.playsound('D:/Dpath/music/disney;dreamwork/the little mermaid Part of Your World 12 Disney Piano Collection by Hirohashi Makiko.mp3', True)"""
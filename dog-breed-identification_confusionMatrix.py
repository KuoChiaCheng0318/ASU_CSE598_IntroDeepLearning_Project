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
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import classification_report, confusion_matrix
import playsound
from datetime import datetime
import time
from PIL import Image
start_time = time.time()
currTime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
currTime2 = datetime.now().strftime("%Y%m%d")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

f2=open(f"Log_{currTime2}.txt", "a", encoding='utf8')        #open txt file

f2.write("<--------------"+currTime+"------------------>\n")

#specify number
#num_breeds = 120
im_size = 224
batch_size = 64                   #original 64                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
train_x = np.zeros((len(df_labels), im_size, im_size, 3), dtype='float32')    #!!!!!!!!!if RGB:  im_size, im_size, 3
#train_x = np.zeros((len(df_labels), 1, im_size, im_size), dtype='float32')    # #!!!!!!!!!if gray scale:  im_size, im_size

#iterate over img_file column of our dataset
for i, img_id in enumerate(df_labels['img_file']):
  #read the image file and convert into numeric format
  #resize all images to one dimension i.e. 224x224
  #we will get array with the shape of
  #(224,224,3) where 3 is the RGB channels layers
  img = cv2.resize(cv2.imread(train_file+img_id,cv2.IMREAD_COLOR),((im_size,im_size)))      #cv2.IMREAD_COLOR!!!!!!!!!!!!!!!!!!!!!!
  #scale array into the range of -1 to 1.
  #preprocess the array and expand its dimension on the axis 0 
  img_array = preprocess_input(np.expand_dims(np.array(img[...,::-1].astype(np.float32)).copy(), axis=0))
  #update the train_x variable with new element
  train_x[i] = img_array


#this will be target for model.
#convert breed names into numerical format
train_y = encoder.fit_transform(df_labels["breed"].values)

#split the dataset in the ratio of 80:20. 
#80% for training and 20% for testing purpose
#x_train, x_test, y_train, y_test = train_test_split(train_x,train_y,test_size=0.2,random_state=42)
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)       #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#print("y_test label")
#print(y_test)


#Image augmentation using ImageDataGenerator class
train_datagen = ImageDataGenerator(rotation_range=45,           #original !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.25,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
"""train_datagen = ImageDataGenerator(rotation_range=180,           #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   
                                   width_shift_range=0.5,
                                   height_shift_range=0.5,
                                   shear_range=0.5,
                                   zoom_range=0.5,
                                   horizontal_flip=True,
                                   fill_mode='nearest')    """                               

#generate images for training sets 
train_generator = train_datagen.flow(x_train, 
                                     y_train, 
                                     batch_size=batch_size)

"""#same process for Testing sets also by declaring the instance

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow(x_test, 
                                     y_test, 
                                     batch_size=batch_size)"""
#same process for validation sets also by declaring the instance
val_datagen = ImageDataGenerator()

val_generator = val_datagen.flow(x_val, 
                                     y_val, 
                                     batch_size=batch_size)


#building the model using ResNet50V2 with input shape of our image array
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
epochs =  20 ##original 20         #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
#print (all_preds)


k=[]
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
print(f"length(number) of validation set data: {len(y_val)}")
print(f"length(number) of test set data: {len(y_test)}")
print(f"Epoch= {epochs}, Learning rate= {learning_rate}, batch_size: {batch_size}")
print(classification_report(y_test, all_preds, target_names = classes))
confusion_mtx = confusion_matrix(y_test , all_preds )
plot_confusion_matrix(confusion_mtx, classes)
print(f"---runtime= {runtime} ---")
f2.write(f"length(number) of label in train set: {len(list(encoder.classes_))}\n")
#f2.write(str(breed_dict2))
f2.write("\nI haven't use original test data, only split train set data to train, validation and test.\n")
f2.write(f"\nlength(number) of train set data: {len(y_train)}\n")
f2.write(f"length(number) of validation set data: {len(y_val)}\n")
f2.write(f"length(number) of test set data: {len(y_test)}\n")
f2.write(f"Epoch= {epochs}, Learning rate= {learning_rate}, batch_size: {batch_size}\n")
f2.write(str(classification_report(y_test, all_preds, target_names = classes)))
f2.write(f"\n---runtime= {runtime} ---\n")
f2.write("\n<------------------------End------------------------>\n\n\n")
f2.close() 

time.sleep(5)
playsound.playsound('D:/Dpath/music/disney;dreamwork/the little mermaid Part of Your World 12 Disney Piano Collection by Hirohashi Makiko.mp3', True)
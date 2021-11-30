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

from sklearn.metrics import classification_report
import playsound
from datetime import datetime
import time

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F

start_time = time.time()
currTime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
currTime2 = datetime.now().strftime("%Y%m%d")

f2=open(f"LogHW4a_{currTime2}.txt", "a", encoding='utf8')        #open txt file

f2.write("<--------------"+currTime+"------------------>\n")

#specify number
#num_breeds = 120
im_size = 128       #original 224
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
train_x = np.zeros((len(df_labels),1,im_size, im_size), dtype='float32')

#iterate over img_file column of our dataset
for i, img_id in enumerate(df_labels['img_file']):
    #read the image file and convert into numeric format
    #resize all images to one dimension i.e. 224x224
    #we will get array with the shape of
    #(224,224,3) where 3 is the RGB channels layers
    img = cv2.resize(cv2.imread(train_file+img_id,cv2.IMREAD_GRAYSCALE),((im_size,im_size)))    #cv2.IMREAD_GRAYSCALE , cv2.IMREAD_COLOR!!!!!!!!!!!!!!!!!!!!!!
    #scale array into the range of -1 to 1.
    #preprocess the array and expand its dimension on the axis 0 
    img_array = preprocess_input(np.expand_dims(np.array(img[...,::-1].astype(np.float32)).copy(), axis=0))
    #update the train_x variable with new element
    train_x[i] = img_array


#this will be target for model.
#convert breed names into numerical format
train_y = encoder.fit_transform(df_labels["breed"].values)
"""plt.hist(train_y, 120)
plt.xlabel('breed')
plt.ylabel('count')
plt.show()"""

#split the dataset in the ratio of 80:20. 
#80% for training and 20% for testing purpose
#x_train, x_test, y_train, y_test = train_test_split(train_x,train_y,test_size=0.2,random_state=42)
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)       #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42) # 0.25 x 0.8 = 0.2   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#print("y_test label")
#print(y_test)
#print(x_train[0].shape)
#print(type(x_train[0]))

train_set=[]
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
                                          shuffle=False)
#print(type(train_loader))


                                     
device = 'cuda'
num_epochs = 20
num_classes = num_breeds
learning_rate = 0.001
#gradient_accumulations = 16

# A simple convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        # In the init function, we define each layer we will use in our model
        # Our images are RGB, so we have input channels = 3. 
        # We will apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        # A second convolutional layer takes 12 input channels, and generates 24 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        # We in the end apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)
        # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
        # This means that our feature tensors are now 32 x 32, and we've generated 24 of them
        # We need to flatten these in order to feed them to a fully-connected layer
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        # In the forward function, pass the data through the layers we defined in the init function
        # Use a ReLU activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x))) 
        # Use a ReLU activation function after layer 2
        x = F.relu(self.pool(self.conv2(x)))  
        # Select some features to drop to prevent overfitting (only drop during training)
        x = F.dropout(self.drop(x), training=self.training)
        # Flatten
        x = x.view(-1, 32 * 32 * 24)
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        # Return class probabilities via a log_softmax function 
        return torch.log_softmax(x, dim=1)

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass

        outputs = model(images)
        #print(outputs)
        #print(type(labels))
        #print(outputs.size())
        #print(labels.size())
        loss = criterion(outputs, labels.long())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

with torch.no_grad():
    all_labels = []
    all_preds = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels)
        all_preds.extend(predicted)
    ##

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
playsound.playsound('D:/Dpath/music/disney;dreamwork/the little mermaid Part of Your World 12 Disney Piano Collection by Hirohashi Makiko.mp3', True)
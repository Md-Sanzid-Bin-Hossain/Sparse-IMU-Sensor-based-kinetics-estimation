
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import numpy
import statistics 
from numpy import loadtxt
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statistics import stdev 
import math
import h5py
 
import numpy as np
import time

from scipy.signal import butter,filtfilt
import sys 
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
import pandas
import matplotlib.pyplot as plt

# from tsf.model import TransformerForecaster


# from tensorflow.keras.utils import np_utils
import itertools
###  Library for attention layers 
import pandas as pd
import os 
import numpy as np
#from tqdm import tqdm # Processing time measurement
from sklearn.model_selection import train_test_split 

import statistics
import gc
import torch.nn.init as init

############################################################################################################################################################################
############################################################################################################################################################################

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.utils.weight_norm as weight_norm
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torchsummary import summary
from torch.nn.parameter import Parameter


import torch.optim as optim
import gc

from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler



"""# File path

# Data loader
"""

def data_loader(subject):
  with h5py.File('/home/sanzidpr/Dataset A-Kinetics/All_subjects_data.h5', 'r') as hf:
    All_subjects = hf['All_subjects']
    Subject = All_subjects[subject]

    treadmill = Subject['Treadmill']
    levelground = Subject['Levelground']
    ramp = Subject['Ramp']
    stair = Subject['Stair']
    
    All_data=np.concatenate((treadmill,levelground,ramp,stair),axis=0)

    return np.array(All_data)

subject_7_data=data_loader('Subject_7')
gc.collect()
subject_8_data=data_loader('Subject_8')
gc.collect()
subject_9_data=data_loader('Subject_9')
gc.collect()
subject_10_data=data_loader('Subject_10')
gc.collect()
subject_11_data=data_loader('Subject_11')
gc.collect()
subject_12_data=data_loader('Subject_12')
gc.collect()
subject_13_data=data_loader('Subject_13')
gc.collect()
subject_14_data=data_loader('Subject_14')
gc.collect()
subject_15_data=data_loader('Subject_15')
gc.collect()
subject_16_data=data_loader('Subject_16')
gc.collect()
subject_17_data=data_loader('Subject_17')
gc.collect()
subject_18_data=data_loader('Subject_18')
gc.collect()
subject_19_data=data_loader('Subject_19')
gc.collect()
subject_21_data=data_loader('Subject_21')
gc.collect()
subject_23_data=data_loader('Subject_23')
gc.collect()
subject_24_data=data_loader('Subject_24')
gc.collect()
subject_25_data=data_loader('Subject_25')
gc.collect()
subject_27_data=data_loader('Subject_27')
gc.collect()
subject_28_data=data_loader('Subject_28')
gc.collect()
subject_30_data=data_loader('Subject_30')
gc.collect()

gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()

##############################################################################################################################################################################################################



"""# Data processing"""




main_dir = "/home/sanzidpr/Journal_3/Dataset_A_model_results_IMU_emg/Subject30"
os.mkdir(main_dir) 
path="/home/sanzidpr/Journal_3/Dataset_A_model_results_IMU_emg/Subject30/"
subject='Subject_30'

train_dataset=np.concatenate((subject_7_data,subject_8_data,subject_9_data,subject_10_data,subject_11_data,
                               subject_12_data,subject_13_data,subject_14_data,subject_15_data,subject_16_data,subject_17_data,
                               subject_18_data,subject_19_data,subject_21_data,subject_23_data,subject_24_data,subject_25_data,subject_27_data,subject_28_data),axis=0)


test_dataset=subject_30_data








##############################################################################################################################################################################################################


encoder='lstm'

# Train features #



## IMUs-0:24
## IK-24:47
## ID-47:70
## GRF-70:79
## GON-79:84
## EMG-84:106
## JP-106:129


x_train_IMUs=train_dataset[:,0:24]
x_train_Kinematics=train_dataset[:,24:47]
x_train_kinetics=train_dataset[:,47:70]
x_train_GRF=train_dataset[:,70:79]
x_train_GON=train_dataset[:,79:84]
x_train_EMG=train_dataset[:,95:106]
x_train_JP=train_dataset[:,106:129]

x_train_Kinematics=np.concatenate((x_train_Kinematics[:,6:12],x_train_Kinematics[:,13:19]),axis=1)
x_train_JP=np.concatenate((x_train_JP[:,6:9],x_train_JP[:,15:16],x_train_JP[:,17:18],x_train_JP[:,19:20],x_train_JP[:,21:22]),axis=1)

x_train_kinetics=np.concatenate((x_train_kinetics[:,6:8],x_train_kinetics[:,15:16],x_train_kinetics[:,17:18]),axis=1)

x_train=np.concatenate((x_train_IMUs,x_train_Kinematics,x_train_EMG,x_train_JP,x_train_GON,x_train_kinetics,x_train_GRF[:,0:3]),axis=1)



train_X_1_1=x_train
 
# # Test features #
x_test_IMUs=test_dataset[:,0:24]
x_test_Kinematics=test_dataset[:,24:47]
x_test_kinetics=test_dataset[:,47:70]
x_test_GRF=test_dataset[:,70:79]
x_test_GON=test_dataset[:,79:84]
x_test_EMG=test_dataset[:,95:106]
x_test_JP=test_dataset[:,106:129]

x_test_Kinematics=np.concatenate((x_test_Kinematics[:,6:12],x_test_Kinematics[:,13:19]),axis=1)
x_test_JP=np.concatenate((x_test_JP[:,6:9],x_test_JP[:,15:16],x_test_JP[:,17:18],x_test_JP[:,19:20],x_test_JP[:,21:22]),axis=1)

print(x_test_Kinematics.shape)
print(x_test_JP.shape)

x_test_kinetics=np.concatenate((x_test_kinetics[:,6:8],x_test_kinetics[:,15:16],x_test_kinetics[:,17:18]),axis=1)

x_test=np.concatenate((x_test_IMUs,x_test_Kinematics,x_test_EMG,x_test_JP,x_test_GON,x_test_kinetics,x_test_GRF[:,0:3]),axis=1)


test_X_1_1=x_test

print(x_test.shape)



m1=59
m2=66



  ### Label ###

train_y_1_1=train_dataset[:,m1:m2]
test_y_1_1=test_dataset[:,m1:m2]

train_dataset_1=np.concatenate((train_X_1_1,train_y_1_1),axis=1)
test_dataset_1=np.concatenate((test_X_1_1,test_y_1_1),axis=1)

train_dataset_1=pd.DataFrame(train_dataset_1)
test_dataset_1=pd.DataFrame(test_dataset_1)

train_dataset_1.dropna(axis=0,inplace=True)
test_dataset_1.dropna(axis=0,inplace=True)

train_dataset_1=np.array(train_dataset_1)
test_dataset_1=np.array(test_dataset_1)

train_dataset_sum = np. sum(train_dataset_1)
array_has_nan = np. isinf(train_dataset_1[:,0:59])

print(array_has_nan)

print(train_dataset_1.shape)



train_X_1=train_dataset_1[:,0:m1]
test_X_1=test_dataset_1[:,0:m1]

train_y_1=train_dataset_1[:,m1:m1+7]
test_y_1=test_dataset_1[:,m1:m1+7]



L1=len(train_X_1)
L2=len(test_X_1)

print(L1+L2)
 
w=100

                   
 
a1=L1//w
b1=L1%w
 
a2=L2//w
b2=L2%w

# a3=L3//w
# b3=L3%w 
 
     #### Features ####
train_X_2=train_X_1[L1-w+b1:L1,:]
test_X_2=test_X_1[L2-w+b2:L2,:]
# validation_X_2=validation_X_1[L3-w+b3:L3,:]
 

    #### Output ####
 
train_y_2=train_y_1[L1-w+b1:L1,:]
test_y_2=test_y_1[L2-w+b2:L2,:]
# validation_y_2=validation_y_1[L3-w+b3:L3,:]


 
     #### Features ####
    
train_X=np.concatenate((train_X_1,train_X_2),axis=0)
test_X=np.concatenate((test_X_1,test_X_2),axis=0)
# validation_X=np.concatenate((validation_X_1,validation_X_2),axis=0)
 
 
    #### Output ####
    
train_y=np.concatenate((train_y_1,train_y_2),axis=0)
test_y=np.concatenate((test_y_1,test_y_2),axis=0)
# validation_y=np.concatenate((validation_y_1,validation_y_2),axis=0)

    
print(train_y.shape) 
    #### Reshaping ####
train_X_3_p= train_X.reshape((a1+1,w,train_X.shape[1]))
test_X = test_X.reshape((a2+1,w,test_X.shape[1]))

output_dim=7


train_y_3_p= train_y.reshape((a1+1,w,output_dim))
test_y= test_y.reshape((a2+1,w,output_dim))

 

# train_X_1D=train_X_3
test_X_1D=test_X

train_X_3=train_X_3_p
train_y_3=train_y_3_p
# print(train_X_4.shape,train_y_3.shape)


train_X_1D, X_validation_1D, train_y_5, Y_validation = train_test_split(train_X_3,train_y_3, test_size=0.20, random_state=True)
#train_X_1D, X_validation_1D_ridge, train_y, Y_validation_ridge = train_test_split(train_X_1D_m,train_y_m, test_size=0.10, random_state=True)   [0:2668,:,:]

print(train_X_1D.shape,train_y_5.shape,X_validation_1D.shape,Y_validation.shape)



Bag_samples=train_X_1D.shape[0]
print(Bag_samples)

s=test_X_1D.shape[0]*w

gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()




### Data Processing

batch_size = 64

val_targets = torch.Tensor(Y_validation)
test_features = torch.Tensor(test_X_1D)
test_targets = torch.Tensor(test_y)


## all Modality Features

train_features = torch.Tensor(train_X_1D)
train_targets = torch.Tensor(train_y_5)
val_features = torch.Tensor(X_validation_1D)


train_features_acc_4=torch.cat((train_features[:,:,0:3],train_features[:,:,6:9],train_features[:,:,12:15],train_features[:,:,18:21]),axis=-1)
test_features_acc_4=torch.cat((test_features[:,:,0:3],test_features[:,:,6:9],test_features[:,:,12:15],test_features[:,:,18:21]),axis=-1)
val_features_acc_4=torch.cat((val_features[:,:,0:3],val_features[:,:,6:9],val_features[:,:,12:15],val_features[:,:,18:21]),axis=-1)


train_features_gyr_4=torch.cat((train_features[:,:,3:6],train_features[:,:,9:12],train_features[:,:,15:18],train_features[:,:,21:24]),axis=-1)
test_features_gyr_4=torch.cat((test_features[:,:,3:6],test_features[:,:,9:12],test_features[:,:,15:18],test_features[:,:,21:24]),axis=-1)
val_features_gyr_4=torch.cat((val_features[:,:,3:6],val_features[:,:,9:12],val_features[:,:,15:18],val_features[:,:,21:24]),axis=-1)



train_features_Kinematics=train_features[:,:,24:36]
test_features_Kinematics=test_features[:,:,24:36]
val_features_Kinematics=val_features[:,:,24:36]


train_features_EMG=train_features[:,:,36:47]
test_features_EMG=test_features[:,:,36:47]
val_features_EMG=val_features[:,:,36:47]


train_features_JP=train_features[:,:,52:59]
test_features_JP=test_features[:,:,52:59]
val_features_JP=val_features[:,:,52:59]



train = TensorDataset(train_features, train_features_acc_4,train_features_gyr_4, train_features_Kinematics,train_features_EMG,train_features_JP, train_targets)
val = TensorDataset(val_features, val_features_acc_4, val_features_gyr_4, val_features_Kinematics, val_features_EMG, val_features_JP,val_targets)
test = TensorDataset(test_features, test_features_acc_4, test_features_gyr_4, test_features_Kinematics,test_features_EMG,test_features_JP, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=False)

"""# Important Functions"""

def RMSE_prediction(yhat_4,test_y,s):

  s1=yhat_4.shape[0]*yhat_4.shape[1]
 
  test_o=test_y.reshape((s1,output_dim))
  yhat=yhat_4.reshape((s1,output_dim))
  
  
  
  y_1_no=yhat[:,0]
  y_2_no=yhat[:,1]
  y_3_no=yhat[:,2]
  y_4_no=yhat[:,3]
  y_5_no=yhat[:,4]
  y_6_no=yhat[:,5]
  y_7_no=yhat[:,6]

  
  y_1=y_1_no
  y_2=y_2_no
  y_3=y_3_no
  y_4=y_4_no
  y_5=y_5_no
  y_6=y_6_no
  y_7=y_7_no

  
  
  y_test_1=test_o[:,0]
  y_test_2=test_o[:,1]
  y_test_3=test_o[:,2]
  y_test_4=test_o[:,3]
  y_test_5=test_o[:,4]
  y_test_6=test_o[:,5]
  y_test_7=test_o[:,6]

  
  
  Z_1=y_1
  Z_2=y_2
  Z_3=y_3
  Z_4=y_4
  Z_5=y_5
  Z_6=y_6
  Z_7=y_7

  
  
  
  ###calculate RMSE
  
  rmse_1 =((np.sqrt(mean_squared_error(y_test_1,y_1)))/(max(y_test_1)-min(y_test_1)))*100
  rmse_2 =((np.sqrt(mean_squared_error(y_test_2,y_2)))/(max(y_test_2)-min(y_test_2)))*100
  rmse_3 =((np.sqrt(mean_squared_error(y_test_3,y_3)))/(max(y_test_3)-min(y_test_3)))*100
  rmse_4 =((np.sqrt(mean_squared_error(y_test_4,y_4)))/(max(y_test_4)-min(y_test_4)))*100
  rmse_5 =((np.sqrt(mean_squared_error(y_test_5,y_5)))/(max(y_test_5)-min(y_test_5)))*100
  rmse_6 =((np.sqrt(mean_squared_error(y_test_6,y_6)))/(max(y_test_6)-min(y_test_6)))*100
  rmse_7 =((np.sqrt(mean_squared_error(y_test_7,y_7)))/(max(y_test_7)-min(y_test_7)))*100

  
  
  print(rmse_1)
  print(rmse_2)
  print(rmse_3)
  print(rmse_4)
  print(rmse_5)
  print(rmse_6)
  print(rmse_7)

  
  
  p_1=np.corrcoef(y_1, y_test_1)[0, 1]
  p_2=np.corrcoef(y_2, y_test_2)[0, 1]
  p_3=np.corrcoef(y_3, y_test_3)[0, 1]
  p_4=np.corrcoef(y_4, y_test_4)[0, 1]
  p_5=np.corrcoef(y_5, y_test_5)[0, 1]
  p_6=np.corrcoef(y_6, y_test_6)[0, 1]
  p_7=np.corrcoef(y_7, y_test_7)[0, 1]

  
  
  print("\n") 
  print(p_1)
  print(p_2)
  print(p_3)
  print(p_4)
  print(p_5)
  print(p_6)
  print(p_7)

  
  
              ### Correlation ###
  p=np.array([p_1,p_2,p_3,p_4,p_5,p_6,p_7])
  
  
  
  
      #### Mean and standard deviation ####
  
  rmse=np.array([rmse_1,rmse_2,rmse_3,rmse_4,rmse_5,rmse_6,rmse_7])
  
      #### Mean and standard deviation ####
  m=statistics.mean(rmse)
  SD=statistics.stdev(rmse)
  print('Mean: %.3f' % m,'+/- %.3f' %SD)
   
  m_c=statistics.mean(p)
  SD_c=statistics.stdev(p)
  print('Mean: %.3f' % m_c,'+/- %.3f' %SD_c)



  return rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7



######################################################################################################################################################################################################################################################################################

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        mse = nn.MSELoss()(pred, target)
        rmse = torch.sqrt(mse)
        return rmse

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


######################################################################################################################################################################################################################################################################################

## IMU4+EMG

## Training Function

def train_mm_early(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    # criterion =nn.MSELoss()
    criterion =RMSELoss()

    # criterion=PearsonCorrLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    # optimizer = torch.optim.Adam(model.parameters())


    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data=torch.cat((data[:,:,0:24], data[:,:,36:47]),dim=-1)
            output= model(data.to(device).float())

            # l2_regularization = 0.0
            # for param in model.parameters():
            #     l2_regularization += torch.norm(param, p=2)  # Compute the L2 norm of the parameter


            loss = criterion(output, target.to(device).float())
            loss.backward()
            optimizer.step()


            running_loss += loss.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_Kinematics, data_EMG,data_JP, target in val_loader:
                data=torch.cat((data[:,:,0:24], data[:,:,36:47]),dim=-1)
                output= model(data.to(device).float())
                val_loss += criterion(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model

def train_mm_m(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    criterion =RMSELoss()

    # criterion=PearsonCorrLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    optimizer = torch.optim.Adam(model.parameters())


    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output= model(data_acc.to(device).float(),data_gyr.to(device).float(), data_EMG.to(device).float())

            loss = criterion(output, target.to(device).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_Kinematics, data_EMG,data_JP, target in val_loader:
                output= model(data_acc.to(device).float(),data_gyr.to(device).float(), data_EMG.to(device).float())
                val_loss += criterion(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break


    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")


    # # Save the trained model
    # torch.save(model.state_dict(), "model.pth")

    return model


class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout(out_2)

        return out_2


## Model training

### Early Fusion

class MM_early(nn.Module):

    def __init__(self, input, drop_prob=0.15):
        super(MM_early, self).__init__()
        self.encoder_input=Encoder(input,drop_prob)
        self.fc = nn.Linear(128, 7)
        self.BN= nn.BatchNorm1d(input, affine=False)

    def forward(self, input_x):

        input_x_1=input_x.view(input_x.size(0)*input_x.size(1),input_x.size(-1))
        input_x_1=self.BN(input_x_1)
        input_x_2=input_x_1.view(-1, w, input_x_1.size(-1))
        out=self.encoder_input(input_x_2)

        out = self.fc(out)

        return out

lr = 0.001
model = MM_early(35)

mm_early = train_mm_early(train_loader, lr,40,model,path + encoder + '_early_IMU4_EMG.pth')

mm_early= MM_early(35)
mm_early.load_state_dict(torch.load(path+encoder+'_early_IMU4_EMG.pth'))
mm_early.to(device)

mm_early.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        data=torch.cat((data[:,:,0:24], data[:,:,36:47]),dim=-1)
        output= mm_early(data.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()



yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_1=np.hstack([rmse,p])

### Feature Concatentaion

class MM_concat(nn.Module):
    def __init__(self, input_acc, input_gyr, input_emg, drop_prob=0.35):
        super(MM_concat, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_emg=Encoder(input_emg, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_emg= nn.BatchNorm1d(input_emg, affine=False)

        self.fc = nn.Linear(3*128, 7)

    def forward(self, x_acc, x_gyr, x_emg):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_emg_1=x_emg.view(x_emg.size(0)*x_emg.size(1),x_emg.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_emg_1=self.BN_emg(x_emg_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_emg_2=x_emg_1.view(-1, w, x_emg_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_emg=self.encoder_emg(x_emg_2)

        x=torch.cat((x_acc,x_gyr,x_emg),dim=-1)

        out = self.fc(x)

        return out

lr = 0.001
model = MM_concat(12,12,11)

mm_concat = train_mm_m(train_loader, lr,40,model,path+encoder+'_concat_IMU4_EMG.pth')

mm_concat= MM_concat(12,12,11)
mm_concat.load_state_dict(torch.load(path+encoder+'_concat_IMU4_EMG.pth'))
mm_concat.to(device)

mm_concat.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        output = mm_concat(data_acc.to(device).float(),data_gyr.to(device).float(),data_EMG.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_2=np.hstack([rmse,p])


### Tensor Fusion with Multiplication

class MM_wfs(nn.Module):
    def __init__(self, input_acc, input_gyr,input_emg, drop_prob=0.05):
        super(MM_wfs, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_emg=Encoder(input_emg, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_emg= nn.BatchNorm1d(input_emg, affine=False)

               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

        self.fc = nn.Linear(128, 7)

    def forward(self, x_acc, x_gyr, x_emg):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_emg_1=x_emg.view(x_emg.size(0)*x_emg.size(1),x_emg.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_emg_1=self.BN_emg(x_emg_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_emg_2=x_emg_1.view(-1, w, x_emg_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_emg=self.encoder_emg(x_emg_2)

        x=torch.cat((x_acc, x_gyr, x_emg),dim=-1)

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        weights_3 = self.weighted_feat(x[:,:,2*128:3*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        x_3=weights_3*x[:,:,2*128:3*128]
        out=x_1+x_2+x_3


        out = self.fc(out)

        return out


lr = 0.001
model = MM_wfs(12,12,11)
mm_wfs= train_mm_m(train_loader, lr,40,model,path+encoder+'_wfs_IMU4_EMG.pth')

mm_wfs= MM_wfs(12,12,11)
mm_wfs.load_state_dict(torch.load(path+encoder+'_wfs_IMU4_EMG.pth'))
mm_wfs.to(device)

mm_wfs.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        output = mm_wfs(data_acc.to(device).float(),data_gyr.to(device).float(),data_EMG.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_4=np.hstack([rmse,p])

### Gated Multi-modal fusion

class MM_gmf(nn.Module):
    def __init__(self, input_acc, input_gyr,input_emg,  drop_prob=0.25):
        super(MM_gmf, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_emg=Encoder(input_emg, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_emg= nn.BatchNorm1d(input_emg, affine=False)


        self.fc = nn.Linear(3*128, 7)

        self.dropout=nn.Dropout(p=0.05)

        # Define the gating network
        self.gating_net = nn.Sequential(
            nn.Linear(128 * 3, 3*128),
            nn.Sigmoid()
        )



    def forward(self, x_acc, x_gyr, x_emg):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_emg_1=x_emg.view(x_emg.size(0)*x_emg.size(1),x_emg.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_emg_1=self.BN_emg(x_emg_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_emg_2=x_emg_1.view(-1, w, x_emg_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_emg=self.encoder_emg(x_emg_2)


        x=torch.cat((x_acc,x_gyr,x_emg),dim=-1)

        gating_weights = self.gating_net(x)

        out=gating_weights*x
        out = self.fc(out)


        return out


lr = 0.001
model = MM_gmf(12,12,11)

mm_gmf= train_mm_m(train_loader, lr,40,model,path+encoder+ '_gmf_IMU4_EMG.pth')

mm_gmf= MM_gmf(12,12,11)
mm_gmf.load_state_dict(torch.load(path+encoder+'_gmf_IMU4_EMG.pth'))
mm_gmf.to(device)

mm_gmf.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        output = mm_gmf(data_acc.to(device).float(),data_gyr.to(device).float(),data_EMG.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_5=np.hstack([rmse,p])

### Multi-Head self Attention Module

class MM_mha(nn.Module):
    def __init__(self, input_acc, input_gyr,input_emg, drop_prob=0.25):
        super(MM_mha, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_emg=Encoder(input_emg, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_emg= nn.BatchNorm1d(input_emg, affine=False)


        self.fc_1 = nn.Linear(2*128, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc = nn.Linear(3*128, 7)

        self.dropout=nn.Dropout(p=0.05)

        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)



    def forward(self, x_acc, x_gyr, x_emg):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_emg_1=x_emg.view(x_emg.size(0)*x_emg.size(1),x_emg.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_emg_1=self.BN_emg(x_emg_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_emg_2=x_emg_1.view(-1, w, x_emg_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_emg=self.encoder_emg(x_emg_2)

        x=torch.cat((x_acc,x_gyr,x_emg),dim=-1)

        out, attn_output_weights=self.attention(x,x,x)

        out = self.fc(out)

        return out


lr = 0.001
model = MM_mha(12,12,11)

mm_mha = train_mm_m(train_loader, lr,40,model,path+encoder+'_mha_IMU4_EMG.pth')

mm_mha= MM_mha(12,12,11)
mm_mha.load_state_dict(torch.load(path+encoder+'_mha_IMU4_EMG.pth'))
mm_mha.to(device)

mm_mha.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        output = mm_mha(data_acc.to(device).float(),data_gyr.to(device).float(),data_EMG.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_6=np.hstack([rmse,p])

### MHA+Weighted Feature Fusion

class MM_mha_wf(nn.Module):
    def __init__(self, input_acc, input_gyr, input_emg, drop_prob=0.25):
        super(MM_mha_wf, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_emg=Encoder(input_emg, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_emg= nn.BatchNorm1d(input_emg, affine=False)

        self.fc = nn.Linear(2*3*128,7)

        self.dropout=nn.Dropout(p=0.05)

        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)

        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128, 2*3*128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr, x_emg):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_emg_1=x_emg.view(x_emg.size(0)*x_emg.size(1),x_emg.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_emg_1=self.BN_emg(x_emg_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_emg_2=x_emg_1.view(-1, w, x_emg_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_emg=self.encoder_emg(x_emg_2)

        x=torch.cat((x_acc,x_gyr,x_emg),dim=-1)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out=torch.cat((out_1,out_2),dim=-1)

        # gating_weights_1 = self.gating_net_1(out)
        # out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = MM_mha_wf(12,12,11)

mm_mha_wf = train_mm_m(train_loader, lr,40,model,path+encoder+'_mha_wf_IMU4_emg.pth')

mm_mha_wf= MM_mha_wf(12,12,11)
mm_mha_wf.load_state_dict(torch.load(path+encoder+'_mha_wf_IMU4_emg.pth'))
mm_mha_wf.to(device)

mm_mha_wf.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        output = mm_mha_wf(data_acc.to(device).float(),data_gyr.to(device).float(), data_EMG.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_7=np.hstack([rmse,p])

### Weighted Fusion of MHA+Weighted Feature Fusion

class MM_mha_wf_fusion(nn.Module):
    def __init__(self, input_acc, input_gyr, input_emg, drop_prob=0.25):
        super(MM_mha_wf_fusion, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_emg=Encoder(input_emg, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_emg= nn.BatchNorm1d(input_emg, affine=False)

        self.fc = nn.Linear(2*3*128,7)

        self.dropout=nn.Dropout(p=0.05)

        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)

        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128, 2*3*128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr, x_emg):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_emg_1=x_emg.view(x_emg.size(0)*x_emg.size(1),x_emg.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_emg_1=self.BN_emg(x_emg_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_emg_2=x_emg_1.view(-1, w, x_emg_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_emg=self.encoder_emg(x_emg_2)

        x=torch.cat((x_acc,x_gyr,x_emg),dim=-1)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out=torch.cat((out_1,out_2),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = MM_mha_wf_fusion(12,12,11)

mm_mha_wf_fusion = train_mm_m(train_loader, lr,40,model,path+encoder+'_mha_wf_fusion_IMU4_emg.pth')

mm_mha_wf_fusion= MM_mha_wf_fusion(12,12,11)
mm_mha_wf_fusion.load_state_dict(torch.load(path+encoder+'_mha_wf_fusion_IMU4_emg.pth'))
mm_mha_wf_fusion.to(device)

mm_mha_wf_fusion.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        output = mm_mha_wf_fusion(data_acc.to(device).float(),data_gyr.to(device).float(), data_EMG.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_8=np.hstack([rmse,p])

### MHA+Tensor Multiplication

class MM_mha_wfs(nn.Module):
    def __init__(self, input_acc, input_gyr, input_emg, drop_prob=0.25):
        super(MM_mha_wfs, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_emg=Encoder(input_emg, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_emg= nn.BatchNorm1d(input_emg, affine=False)

        self.fc = nn.Linear(3*128+128,7)

                        # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())


        self.dropout=nn.Dropout(p=0.05)

        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)

        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(3*128+128, 3*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr, x_emg):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_emg_1=x_emg.view(x_emg.size(0)*x_emg.size(1),x_emg.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_emg_1=self.BN_emg(x_emg_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_emg_2=x_emg_1.view(-1, w, x_emg_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_emg=self.encoder_emg(x_emg_2)

        x=torch.cat((x_acc,x_gyr,x_emg),dim=-1)

        out_1, attn_output_weights=self.attention(x,x,x)


        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        weights_3 = self.weighted_feat(x[:,:,2*128:3*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        x_3=weights_3*x[:,:,2*128:3*128]
        out_3=x_1+x_2+x_3


        out=torch.cat((out_1,out_3),dim=-1)

        # gating_weights_1 = self.gating_net_1(out)
        # out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = MM_mha_wfs(12,12,11)

mm_mha_wfs = train_mm_m(train_loader, lr,40,model,path+encoder+'_mha_wfs_IMU4_emg.pth')

mm_mha_wfs= MM_mha_wfs(12,12,11)
mm_mha_wfs.load_state_dict(torch.load(path+encoder+'_mha_wfs_IMU4_emg.pth'))
mm_mha_wfs.to(device)

mm_mha_wfs.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        output = mm_mha_wfs(data_acc.to(device).float(),data_gyr.to(device).float(), data_EMG.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_9=np.hstack([rmse,p])

### Weighted Fusion of MHA+Tensor Multiplication

class MM_mha_wfs_fusion(nn.Module):
    def __init__(self, input_acc, input_gyr, input_emg, drop_prob=0.25):
        super(MM_mha_wfs_fusion, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_emg=Encoder(input_emg, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_emg= nn.BatchNorm1d(input_emg, affine=False)

        self.fc = nn.Linear(3*128+128,7)

               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())


        self.dropout=nn.Dropout(p=0.05)
        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)

        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(3*128+128, 3*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr, x_emg):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_emg_1=x_emg.view(x_emg.size(0)*x_emg.size(1),x_emg.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_emg_1=self.BN_emg(x_emg_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_emg_2=x_emg_1.view(-1, w, x_emg_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_emg=self.encoder_emg(x_emg_2)

        x=torch.cat((x_acc,x_gyr,x_emg),dim=-1)

        out_1, attn_output_weights=self.attention(x,x,x)

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        weights_3 = self.weighted_feat(x[:,:,2*128:3*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        x_3=weights_3*x[:,:,2*128:3*128]
        out_3=x_1+x_2+x_3


        out=torch.cat((out_1,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = MM_mha_wfs_fusion(12,12,11)

mm_mha_wfs_fusion = train_mm_m(train_loader, lr,40,model,path+encoder+'_mha_wfs_fusion_IMU4_emg.pth')

mm_mha_wfs_fusion= MM_mha_wfs_fusion(12,12,11)
mm_mha_wfs_fusion.load_state_dict(torch.load(path+encoder+'_mha_wfs_fusion_IMU4_emg.pth'))
mm_mha_wfs_fusion.to(device)

mm_mha_wfs_fusion.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        output = mm_mha_wfs_fusion(data_acc.to(device).float(),data_gyr.to(device).float(), data_EMG.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_10=np.hstack([rmse,p])

### Weighted Features + Tensor Multiplication

class MM_wf_wfs(nn.Module):
    def __init__(self, input_acc, input_gyr, input_emg, drop_prob=0.25):
        super(MM_wf_wfs, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_emg=Encoder(input_emg, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_emg= nn.BatchNorm1d(input_emg, affine=False)

        self.fc = nn.Linear(3*128+128,7)
               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())


        self.dropout=nn.Dropout(p=0.05)

        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(3*128+128, 3*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr, x_emg):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_emg_1=x_emg.view(x_emg.size(0)*x_emg.size(1),x_emg.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_emg_1=self.BN_emg(x_emg_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_emg_2=x_emg_1.view(-1, w, x_emg_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_emg=self.encoder_emg(x_emg_2)

        x=torch.cat((x_acc,x_gyr,x_emg),dim=-1)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        weights_3 = self.weighted_feat(x[:,:,2*128:3*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        x_3=weights_3*x[:,:,2*128:3*128]
        out_3=x_1+x_2+x_3


        out=torch.cat((out_2,out_3),dim=-1)

        # gating_weights_1 = self.gating_net_1(out)
        # out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = MM_wf_wfs(12,12,11)

mm_wf_wfs = train_mm_m(train_loader, lr,40,model,path+encoder+'_wf_wfs_IMU4_emg.pth')

mm_wf_wfs= MM_wf_wfs(12,12,11)
mm_wf_wfs.load_state_dict(torch.load(path+encoder+'_wf_wfs_IMU4_emg.pth'))
mm_wf_wfs.to(device)

mm_wf_wfs.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        output = mm_wf_wfs(data_acc.to(device).float(),data_gyr.to(device).float(), data_EMG.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_11=np.hstack([rmse,p])

### Weighted Fusion of weighted features +Tensor Multiplication

class MM_wf_wfs_fusion(nn.Module):
    def __init__(self, input_acc, input_gyr, input_emg, drop_prob=0.25):
        super(MM_wf_wfs_fusion, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_emg=Encoder(input_emg, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_emg= nn.BatchNorm1d(input_emg, affine=False)

        self.fc = nn.Linear(3*128+128,7)

        self.dropout=nn.Dropout(p=0.05)
               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())


        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(3*128+128, 3*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr, x_emg):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_emg_1=x_emg.view(x_emg.size(0)*x_emg.size(1),x_emg.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_emg_1=self.BN_emg(x_emg_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_emg_2=x_emg_1.view(-1, w, x_emg_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_emg=self.encoder_emg(x_emg_2)

        x=torch.cat((x_acc,x_gyr,x_emg),dim=-1)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        weights_3 = self.weighted_feat(x[:,:,2*128:3*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        x_3=weights_3*x[:,:,2*128:3*128]
        out_3=x_1+x_2+x_3


        out=torch.cat((out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = MM_wf_wfs_fusion(12,12,11)

mm_wf_wfs_fusion = train_mm_m(train_loader, lr,40,model,path+encoder+'_wf_wfs_fusion_IMU4_emg.pth')

mm_wf_wfs_fusion= MM_wf_wfs_fusion(12,12,11)
mm_wf_wfs_fusion.load_state_dict(torch.load(path+encoder+'_wf_wfs_fusion_IMU4_emg.pth'))
mm_wf_wfs_fusion.to(device)

mm_wf_wfs_fusion.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        output = mm_wf_wfs_fusion(data_acc.to(device).float(),data_gyr.to(device).float(), data_EMG.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_12=np.hstack([rmse,p])

### MHA+Weighted Feature+ Tensor Multiplication Fusion

class MM_mha_wf_wfs(nn.Module):
    def __init__(self, input_acc, input_gyr, input_emg, drop_prob=0.25):
        super(MM_mha_wf_wfs, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_emg=Encoder(input_emg, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_emg= nn.BatchNorm1d(input_emg, affine=False)

        self.fc = nn.Linear(2*3*128+128,7)

        self.dropout=nn.Dropout(p=0.05)

               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())


        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)

        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr, x_emg):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_emg_1=x_emg.view(x_emg.size(0)*x_emg.size(1),x_emg.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_emg_1=self.BN_emg(x_emg_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_emg_2=x_emg_1.view(-1, w, x_emg_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_emg=self.encoder_emg(x_emg_2)

        x=torch.cat((x_acc,x_gyr,x_emg),dim=-1)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        weights_3 = self.weighted_feat(x[:,:,2*128:3*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        x_3=weights_3*x[:,:,2*128:3*128]
        out_3=x_1+x_2+x_3

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        # gating_weights_1 = self.gating_net_1(out)
        # out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = MM_mha_wf_wfs(12,12,11)

mm_mha_wf_wfs = train_mm_m(train_loader, lr,40,model,path+encoder+'_mha_wf_wfs_IMU4_emg.pth')

mm_mha_wf_wfs= MM_mha_wf_wfs(12,12,11)
mm_mha_wf_wfs.load_state_dict(torch.load(path+encoder+'_mha_wf_wfs_IMU4_emg.pth'))
mm_mha_wf_wfs.to(device)

mm_mha_wf_wfs.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        output = mm_mha_wf_wfs(data_acc.to(device).float(),data_gyr.to(device).float(), data_EMG.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_13=np.hstack([rmse,p])

### Weighted Fusion of MHA+Weighted Feature+ Tensor Multiplication Fusion

class MM_mha_wf_wfs_fusion(nn.Module):
    def __init__(self, input_acc, input_gyr, input_emg, drop_prob=0.25):
        super(MM_mha_wf_wfs_fusion, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_emg=Encoder(input_emg, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_emg= nn.BatchNorm1d(input_emg, affine=False)

        self.fc = nn.Linear(2*3*128+128,7)

        self.dropout=nn.Dropout(p=0.05)

               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())


        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)

        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr, x_emg):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_emg_1=x_emg.view(x_emg.size(0)*x_emg.size(1),x_emg.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_emg_1=self.BN_emg(x_emg_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_emg_2=x_emg_1.view(-1, w, x_emg_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_emg=self.encoder_emg(x_emg_2)

        x=torch.cat((x_acc,x_gyr,x_emg),dim=-1)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        weights_3 = self.weighted_feat(x[:,:,2*128:3*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        x_3=weights_3*x[:,:,2*128:3*128]
        out_3=x_1+x_2+x_3


        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = MM_mha_wf_wfs_fusion(12,12,11)

mm_mha_wf_wfs_fusion = train_mm_m(train_loader, lr,40,model,path+encoder+'_mha_wf_wfs_fusion_IMU4_emg.pth')

mm_mha_wf_wfs_fusion= MM_mha_wf_wfs_fusion(12,12,11)
mm_mha_wf_wfs_fusion.load_state_dict(torch.load(path+encoder+'_mha_wf_wfs_fusion_IMU4_emg.pth'))
mm_mha_wf_wfs_fusion.to(device)

mm_mha_wf_wfs_fusion.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_Kinematics,data_EMG,data_JP, target) in enumerate(test_loader):
        output = mm_mha_wf_wfs_fusion(data_acc.to(device).float(),data_gyr.to(device).float(), data_EMG.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7=RMSE_prediction(yhat_4,test_target,s)

ablation_14=np.hstack([rmse,p])

##############################################################################################################################################


mm_result_IMU4=np.vstack([ablation_1,ablation_2,ablation_4,ablation_5,ablation_6,ablation_7,ablation_8,ablation_9,ablation_10,ablation_11,ablation_12,ablation_13,ablation_14])


path_1='/home/sanzidpr/Journal_3/Dataset_A_model_results_IMU_emg/Results/'

from numpy import savetxt
savetxt(path_1+subject+'_'+encoder+'_IMU_EMG_results.csv', mm_result_IMU4, delimiter=',')




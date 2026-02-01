
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


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torchsummary import summary
from torch.nn.parameter import Parameter


import torch.optim as optim


from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler


# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch random seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU random seed
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner to find best algorithms

# Set your seed value
set_seed(42)



if __name__ == '__main__':
    with h5py.File('/home/sanzidpr/all_17_subjects.h5', 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])

def data_extraction(A):
  for k in range(len(A)):
    zero_index_1=np.all(A[k:k+1,:,:] == 0, axis=0)
    zero_index = np.multiply(zero_index_1, 1)
    zero_index=np.array(zero_index)

    for i in range(len(zero_index)):
      if (sum(zero_index[i])==256):
        index=i
        break;

    # print(index)
### Taking only the stance phase of the gait
###################################################################################################################################################    
    B=A[k:k+1,0:index,:]  ### Taking only the stance phase of the gait
    C_1=B.reshape((B.shape[0]*B.shape[1],B.shape[2]))
    if (k==0):
      C=C_1
    else:
      C=np.append(C,C_1,axis=0)

  index_24 = data_fields.index('body weight')
  index_25 = data_fields.index('body height')

  BW=(C[0:1, index_24]*9.8)
  BWH=(C[0:1, index_24]*9.8)*C[:, index_25]

  V=C[:,110:154]
  V=V.reshape(V.shape[0],11,4)

  V=(V-V[:,2:3,:])/C[0:1, index_25]
  V=V.reshape(-1,44)

      ### IMUs- Chest, Waist, Right Foot, Right shank, Right thigh, Left Foot, Left shank, Left thigh, 2D-body coordinate
    ### 0:48- IMU, 48:92-2D body coordinate, 92:97-- Target

  D=np.hstack((C[:,71:77],C[:,58:64],C[:,19:25],C[:,32:38],C[:,45:51],C[:,6:12],C[:,84:90],C[:,97:103],V,C[:,3:5],-C[:, 154:155]/BW,
              -C[:, 156:157]/BW,-C[:, 155:156]/BW))  
  
  
  # D=np.hstack((C[:,70:76],C[:,57:63],C[:,18:24],C[:,31:37],C[:,44:50],C[:,5:11],C[:,83:89],C[:,96:102],C[:,109:153])) 

  return D

# print(np.array(data_fields))

# index_21 = data_fields.index('plate_2_force_x')
# print(index_21)

data_subject_01 = data_all_sub['subject_01']
subject_1=data_extraction(data_subject_01)

print(subject_1.shape)

data_subject_01 = data_all_sub['subject_01']
data_subject_02 = data_all_sub['subject_02']
data_subject_03 = data_all_sub['subject_03']
data_subject_04 = data_all_sub['subject_04']
data_subject_05 = data_all_sub['subject_05']
data_subject_06 = data_all_sub['subject_06']
data_subject_07 = data_all_sub['subject_07']
data_subject_08 = data_all_sub['subject_08']
data_subject_09 = data_all_sub['subject_09']
data_subject_10 = data_all_sub['subject_10']
data_subject_11 = data_all_sub['subject_11']
data_subject_12 = data_all_sub['subject_12']
data_subject_13 = data_all_sub['subject_13']
data_subject_14 = data_all_sub['subject_14']
data_subject_15 = data_all_sub['subject_15']
data_subject_16 = data_all_sub['subject_16']
data_subject_17 = data_all_sub['subject_17']


subject_1=data_extraction(data_subject_01)
subject_2=data_extraction(data_subject_02)
subject_3=data_extraction(data_subject_03)
subject_4=data_extraction(data_subject_04)
subject_5=data_extraction(data_subject_05)
subject_6=data_extraction(data_subject_06)
subject_7=data_extraction(data_subject_07)
subject_8=data_extraction(data_subject_08)
subject_9=data_extraction(data_subject_09)
subject_10=data_extraction(data_subject_10)
subject_11=data_extraction(data_subject_11)
subject_12=data_extraction(data_subject_12)
subject_13=data_extraction(data_subject_13)
subject_14=data_extraction(data_subject_14)
subject_15=data_extraction(data_subject_15)
subject_16=data_extraction(data_subject_16)
subject_17=data_extraction(data_subject_17)

subject_1.shape



###############################################################################################################################################


"""# Data processing"""



main_dir = "/home/sanzidpr/Journal_3/Dataset_B_model_results_1/Subject17"
#os.mkdir(main_dir) 
path="/home/sanzidpr/Journal_3/Dataset_B_model_results_1/Subject17/"
subject='Subject_17'

train_dataset=np.concatenate((subject_1,subject_2,subject_3,subject_4,subject_5,subject_6,subject_7,subject_8,subject_9,subject_10,
                              subject_11,subject_12,subject_13,subject_14,subject_15,subject_16),axis=0)

test_dataset=subject_17



###############################################################################################################################################


alpha=0.10
beta=5.00


num_epoch=40


encoder='GRU_LSTM'



# Train features #
from sklearn.preprocessing import StandardScaler

# x_train_1=train_dataset[:,0:18]
# x_train_2=train_dataset[:,23:67]

# x_train=np.concatenate((x_train_1,x_train_2),axis=1)

x_train=train_dataset[:,0:92]


scale= StandardScaler()
scaler = MinMaxScaler(feature_range=(0, 1))
train_X_1_1=x_train

# # Test features #
# x_test_1=test_dataset[:,0:18]
# x_test_2=test_dataset[:,23:67]

# x_test=np.concatenate((x_test_1,x_test_2),axis=1)

x_test=test_dataset[:,0:92]

test_X_1_1=x_test

m1=92
m2=97


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
array_has_nan = np. isinf(train_dataset_1[:,48:92])

print(array_has_nan)

print(train_dataset_1.shape)



train_X_1=train_dataset_1[:,0:m1]
test_X_1=test_dataset_1[:,0:m1]

train_y_1=train_dataset_1[:,m1:m2]
test_y_1=test_dataset_1[:,m1:m2]



L1=len(train_X_1)
L2=len(test_X_1)


w=50



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


train_y_3_p= train_y.reshape((a1+1,w,5))
test_y= test_y.reshape((a2+1,w,5))



# train_X_1D=train_X_3
test_X_1D=test_X

train_X_3=train_X_3_p
train_y_3=train_y_3_p
# print(train_X_4.shape,train_y_3.shape)


train_X_1D, X_validation_1D, train_y_5, Y_validation = train_test_split(train_X_3,train_y_3, test_size=0.20, random_state=True)
#train_X_1D, X_validation_1D_ridge, train_y, Y_validation_ridge = train_test_split(train_X_1D_m,train_y_m, test_size=0.10, random_state=True)   [0:2668,:,:]

print(train_X_1D.shape,train_y_5.shape,X_validation_1D.shape,Y_validation.shape)

features=6



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

# from numpy import savetxt
# savetxt('train_data_check.csv', train_dataset_1[:,48:92], delimiter=',')

### IMUs- Chest, Waist, Right Foot, Right shank, Right thigh, Left Foot, Left shank, Left thigh, 2D-body coordinate
### 0:48- IMU, 48:92-2D body coordinate, 92:97-- Target


### Data Processing

batch_size = 64

val_targets = torch.Tensor(Y_validation)
test_features = torch.Tensor(test_X_1D)
test_targets = torch.Tensor(test_y)


## all Modality Features

train_features = torch.Tensor(train_X_1D)
train_targets = torch.Tensor(train_y_5)
val_features = torch.Tensor(X_validation_1D)


train_features_acc_8=torch.cat((train_features[:,:,0:3],train_features[:,:,6:9],train_features[:,:,12:15],train_features[:,:,18:21],train_features[:,:,24:27]\
                             ,train_features[:,:,30:33],train_features[:,:,36:39],train_features[:,:,42:45]),axis=-1)
test_features_acc_8=torch.cat((test_features[:,:,0:3],test_features[:,:,6:9],test_features[:,:,12:15],test_features[:,:,18:21],test_features[:,:,24:27]\
                             ,test_features[:,:,30:33],test_features[:,:,36:39],test_features[:,:,42:45]),axis=-1)
val_features_acc_8=torch.cat((val_features[:,:,0:3],val_features[:,:,6:9],val_features[:,:,12:15],val_features[:,:,18:21],val_features[:,:,24:27]\
                             ,val_features[:,:,30:33],val_features[:,:,36:39],val_features[:,:,42:45]),axis=-1)


train_features_gyr_8=torch.cat((train_features[:,:,3:6],train_features[:,:,9:12],train_features[:,:,15:18],train_features[:,:,21:24],train_features[:,:,27:30]\
                             ,train_features[:,:,33:36],train_features[:,:,39:42],train_features[:,:,45:48]),axis=-1)
test_features_gyr_8=torch.cat((test_features[:,:,3:6],test_features[:,:,9:12],test_features[:,:,15:18],test_features[:,:,21:24],test_features[:,:,27:30]\
                             ,test_features[:,:,33:36],test_features[:,:,39:42],test_features[:,:,45:48]),axis=-1)
val_features_gyr_8=torch.cat((val_features[:,:,3:6],val_features[:,:,9:12],val_features[:,:,15:18],val_features[:,:,21:24],val_features[:,:,27:30]\
                             ,val_features[:,:,33:36],val_features[:,:,39:42],val_features[:,:,45:48]),axis=-1)



train_features_2D_point=train_features[:,:,48:92]
test_features_2D_point=test_features[:,:,48:92]
val_features_2D_point=val_features[:,:,48:92]



train = TensorDataset(train_features, train_features_acc_8,train_features_gyr_8, train_features_2D_point, train_targets)
val = TensorDataset(val_features, val_features_acc_8, val_features_gyr_8, val_features_2D_point,val_targets)
test = TensorDataset(test_features, test_features_acc_8, test_features_gyr_8, test_features_2D_point, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)




# Important Functions

def RMSE_prediction(yhat_4,test_y,s):

  s1=yhat_4.shape[0]*yhat_4.shape[1]

  test_o=test_y.reshape((s1,5))
  yhat=yhat_4.reshape((s1,5))




  y_1_no=yhat[:,0]
  y_2_no=yhat[:,1]
  y_3_no=yhat[:,2]
  y_4_no=yhat[:,3]
  y_5_no=yhat[:,4]


  y_1=y_1_no
  y_2=y_2_no
  y_3=y_3_no
  y_4=y_4_no
  y_5=y_5_no


  y_test_1=test_o[:,0]
  y_test_2=test_o[:,1]
  y_test_3=test_o[:,2]
  y_test_4=test_o[:,3]
  y_test_5=test_o[:,4]




  Z_1=y_1
  Z_2=y_2
  Z_3=y_3
  Z_4=y_4
  Z_5=y_5



  ###calculate RMSE

  rmse_1 =((np.sqrt(mean_squared_error(y_test_1,y_1)))/(max(y_test_1)-min(y_test_1)))*100
  rmse_2 =((np.sqrt(mean_squared_error(y_test_2,y_2)))/(max(y_test_2)-min(y_test_2)))*100
  rmse_3 =((np.sqrt(mean_squared_error(y_test_3,y_3)))/(max(y_test_3)-min(y_test_3)))*100
  rmse_4 =((np.sqrt(mean_squared_error(y_test_4,y_4)))/(max(y_test_4)-min(y_test_4)))*100
  rmse_5 =((np.sqrt(mean_squared_error(y_test_5,y_5)))/(max(y_test_5)-min(y_test_5)))*100



  print(rmse_1)
  print(rmse_2)
  print(rmse_3)
  print(rmse_4)
  print(rmse_5)



  p_1=np.corrcoef(y_1, y_test_1)[0, 1]
  p_2=np.corrcoef(y_2, y_test_2)[0, 1]
  p_3=np.corrcoef(y_3, y_test_3)[0, 1]
  p_4=np.corrcoef(y_4, y_test_4)[0, 1]
  p_5=np.corrcoef(y_5, y_test_5)[0, 1]


  print("\n")
  print(p_1)
  print(p_2)
  print(p_3)
  print(p_4)
  print(p_5)



              ### Correlation ###
  p=np.array([p_1,p_2,p_3,p_4,p_5])




      #### Mean and standard deviation ####

  rmse=np.array([rmse_1,rmse_2,rmse_3,rmse_4,rmse_5])

      #### Mean and standard deviation ####
  m=statistics.mean(rmse)
  SD=statistics.stdev(rmse)
  print('Mean: %.3f' % m,'+/- %.3f' %SD)

  m_c=statistics.mean(p)
  SD_c=statistics.stdev(p)
  print('Mean: %.3f' % m_c,'+/- %.3f' %SD_c)



  return rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5




# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        mse = nn.MSELoss()(pred, target)
        rmse = torch.sqrt(mse)
        return rmse






# Teacher Model Training -- IMU-2D

## Training Function

def train_mm_teacher(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    # criterion =nn.MSELoss()
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
        for i, (data, data_acc, data_gyr, data_2D, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output,x_1= model(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())

            loss = criterion(output, target.to(device).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D,  target in val_loader:
                output,x_1= model(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())
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

## Multi-Encoder Teacher Training

class Encoder_1(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_1, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2




class Encoder_2(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_2, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2



class GatingModule(nn.Module):
    def __init__(self, input_size):
        super(GatingModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        # Apply gating mechanism
        gate_output = self.gate(torch.cat((input1,input2),dim=-1))

        # Scale the inputs based on the gate output
        gated_input1 = input1 * gate_output
        gated_input2 = input2 * (1 - gate_output)

        # Combine the gated inputs
        output = gated_input1 + gated_input2
        return output

class teacher(nn.Module):
    def __init__(self, input_acc, input_gyr,input_2D, drop_prob=0.25):
        super(teacher, self).__init__()

        self.encoder_1_acc=Encoder_1(input_acc, drop_prob)
        self.encoder_1_gyr=Encoder_1(input_gyr, drop_prob)
        self.encoder_1_2d=Encoder_1(input_2D, drop_prob)

        self.encoder_2_acc=Encoder_2(input_acc, drop_prob)
        self.encoder_2_gyr=Encoder_2(input_gyr, drop_prob)
        self.encoder_2_2d=Encoder_2(input_2D, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_2d= nn.BatchNorm1d(input_2D, affine=False)

        self.fc = nn.Linear(2*3*128+128,5)
        self.dropout=nn.Dropout(p=0.05)

        self.gate_1=GatingModule(128)
        self.gate_2=GatingModule(128)
        self.gate_3=GatingModule(128)

        self.fc_kd = nn.Linear(3*128, 128)

               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr, x_2d):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_2d_1=x_2d.view(x_2d.size(0)*x_2d.size(1),x_2d.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_2d_1=self.BN_2d(x_2d_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))
        x_2d_2=x_2d_1.view(-1, w, x_2d_1.size(-1))

        x_acc_1=self.encoder_1_acc(x_acc_2)
        x_gyr_1=self.encoder_1_gyr(x_gyr_2)
        x_2d_1=self.encoder_1_2d(x_2d_2)

        x_acc_2=self.encoder_2_acc(x_acc_2)
        x_gyr_2=self.encoder_2_gyr(x_gyr_2)
        x_2d_2=self.encoder_2_2d(x_2d_2)

        x_acc=self.gate_1(x_acc_1,x_acc_2)
        x_gyr=self.gate_2(x_gyr_1,x_gyr_2)
        x_2d=self.gate_3(x_2d_1,x_2d_2)

        x=torch.cat((x_acc,x_gyr,x_2d),dim=-1)
        x_kd=self.fc_kd(x)

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

        return out,x_kd


lr = 0.001
model = teacher(24,24,44)

#mm_teacher = train_mm_teacher(train_loader, lr,num_epoch,model,path+'_teacher.pth')

mm_teacher= teacher(24,24,44)
mm_teacher.load_state_dict(torch.load(path+'_teacher.pth'))
mm_teacher.to(device)

mm_teacher.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(test_loader):
        output,x = mm_teacher(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())
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

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_1=np.hstack([rmse,p])

# Student Model-- 1. Foot

## Training Function

def train_mm_student_1(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    criterion =RMSELoss()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=2e-5)


    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(train_loader):
            optimizer.zero_grad()
            data_acc=torch.cat((data_acc[:,:,6:9],data_acc[:,:,15:18]),dim=-1)
            data_gyr=torch.cat((data_gyr[:,:,6:9],data_gyr[:,:,15:18]),dim=-1)
            output= model(data_acc.to(device).float(),data_gyr.to(device).float())

            loss = criterion(output, target.to(device).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D,  target in val_loader:
                data_acc=torch.cat((data_acc[:,:,6:9],data_acc[:,:,15:18]),dim=-1)
                data_gyr=torch.cat((data_gyr[:,:,6:9],data_gyr[:,:,15:18]),dim=-1)
                output= model(data_acc.to(device).float(),data_gyr.to(device).float())
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

## Multi-Encoder Student Training

class Encoder_1(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_1, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2




class Encoder_2(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_2, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2



class GatingModule(nn.Module):
    def __init__(self, input_size):
        super(GatingModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        # Apply gating mechanism
        gate_output = self.gate(torch.cat((input1,input2),dim=-1))

        # Scale the inputs based on the gate output
        gated_input1 = input1 * gate_output
        gated_input2 = input2 * (1 - gate_output)

        # Combine the gated inputs
        output = gated_input1 + gated_input2
        return output

class student_1(nn.Module):
    def __init__(self, input_acc, input_gyr, drop_prob=0.25):
        super(student_1, self).__init__()

        self.encoder_1_acc=Encoder_1(input_acc, drop_prob)
        self.encoder_1_gyr=Encoder_1(input_gyr, drop_prob)

        self.encoder_2_acc=Encoder_2(input_acc, drop_prob)
        self.encoder_2_gyr=Encoder_2(input_gyr, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)

        self.fc = nn.Linear(2*2*128+128,5)
        self.dropout=nn.Dropout(p=0.05)

        self.gate_1=GatingModule(128)
        self.gate_2=GatingModule(128)


               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

        self.attention=nn.MultiheadAttention(2*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*2, 2*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*2*128+128, 2*2*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))

        x_acc_1=self.encoder_1_acc(x_acc_2)
        x_gyr_1=self.encoder_1_gyr(x_gyr_2)

        x_acc_2=self.encoder_2_acc(x_acc_2)
        x_gyr_2=self.encoder_2_gyr(x_gyr_2)

        x_acc=self.gate_1(x_acc_1,x_acc_2)
        x_gyr=self.gate_2(x_gyr_1,x_gyr_2)

        x=torch.cat((x_acc,x_gyr),dim=-1)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        out_3=x_1+x_2

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = student_1(6,6)

mm_student_1 = train_mm_student_1(train_loader, lr,num_epoch,model,path+'_student_1.pth')

mm_student_1= student_1(6,6)
mm_student_1.load_state_dict(torch.load(path+'_student_1.pth'))
mm_student_1.to(device)

mm_student_1.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(test_loader):
        data_acc=torch.cat((data_acc[:,:,6:9],data_acc[:,:,15:18]),dim=-1)
        data_gyr=torch.cat((data_gyr[:,:,6:9],data_gyr[:,:,15:18]),dim=-1)
        output = mm_student_1(data_acc.to(device).float(),data_gyr.to(device).float())
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

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_2=np.hstack([rmse,p])

## Student+sensor and knowledge distillation


"""## Knowledge distillation"""

def train_student_kd_1(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()

    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(train_loader):
            optimizer.zero_grad()
            data_acc_1=torch.cat((data_acc[:,:,6:9],data_acc[:,:,15:18]),dim=-1)
            data_gyr_1=torch.cat((data_gyr[:,:,6:9],data_gyr[:,:,15:18]),dim=-1)
            student_output, x_student_1= model(data_acc_1.to(device).float(),data_gyr_1.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(), data_2D.to(device).float())
                

            layer_loss=criterion_2(x_student_1,x_teacher_1)
            vanilla_loss=criterion_2(student_output,teacher_output)

            loss=criterion_2(student_output,target.to(device))+beta*layer_loss+alpha*vanilla_loss


            loss_1=criterion_2(student_output,target.to(device))


            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

        running_loss=0

     # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D,  target in val_loader:
                data_acc=torch.cat((data_acc[:,:,6:9],data_acc[:,:,15:18]),dim=-1)
                data_gyr=torch.cat((data_gyr[:,:,6:9],data_gyr[:,:,15:18]),dim=-1)
                output,student_1= model(data_acc.to(device).float(),data_gyr.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

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


    return model
    
    
    
    

class student_1_kd(nn.Module):
    def __init__(self, input_acc, input_gyr, drop_prob=0.25):
        super(student_1_kd, self).__init__()

        self.encoder_1_acc=Encoder_1(input_acc, drop_prob)
        self.encoder_1_gyr=Encoder_1(input_gyr, drop_prob)

        self.encoder_2_acc=Encoder_2(input_acc, drop_prob)
        self.encoder_2_gyr=Encoder_2(input_gyr, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)

        self.fc_kd=nn.Linear(2*128,128)

        self.fc = nn.Linear(2*2*128+128,5)
        self.dropout=nn.Dropout(p=0.05)


        self.gate_1=GatingModule(128)
        self.gate_2=GatingModule(128)


               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

        self.attention=nn.MultiheadAttention(2*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*2, 2*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*2*128+128, 2*2*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))

        x_acc_1=self.encoder_1_acc(x_acc_2)
        x_gyr_1=self.encoder_1_gyr(x_gyr_2)

        x_acc_2=self.encoder_2_acc(x_acc_2)
        x_gyr_2=self.encoder_2_gyr(x_gyr_2)


        x_acc=self.gate_1(x_acc_1,x_acc_2)
        x_gyr=self.gate_2(x_gyr_1,x_gyr_2)

        x=torch.cat((x_acc,x_gyr),dim=-1)

        x_KD=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        out_3=x_1+x_2

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out,x_KD

lr = 0.001
student = student_1_kd(6,6)

teacher_trained= teacher(24,24,44)
teacher_trained.load_state_dict(torch.load(path+'_teacher.pth'))
teacher_trained.to(device)


student_KD= train_student_kd_1(train_loader, lr,num_epoch, student,path+'student_1_kd.pth', teacher_trained)

mm_student_1= student_1_kd(6,6)
mm_student_1.load_state_dict(torch.load(path+'student_1_kd.pth'))
mm_student_1.to(device)

mm_student_1.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(test_loader):
        data_acc=torch.cat((data_acc[:,:,6:9],data_acc[:,:,15:18]),dim=-1)
        data_gyr=torch.cat((data_gyr[:,:,6:9],data_gyr[:,:,15:18]),dim=-1)
        output,student_1 = mm_student_1(data_acc.to(device).float(),data_gyr.to(device).float())
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

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_3=np.hstack([rmse,p])

# Student Model-- 2. Foot+pelvis

## Training Function

def train_mm_student_2(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    criterion =RMSELoss()

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
        for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(train_loader):
            optimizer.zero_grad()
            data_acc=torch.cat((data_acc[:,:,3:9],data_acc[:,:,15:18]),dim=-1)
            data_gyr=torch.cat((data_gyr[:,:,3:9],data_gyr[:,:,15:18]),dim=-1)
            output= model(data_acc.to(device).float(),data_gyr.to(device).float())

            loss = criterion(output, target.to(device).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D,  target in val_loader:
                data_acc=torch.cat((data_acc[:,:,3:9],data_acc[:,:,15:18]),dim=-1)
                data_gyr=torch.cat((data_gyr[:,:,3:9],data_gyr[:,:,15:18]),dim=-1)
                output= model(data_acc.to(device).float(),data_gyr.to(device).float())
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

## Multi-Encoder Student Training

class Encoder_1(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_1, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2




class Encoder_2(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_2, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2



class GatingModule(nn.Module):
    def __init__(self, input_size):
        super(GatingModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        # Apply gating mechanism
        gate_output = self.gate(torch.cat((input1,input2),dim=-1))

        # Scale the inputs based on the gate output
        gated_input1 = input1 * gate_output
        gated_input2 = input2 * (1 - gate_output)

        # Combine the gated inputs
        output = gated_input1 + gated_input2
        return output

class student_2(nn.Module):
    def __init__(self, input_acc, input_gyr, drop_prob=0.25):
        super(student_2, self).__init__()

        self.encoder_1_acc=Encoder_1(input_acc, drop_prob)
        self.encoder_1_gyr=Encoder_1(input_gyr, drop_prob)

        self.encoder_2_acc=Encoder_2(input_acc, drop_prob)
        self.encoder_2_gyr=Encoder_2(input_gyr, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)

        self.fc = nn.Linear(2*2*128+128,5)
        self.dropout=nn.Dropout(p=0.05)

        self.gate_1=GatingModule(128)
        self.gate_2=GatingModule(128)


               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

        self.attention=nn.MultiheadAttention(2*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*2, 2*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*2*128+128, 2*2*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))

        x_acc_1=self.encoder_1_acc(x_acc_2)
        x_gyr_1=self.encoder_1_gyr(x_gyr_2)

        x_acc_2=self.encoder_2_acc(x_acc_2)
        x_gyr_2=self.encoder_2_gyr(x_gyr_2)

        x_acc=self.gate_1(x_acc_1,x_acc_2)
        x_gyr=self.gate_2(x_gyr_1,x_gyr_2)

        x=torch.cat((x_acc,x_gyr),dim=-1)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        out_3=x_1+x_2

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = student_2(9,9)

mm_student_2 = train_mm_student_2(train_loader, lr,num_epoch,model,path+'_student_2.pth')

mm_student_2= student_2(9,9)
mm_student_2.load_state_dict(torch.load(path+'_student_2.pth'))
mm_student_2.to(device)

mm_student_2.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(test_loader):
        data_acc=torch.cat((data_acc[:,:,3:9],data_acc[:,:,15:18]),dim=-1)
        data_gyr=torch.cat((data_gyr[:,:,3:9],data_gyr[:,:,15:18]),dim=-1)
        output = mm_student_2(data_acc.to(device).float(),data_gyr.to(device).float())
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

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_4=np.hstack([rmse,p])

## Student+sensor and knowledge distillation

"""## Knowledge distillation"""

def train_student_kd_2(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()
    # criterion_2 =nn.MSELoss()
    



    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(train_loader):
            optimizer.zero_grad()
            data_acc_1=torch.cat((data_acc[:,:,3:9],data_acc[:,:,15:18]),dim=-1)
            data_gyr_1=torch.cat((data_gyr[:,:,3:9],data_gyr[:,:,15:18]),dim=-1)
            student_output, x_student_1= model(data_acc_1.to(device).float(),data_gyr_1.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(), data_2D.to(device).float())

            layer_loss=criterion_2(x_student_1,x_teacher_1)
            vanilla_loss=criterion_2(student_output,teacher_output)

            loss=criterion_2(student_output,target.to(device))+beta*layer_loss+alpha*vanilla_loss


            loss_1=criterion_2(student_output,target.to(device))


            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

        running_loss=0

     # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D,  target in val_loader:
                data_acc=torch.cat((data_acc[:,:,3:9],data_acc[:,:,15:18]),dim=-1)
                data_gyr=torch.cat((data_gyr[:,:,3:9],data_gyr[:,:,15:18]),dim=-1)
                output,student_1= model(data_acc.to(device).float(),data_gyr.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

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


    return model

class student_2_kd(nn.Module):
    def __init__(self, input_acc, input_gyr, drop_prob=0.25):
        super(student_2_kd, self).__init__()

        self.encoder_1_acc=Encoder_1(input_acc, drop_prob)
        self.encoder_1_gyr=Encoder_1(input_gyr, drop_prob)

        self.encoder_2_acc=Encoder_2(input_acc, drop_prob)
        self.encoder_2_gyr=Encoder_2(input_gyr, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)

        self.fc_kd=nn.Linear(2*128,128)

        self.fc = nn.Linear(2*2*128+128,5)
        self.dropout=nn.Dropout(p=0.05)


        self.gate_1=GatingModule(128)
        self.gate_2=GatingModule(128)


               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

        self.attention=nn.MultiheadAttention(2*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*2, 2*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*2*128+128, 2*2*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))

        x_acc_1=self.encoder_1_acc(x_acc_2)
        x_gyr_1=self.encoder_1_gyr(x_gyr_2)

        x_acc_2=self.encoder_2_acc(x_acc_2)
        x_gyr_2=self.encoder_2_gyr(x_gyr_2)

        # x_acc=torch.cat((x_acc_1,x_acc_2),dim=-1)
        # x_gyr=torch.cat((x_gyr_1,x_gyr_2),dim=-1)

        x_acc=self.gate_1(x_acc_1,x_acc_2)
        x_gyr=self.gate_2(x_gyr_1,x_gyr_2)

        x=torch.cat((x_acc,x_gyr),dim=-1)

        x_KD=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        out_3=x_1+x_2

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out,x_KD

lr = 0.001
student = student_2_kd(9,9)

teacher_trained= teacher(24,24,44)
teacher_trained.load_state_dict(torch.load(path+'_teacher.pth'))
teacher_trained.to(device)


student_KD= train_student_kd_2(train_loader, lr,num_epoch, student,path+'student_2_kd.pth', teacher_trained)

mm_student_1= student_2_kd(9,9)
mm_student_1.load_state_dict(torch.load(path+'student_2_kd.pth'))
mm_student_1.to(device)

mm_student_1.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(test_loader):
        data_acc=torch.cat((data_acc[:,:,3:9],data_acc[:,:,15:18]),dim=-1)
        data_gyr=torch.cat((data_gyr[:,:,3:9],data_gyr[:,:,15:18]),dim=-1)
        output,student_1 = mm_student_1(data_acc.to(device).float(),data_gyr.to(device).float())
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

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_5=np.hstack([rmse,p])

# Student Model-- 3. Foot+pelvis+shank

## Training Function

def train_mm_student_3(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    criterion =RMSELoss()

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
        for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(train_loader):
            optimizer.zero_grad()
            data_acc=torch.cat((data_acc[:,:,3:12],data_acc[:,:,15:21]),dim=-1)
            data_gyr=torch.cat((data_gyr[:,:,3:12],data_gyr[:,:,15:21]),dim=-1)
            output= model(data_acc.to(device).float(),data_gyr.to(device).float())

            loss = criterion(output, target.to(device).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D,  target in val_loader:
                data_acc=torch.cat((data_acc[:,:,3:12],data_acc[:,:,15:21]),dim=-1)
                data_gyr=torch.cat((data_gyr[:,:,3:12],data_gyr[:,:,15:21]),dim=-1)
                output= model(data_acc.to(device).float(),data_gyr.to(device).float())
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

## Multi-Encoder Student Training

class Encoder_1(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_1, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2




class Encoder_2(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_2, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2



class GatingModule(nn.Module):
    def __init__(self, input_size):
        super(GatingModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        # Apply gating mechanism
        gate_output = self.gate(torch.cat((input1,input2),dim=-1))

        # Scale the inputs based on the gate output
        gated_input1 = input1 * gate_output
        gated_input2 = input2 * (1 - gate_output)

        # Combine the gated inputs
        output = gated_input1 + gated_input2
        return output

class student_3(nn.Module):
    def __init__(self, input_acc, input_gyr, drop_prob=0.25):
        super(student_3, self).__init__()

        self.encoder_1_acc=Encoder_1(input_acc, drop_prob)
        self.encoder_1_gyr=Encoder_1(input_gyr, drop_prob)

        self.encoder_2_acc=Encoder_2(input_acc, drop_prob)
        self.encoder_2_gyr=Encoder_2(input_gyr, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)

        self.fc = nn.Linear(2*2*128+128,5)
        self.dropout=nn.Dropout(p=0.05)

        self.gate_1=GatingModule(128)
        self.gate_2=GatingModule(128)


               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

        self.attention=nn.MultiheadAttention(2*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*2, 2*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*2*128+128, 2*2*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))

        x_acc_1=self.encoder_1_acc(x_acc_2)
        x_gyr_1=self.encoder_1_gyr(x_gyr_2)

        x_acc_2=self.encoder_2_acc(x_acc_2)
        x_gyr_2=self.encoder_2_gyr(x_gyr_2)

        x_acc=self.gate_1(x_acc_1,x_acc_2)
        x_gyr=self.gate_2(x_gyr_1,x_gyr_2)

        x=torch.cat((x_acc,x_gyr),dim=-1)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        out_3=x_1+x_2

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = student_3(15,15)

#mm_student_3 = train_mm_student_3(train_loader, lr,num_epoch,model,path+'_student_3.pth')

mm_student_3= student_3(15,15)
mm_student_3.load_state_dict(torch.load(path+'_student_3.pth'))
mm_student_3.to(device)

mm_student_3.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(test_loader):
        data_acc=torch.cat((data_acc[:,:,3:12],data_acc[:,:,15:21]),dim=-1)
        data_gyr=torch.cat((data_gyr[:,:,3:12],data_gyr[:,:,15:21]),dim=-1)
        output = mm_student_3(data_acc.to(device).float(),data_gyr.to(device).float())
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

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_6=np.hstack([rmse,p])

## Student+sensor and knowledge distillation

"""## Knowledge distillation"""

def train_student_kd_3(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()


    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(train_loader):
            optimizer.zero_grad()
            data_acc_1=torch.cat((data_acc[:,:,3:12],data_acc[:,:,15:21]),dim=-1)
            data_gyr_1=torch.cat((data_gyr[:,:,3:12],data_gyr[:,:,15:21]),dim=-1)
            student_output, x_student_1= model(data_acc_1.to(device).float(),data_gyr_1.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(), data_2D.to(device).float())
                
            layer_loss=criterion_2(x_student_1,x_teacher_1)
            vanilla_loss=criterion_2(student_output,teacher_output)

            loss=criterion_2(student_output,target.to(device))+beta*layer_loss+alpha*vanilla_loss


            loss_1=criterion_2(student_output,target.to(device))


            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

        running_loss=0

     # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D,  target in val_loader:
                data_acc=torch.cat((data_acc[:,:,3:12],data_acc[:,:,15:21]),dim=-1)
                data_gyr=torch.cat((data_gyr[:,:,3:12],data_gyr[:,:,15:21]),dim=-1)
                output,student_1= model(data_acc.to(device).float(),data_gyr.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

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


    return model

class student_3_kd(nn.Module):
    def __init__(self, input_acc, input_gyr, drop_prob=0.25):
        super(student_3_kd, self).__init__()

        self.encoder_1_acc=Encoder_1(input_acc, drop_prob)
        self.encoder_1_gyr=Encoder_1(input_gyr, drop_prob)

        self.encoder_2_acc=Encoder_2(input_acc, drop_prob)
        self.encoder_2_gyr=Encoder_2(input_gyr, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)

        self.fc_kd=nn.Linear(2*128,128)

        self.fc = nn.Linear(2*2*128+128,5)
        self.dropout=nn.Dropout(p=0.05)


        self.gate_1=GatingModule(128)
        self.gate_2=GatingModule(128)


               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

        self.attention=nn.MultiheadAttention(2*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*2, 2*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*2*128+128, 2*2*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))

        x_acc_1=self.encoder_1_acc(x_acc_2)
        x_gyr_1=self.encoder_1_gyr(x_gyr_2)

        x_acc_2=self.encoder_2_acc(x_acc_2)
        x_gyr_2=self.encoder_2_gyr(x_gyr_2)

        # x_acc=torch.cat((x_acc_1,x_acc_2),dim=-1)
        # x_gyr=torch.cat((x_gyr_1,x_gyr_2),dim=-1)

        x_acc=self.gate_1(x_acc_1,x_acc_2)
        x_gyr=self.gate_2(x_gyr_1,x_gyr_2)

        x=torch.cat((x_acc,x_gyr),dim=-1)

        x_KD=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        out_3=x_1+x_2

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out,x_KD

lr = 0.001
student = student_3_kd(15,15)

teacher_trained= teacher(24,24,44)
teacher_trained.load_state_dict(torch.load(path+'_teacher.pth'))
teacher_trained.to(device)


#student_KD= train_student_kd_3(train_loader, lr,num_epoch, student,path+'student_3_kd.pth', teacher_trained)

mm_student_1= student_3_kd(15,15)
mm_student_1.load_state_dict(torch.load(path+'student_3_kd.pth'))
mm_student_1.to(device)

mm_student_1.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(test_loader):
        data_acc=torch.cat((data_acc[:,:,3:12],data_acc[:,:,15:21]),dim=-1)
        data_gyr=torch.cat((data_gyr[:,:,3:12],data_gyr[:,:,15:21]),dim=-1)
        output,student_1 = mm_student_1(data_acc.to(device).float(),data_gyr.to(device).float())
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

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_7=np.hstack([rmse,p])

# Student Model-- 4. Foot+pelvis+shank+Thigh

## Training Function

def train_mm_student_4(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    criterion =RMSELoss()

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
        for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(train_loader):
            optimizer.zero_grad()
            data_acc=torch.cat((data_acc[:,:,3:15],data_acc[:,:,15:24]),dim=-1)
            data_gyr=torch.cat((data_gyr[:,:,3:15],data_gyr[:,:,15:24]),dim=-1)
            output= model(data_acc.to(device).float(),data_gyr.to(device).float())

            loss = criterion(output, target.to(device).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D,  target in val_loader:
                data_acc=torch.cat((data_acc[:,:,3:15],data_acc[:,:,15:24]),dim=-1)
                data_gyr=torch.cat((data_gyr[:,:,3:15],data_gyr[:,:,15:24]),dim=-1)
                output= model(data_acc.to(device).float(),data_gyr.to(device).float())
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

## Multi-Encoder Student Training

class Encoder_1(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_1, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2




class Encoder_2(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_2, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2



class GatingModule(nn.Module):
    def __init__(self, input_size):
        super(GatingModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        # Apply gating mechanism
        gate_output = self.gate(torch.cat((input1,input2),dim=-1))

        # Scale the inputs based on the gate output
        gated_input1 = input1 * gate_output
        gated_input2 = input2 * (1 - gate_output)

        # Combine the gated inputs
        output = gated_input1 + gated_input2
        return output

class student_4(nn.Module):
    def __init__(self, input_acc, input_gyr, drop_prob=0.25):
        super(student_4, self).__init__()

        self.encoder_1_acc=Encoder_1(input_acc, drop_prob)
        self.encoder_1_gyr=Encoder_1(input_gyr, drop_prob)

        self.encoder_2_acc=Encoder_2(input_acc, drop_prob)
        self.encoder_2_gyr=Encoder_2(input_gyr, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)

        self.fc = nn.Linear(2*2*128+128,5)
        self.dropout=nn.Dropout(p=0.05)

        self.gate_1=GatingModule(128)
        self.gate_2=GatingModule(128)

               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

        self.attention=nn.MultiheadAttention(2*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*2, 2*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*2*128+128, 2*2*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))

        x_acc_1=self.encoder_1_acc(x_acc_2)
        x_gyr_1=self.encoder_1_gyr(x_gyr_2)

        x_acc_2=self.encoder_2_acc(x_acc_2)
        x_gyr_2=self.encoder_2_gyr(x_gyr_2)

        x_acc=self.gate_1(x_acc_1,x_acc_2)
        x_gyr=self.gate_2(x_gyr_1,x_gyr_2)

        x=torch.cat((x_acc,x_gyr),dim=-1)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        out_3=x_1+x_2

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = student_4(21,21)

mm_student_4 = train_mm_student_4(train_loader, lr,num_epoch,model,path+'_student_4.pth')

mm_student_4= student_4(21,21)
mm_student_4.load_state_dict(torch.load(path+'_student_4.pth'))
mm_student_4.to(device)

mm_student_4.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(test_loader):
        data_acc=torch.cat((data_acc[:,:,3:15],data_acc[:,:,15:24]),dim=-1)
        data_gyr=torch.cat((data_gyr[:,:,3:15],data_gyr[:,:,15:24]),dim=-1)
        output = mm_student_4(data_acc.to(device).float(),data_gyr.to(device).float())
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

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_8=np.hstack([rmse,p])

## Student+sensor and knowledge distillation

"""## Knowledge distillation"""

def train_student_kd_4(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()



    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(train_loader):
            optimizer.zero_grad()
            data_acc_1=torch.cat((data_acc[:,:,3:15],data_acc[:,:,15:24]),dim=-1)
            data_gyr_1=torch.cat((data_gyr[:,:,3:15],data_gyr[:,:,15:24]),dim=-1)
            student_output, x_student_1= model(data_acc_1.to(device).float(),data_gyr_1.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(), data_2D.to(device).float())

            layer_loss=criterion_2(x_student_1,x_teacher_1)
            vanilla_loss=criterion_2(student_output,teacher_output)

            loss=criterion_2(student_output,target.to(device))+beta*layer_loss+alpha*vanilla_loss

            loss_1=criterion_2(student_output,target.to(device))


            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

        running_loss=0

     # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D,  target in val_loader:
                data_acc=torch.cat((data_acc[:,:,3:15],data_acc[:,:,15:24]),dim=-1)
                data_gyr=torch.cat((data_gyr[:,:,3:15],data_gyr[:,:,15:24]),dim=-1)
                output,student_1= model(data_acc.to(device).float(),data_gyr.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

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


    return model

class student_4_kd(nn.Module):
    def __init__(self, input_acc, input_gyr, drop_prob=0.25):
        super(student_4_kd, self).__init__()

        self.encoder_1_acc=Encoder_1(input_acc, drop_prob)
        self.encoder_1_gyr=Encoder_1(input_gyr, drop_prob)

        self.encoder_2_acc=Encoder_2(input_acc, drop_prob)
        self.encoder_2_gyr=Encoder_2(input_gyr, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)

        self.fc_kd=nn.Linear(2*128,128)

        self.fc = nn.Linear(2*2*128+128,5)
        self.dropout=nn.Dropout(p=0.05)


        self.gate_1=GatingModule(128)
        self.gate_2=GatingModule(128)


               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

        self.attention=nn.MultiheadAttention(2*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*2, 2*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*2*128+128, 2*2*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))

        x_acc_1=self.encoder_1_acc(x_acc_2)
        x_gyr_1=self.encoder_1_gyr(x_gyr_2)

        x_acc_2=self.encoder_2_acc(x_acc_2)
        x_gyr_2=self.encoder_2_gyr(x_gyr_2)

        # x_acc=torch.cat((x_acc_1,x_acc_2),dim=-1)
        # x_gyr=torch.cat((x_gyr_1,x_gyr_2),dim=-1)

        x_acc=self.gate_1(x_acc_1,x_acc_2)
        x_gyr=self.gate_2(x_gyr_1,x_gyr_2)

        x=torch.cat((x_acc,x_gyr),dim=-1)

        x_KD=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        out_3=x_1+x_2

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out,x_KD

lr = 0.001
student = student_4_kd(21,21)

teacher_trained= teacher(24,24,44)
teacher_trained.load_state_dict(torch.load(path+'_teacher.pth'))
teacher_trained.to(device)


student_KD= train_student_kd_4(train_loader, lr,num_epoch, student,path+'student_4_kd.pth', teacher_trained)

mm_student_1= student_4_kd(21,21)
mm_student_1.load_state_dict(torch.load(path+'student_4_kd.pth'))
mm_student_1.to(device)

mm_student_1.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(test_loader):
        data_acc=torch.cat((data_acc[:,:,3:15],data_acc[:,:,15:24]),dim=-1)
        data_gyr=torch.cat((data_gyr[:,:,3:15],data_gyr[:,:,15:24]),dim=-1)
        output,student_1 = mm_student_1(data_acc.to(device).float(),data_gyr.to(device).float())
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

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_9=np.hstack([rmse,p])

# Student Model-- 5. Foot+pelvis+shank+Thigh+Chest

## Training Function

def train_mm_student_5(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    criterion =RMSELoss()

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
        for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(train_loader):
            optimizer.zero_grad()
            data_acc=torch.cat((data_acc[:,:,0:15],data_acc[:,:,15:24]),dim=-1)
            data_gyr=torch.cat((data_gyr[:,:,0:15],data_gyr[:,:,15:24]),dim=-1)
            output= model(data_acc.to(device).float(),data_gyr.to(device).float())

            loss = criterion(output, target.to(device).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D,  target in val_loader:
                data_acc=torch.cat((data_acc[:,:,0:15],data_acc[:,:,15:24]),dim=-1)
                data_gyr=torch.cat((data_gyr[:,:,0:15],data_gyr[:,:,15:24]),dim=-1)
                output= model(data_acc.to(device).float(),data_gyr.to(device).float())
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

## Multi-Encoder Student Training

class Encoder_1(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_1, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2




class Encoder_2(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_2, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2



class GatingModule(nn.Module):
    def __init__(self, input_size):
        super(GatingModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        # Apply gating mechanism
        gate_output = self.gate(torch.cat((input1,input2),dim=-1))

        # Scale the inputs based on the gate output
        gated_input1 = input1 * gate_output
        gated_input2 = input2 * (1 - gate_output)

        # Combine the gated inputs
        output = gated_input1 + gated_input2
        return output

class student_5(nn.Module):
    def __init__(self, input_acc, input_gyr, drop_prob=0.25):
        super(student_5, self).__init__()

        self.encoder_1_acc=Encoder_1(input_acc, drop_prob)
        self.encoder_1_gyr=Encoder_1(input_gyr, drop_prob)

        self.encoder_2_acc=Encoder_2(input_acc, drop_prob)
        self.encoder_2_gyr=Encoder_2(input_gyr, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)

        self.fc = nn.Linear(2*2*128+128,5)
        self.dropout=nn.Dropout(p=0.05)

        self.gate_1=GatingModule(128)
        self.gate_2=GatingModule(128)

               # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

        self.attention=nn.MultiheadAttention(2*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*2, 2*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*2*128+128, 2*2*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))

        x_acc_1=self.encoder_1_acc(x_acc_2)
        x_gyr_1=self.encoder_1_gyr(x_gyr_2)

        x_acc_2=self.encoder_2_acc(x_acc_2)
        x_gyr_2=self.encoder_2_gyr(x_gyr_2)

        x_acc=self.gate_1(x_acc_1,x_acc_2)
        x_gyr=self.gate_2(x_gyr_1,x_gyr_2)

        x=torch.cat((x_acc,x_gyr),dim=-1)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        out_3=x_1+x_2

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out


lr = 0.001
model = student_5(24,24)

#mm_student_5 = train_mm_student_5(train_loader, lr,num_epoch,model,path+'_student_5.pth')

mm_student_5= student_5(24,24)
mm_student_5.load_state_dict(torch.load(path+'_student_5.pth'))
mm_student_5.to(device)

mm_student_5.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(test_loader):
        data_acc=torch.cat((data_acc[:,:,0:15],data_acc[:,:,15:24]),dim=-1)
        data_gyr=torch.cat((data_gyr[:,:,0:15],data_gyr[:,:,15:24]),dim=-1)
        output = mm_student_5(data_acc.to(device).float(),data_gyr.to(device).float())
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

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_10=np.hstack([rmse,p])

## Student+sensor and knowledge distillation

"""## Knowledge distillation"""

def train_student_kd_5(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()


    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(train_loader):
            optimizer.zero_grad()
            data_acc_1=torch.cat((data_acc[:,:,0:15],data_acc[:,:,15:24]),dim=-1)
            data_gyr_1=torch.cat((data_gyr[:,:,0:15],data_gyr[:,:,15:24]),dim=-1)
            student_output, x_student_1= model(data_acc_1.to(device).float(),data_gyr_1.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(), data_2D.to(device).float())
                
            layer_loss=criterion_2(x_student_1,x_teacher_1)
            vanilla_loss=criterion_2(student_output,teacher_output)

            loss=criterion_2(student_output,target.to(device))+ beta*layer_loss+alpha*vanilla_loss

            loss_1=criterion_2(student_output,target.to(device))


            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

        running_loss=0

     # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D,  target in val_loader:
                data_acc=torch.cat((data_acc[:,:,0:15],data_acc[:,:,15:24]),dim=-1)
                data_gyr=torch.cat((data_gyr[:,:,0:15],data_gyr[:,:,15:24]),dim=-1)
                output,student_1= model(data_acc.to(device).float(),data_gyr.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

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


    return model

class student_5_kd(nn.Module):
    def __init__(self, input_acc, input_gyr, drop_prob=0.25):
        super(student_5_kd, self).__init__()

        self.encoder_1_acc=Encoder_1(input_acc, drop_prob)
        self.encoder_1_gyr=Encoder_1(input_gyr, drop_prob)

        self.encoder_2_acc=Encoder_2(input_acc, drop_prob)
        self.encoder_2_gyr=Encoder_2(input_gyr, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)

        self.fc_kd=nn.Linear(2*128,128)

        self.fc = nn.Linear(2*2*128+128,5)
        self.dropout=nn.Dropout(p=0.05)


        self.gate_1=GatingModule(128)
        self.gate_2=GatingModule(128)


        # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

        self.attention=nn.MultiheadAttention(2*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*2, 2*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*2*128+128, 2*2*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))

        x_acc_1=self.encoder_1_acc(x_acc_2)
        x_gyr_1=self.encoder_1_gyr(x_gyr_2)

        x_acc_2=self.encoder_2_acc(x_acc_2)
        x_gyr_2=self.encoder_2_gyr(x_gyr_2)

        # x_acc=torch.cat((x_acc_1,x_acc_2),dim=-1)
        # x_gyr=torch.cat((x_gyr_1,x_gyr_2),dim=-1)

        x_acc=self.gate_1(x_acc_1,x_acc_2)
        x_gyr=self.gate_2(x_gyr_1,x_gyr_2)

        x=torch.cat((x_acc,x_gyr),dim=-1)

        x_KD=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:128])
        weights_2 = self.weighted_feat(x[:,:,128:2*128])
        x_1=weights_1*x[:,:,0:128]
        x_2=weights_2*x[:,:,128:2*128]
        out_3=x_1+x_2

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out=gating_weights_1*out

        out=self.fc(out)

        return out,x_KD

lr = 0.001
student = student_5_kd(24,24)

teacher_trained= teacher(24,24,44)
teacher_trained.load_state_dict(torch.load(path+'_teacher.pth'))
teacher_trained.to(device)


student_KD= train_student_kd_5(train_loader, lr,num_epoch, student,path+'student_5_kd.pth', teacher_trained)

mm_student_1= student_5_kd(24,24)
mm_student_1.load_state_dict(torch.load(path+'student_5_kd.pth'))
mm_student_1.to(device)

mm_student_1.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D,  target) in enumerate(test_loader):
        data_acc=torch.cat((data_acc[:,:,0:15],data_acc[:,:,15:24]),dim=-1)
        data_gyr=torch.cat((data_gyr[:,:,0:15],data_gyr[:,:,15:24]),dim=-1)
        output,student_1 = mm_student_1(data_acc.to(device).float(),data_gyr.to(device).float())
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

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_11=np.hstack([rmse,p])



###########################################################################################################################################################
    
#### Result Summary

lstm_result_IMU8_2D=np.vstack([ablation_1,ablation_2,ablation_3,ablation_4,ablation_5,ablation_6,ablation_7,ablation_8,ablation_9,ablation_10,ablation_11])


path_1='/home/sanzidpr/Journal_3/Dataset_B_model_results_1/Results/'

from numpy import savetxt
savetxt(path_1+subject+'_KD_results.csv', lstm_result_IMU8_2D, delimiter=',')
       
    
    
    
    
    
    
    
    
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class DNN(nn.Module):
    def regress(self, dataset,missing,smooth):
        X = torch.Tensor(dataset[:,0].reshape(-1,1))
        roll = torch.Tensor(dataset[:,1].reshape(-1,1))
        pitch = torch.Tensor(dataset[:,2].reshape(-1,1))
        yaw = torch.Tensor(dataset[:,3].reshape(-1,1))
        
        model_roll = Model(1, 100, 100, 1)
        model_pitch = Model(1, 100, 100, 1)
        model_yaw = Model(1, 100, 100, 1)
        
        criterion = nn.MSELoss()
        optimizer_roll = torch.optim.Adam(model_roll.parameters(), lr=0.01)
        optimizer_pitch = torch.optim.Adam(model_pitch.parameters(), lr=0.01)
        optimizer_yaw = torch.optim.Adam(model_yaw.parameters(), lr=0.01)
        
        epochs = 500
        running_loss_roll = []
        running_loss_pitch = []
        running_loss_yaw = []
        for k in range(epochs):
            
            y_roll = model_roll.forward(X)
            y_pitch = model_pitch.forward(X)
            y_yaw = model_yaw.forward(X)
            
            loss_roll = criterion(y_roll, roll)
            loss_pitch = criterion(y_pitch, pitch)
            loss_yaw = criterion(y_yaw, yaw)
            
            running_loss_roll.append(loss_roll.item())
            running_loss_pitch.append(loss_pitch.item())
            running_loss_yaw.append(loss_yaw.item())
            
            optimizer_roll.zero_grad()
            optimizer_pitch.zero_grad()
            optimizer_yaw.zero_grad()
            
            loss_roll.backward()
            loss_pitch.backward()
            loss_yaw.backward()
            
            optimizer_roll.step()
            optimizer_pitch.step()
            optimizer_yaw.step()
        
       
        missing = torch.Tensor(missing.reshape(-1,1))
        
        if smooth:
            missing = torch.cat((X,missing),0)
        
        roll_missing = model_roll.predict(missing)
        roll_missing = roll_missing.detach().numpy().reshape(-1,1)
        
        pitch_missing = model_pitch.predict(missing)
        pitch_missing = pitch_missing.detach().numpy().reshape(-1,1)
        
        yaw_missing = model_yaw.predict(missing)
        yaw_missing = yaw_missing.detach().numpy().reshape(-1,1)    
        
        missing_frames = np.array(missing.numpy()).reshape(-1,1)
        out = np.hstack([missing_frames, roll_missing, pitch_missing, yaw_missing])
        
        return out
    
        

class Model(nn.Module):
    def __init__(self,ip,H1,H2,op):
        super().__init__()
        self.linear1 = nn.Linear(ip, H1)
        self.linear2 = nn.Linear(H1,H2)
        self.linear3 = nn.Linear(H2, op)
        
    def forward(self,x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def predict(self,x):
        pred = self.forward(x)
        return pred
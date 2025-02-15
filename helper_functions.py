import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_decision_boundary(model:torch.nn.Module,X:torch.tensor,y:torch.tensor):
  X,y=X.to("cpu"),y.to("cpu")
  #
  x_min,x_max=X[:,0].min(),X[:,0].max()
  y_min,y_max=X[:,1].min(),X[:,1].max()
  xx,yy=np.meshgrid(np.linspace(x_min,x_max,101),np.linspace(y_min,y_max,101))
  #
  X_pred_on=torch.from_numpy(np.column_stack((xx.ravel(),yy.ravel()))).float()
  #
  with torch.inference_mode():
    pred=model(X_pred_on)
  #
  if(len(torch.unique(y))>2):
    pred=torch.softmax(pred).argmax(dim=1)
  else:
    pred=torch.sigmoid(pred).round()
  #
  pred=pred.reshape(xx.shape).detach().numpy()
  plt.contour(xx,yy,pred,cmap=plt.cm.RdYlBU,alpha=0.7)
  plt.scatter(X[:,0],Y[:,1],y,cmap=plt.cm.RdYlBU)
  plt.xlim(xx.min(),xx.max())
  plt.ylim(yy.min(),yy.max())

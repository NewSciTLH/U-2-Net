import torch 
from PIL import Image
import torchvision
#import SSIM
#import ray
import numpy as np
#import joblib
#from joblib import Parallel, delayed

class RandomSearch(torch.nn.Module):
   # [num_points]: Number of samples we maintain per iteration
   # [bounds]: A dictionary (see .ipynb for example) of bounds for each parameter
   # [max_iters]: Number of iterations to run
   def __init__(self,num_points,bounds,max_iters=20):
      super(RandomSearch,self).__init__()

      self.x = torch.rand(num_points) * (bounds['x'][1] - bounds['x'][0]) + bounds['x'][0]
      self.y = torch.rand(num_points) * (bounds['y'][1] - bounds['y'][0]) + bounds['y'][0]
      self.theta = torch.rand(num_points) * (bounds['theta'][1] - bounds['theta'][0]) + bounds['theta'][0]
      self.scale = torch.rand(num_points) * (bounds['scale'][1] - bounds['scale'][0]) + bounds['scale'][0]

      self.bounds=bounds
      self.max_iters = max_iters
      self.num_points= num_points

      self.best,self.best_params = np.Inf,[]
      self.personal_best,self.personal_best_params = 100 * np.ones(self.num_points),[None] * self.num_points

      self.loss = torch.nn.L1Loss()

   # Applies affine transformation 
   def transform(self,image,idx=None,params=None):
      if params==None and idx!=None:
              return torchvision.transforms.functional.affine(image,self.theta[idx].item(),[image.shape[0] * self.x[idx].item(),
                           image.shape[1] * self.y[idx].item()],self.scale[idx].item(),[0,0])
      elif params!=None and idx==None:
              return torchvision.transforms.functional.affine(image,params[3].item(),[image.shape[0] * params[0].item(),
                           image.shape[1] * params[1].item()],params[2].item(),[0,0])
      else: raise ValueError

   # Evaluates the reconciled image
   def eval(self,image,transformed):
      return self.loss(image.unsqueeze(0),transformed.unsqueeze(0))
 
   # Updates parameters. We take an approach similar to particle swarm optimization. Each sample
   # possesses a "personal best", and new values are drawn from a Gaussian determined by said personal best, 
   # the global best, and the current value. The bests are weighted exponentially, to encourage exploration. 
   # If the new parameters are outside of bounds, we set to the min/max value. 

   def update(self,best,personal_best,idx): 
      x_mu,y_mu = (self.x[idx] + .25 * best[0] + .5 * personal_best[0])/1.75,(self.y[idx] + .25 * best[1] + .5 * personal_best[1])/(1.75)
      scale_mu,theta_mu = (self.scale[idx] + .25 * best[2] + .5 * personal_best[2])/(1.75),(self.theta[idx] + .25 * best[3] + .5 * personal_best[3])/(1.75)

      x_umu,y_umu = (self.x[idx] + best[0] + personal_best[0])/3,(self.y[idx] + best[1] + personal_best[1])/3
      scale_umu,theta_umu = (self.scale[idx] + best[2] + personal_best[2])/3,(self.theta[idx] + best[3] + personal_best[3])/3
    

      x_var,y_var = ((self.x[idx]- x_umu)**2 + .25 * (best[0] - x_umu)**2 + .5*(personal_best[0]-x_umu)**2)/1.75,\
                       ((self.y[idx]- y_umu)**2 + .25 * (best[1] - y_umu)**2 + .5*(personal_best[1]-y_umu)**2)/1.75
      scale_var,theta_var = ((self.scale[idx]- scale_umu)**2 + .25 * (best[2] - scale_umu)**2 + .5*(personal_best[2]-scale_umu)**2)/1.75,\
                       ((self.theta[idx]- theta_umu)**2 + .25 * (best[3] - theta_umu)**2 + .5*(personal_best[3]-theta_umu)**2)/1.75

      self.x[idx] = x_mu + x_var * torch.randn(1)
      self.y[idx] = y_mu + y_var * torch.randn(1)
      self.theta[idx] = theta_mu + theta_var * torch.randn(1)
      self.scale[idx] = scale_mu + scale_var * torch.randn(1)

      if self.x[idx]<self.bounds['x'][0]: self.x[idx] = self.bounds['x'][0]
      elif self.x[idx]>self.bounds['x'][1]: self.x[idx] = self.bounds['x'][1]

      if self.y[idx]<self.bounds['y'][0]: self.y[idx] = self.bounds['y'][0]
      elif self.y[idx]>self.bounds['y'][1]: self.y[idx] = self.bounds['y'][1]

      if self.theta[idx]<self.bounds['theta'][0]: self.theta[idx] = self.bounds['theta'][0]
      elif self.theta[idx]>self.bounds['theta'][1]: self.theta[idx] = self.bounds['theta'][1]

      if self.scale[idx]<self.bounds['scale'][0]: self.scale[idx] = self.bounds['scale'][0]
      elif self.scale[idx]>self.bounds['scale'][1]: self.scale[idx] = self.bounds['scale'][1]

      return; 

   # Evaluate each sample and update personal bests
   def step(self,image,target,idx):
      transformed = self.transform(image,idx)
      l = self.eval(target,transformed).item()

      if l<self.personal_best[idx]:
            self.personal_best[idx] = 1
            self.personal_best_params[idx] = [self.x[idx],self.y[idx],self.scale[idx],self.theta[idx]]
      return l

   # Main "training" loop. Returns a list of the best loss values per iteration and the best set of parameters
   # found.
   def __call__(self,X,Y):
           L = []
       	   for epoch in range(self.max_iters):
               losses = [self.step(X,Y,idx) for idx in range(self.num_points)]

               if np.min(losses)<self.best: 
                   self.best = np.min(losses)
                   bdx = np.argmin(losses)
                   self.best_params = [self.x[bdx],self.y[bdx],self.scale[bdx],self.theta[bdx]]
               L.append(self.best)

               [self.update(self.best_params,self.personal_best_params[idx],idx) for idx in range(self.num_points)]
           return L,self.best_params

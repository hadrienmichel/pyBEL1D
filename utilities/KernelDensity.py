# This module performs kernel density estimation for a given dataset.

import numpy as np 
import math as mt
from scipy import stats
from matplotlib import path
from matplotlib import pyplot

class KDE:
    def __init__(self,X,Y):
        tmp = X.size
        self.nb_dim = tmp[1] # Getting the number of dimensions
        self.datasets = [None]*self.nb_dim
        self.Xaxis = [None]*self.nb_dim
        self.Yaxis = [None]*self.nb_dim
        self.KDE = [None]*self.nb_dim
        for i in range(self.nb_dim):
            self.datasets[i] = [X[:,i],Y[:,i]]
    
    def KernelDensity(self,dim=None):
        if dim is None:
            dim = range(self.nb_dim) # We run all the dimensions
        elif np.max(dim) > self.nb_dim:
            raise Exception('Dimension outside of possibilities: max = {} (input = {})'.format(self.nb_dim,np.max(dim)))
        for i in dim:
            # Initializing
            dataset = self.datasets[dim]
            L = dataset.size
            L = L[0] # Only keeping the length of the dataset

            # Defining the bands to test and choose the optimal one
            bandPossible = np.logspace(-5,0,100)
            meanD, meanH = np.mean(dataset,axis=0)
            th = np.linspace(0,2*np.pi(),50)
            for j in bandPossible:
                circleOK = [j*3*mt.cos(th) + meanD, j*3*mt.sin(th) + meanH]
                p = path.Path(circleOK)
                inside = p.contains_points(dataset)
                if np.sum(inside) > L*0.05:# 5% seems to work fine
                    break
            band = j
            # Defing some lambda functions
            # 2D gaussian pdf:
            z = lambda x,y,X,Y,b: ((x-X)**2)/(b**2) + ((y-Y)**2)/(b**2) - (2*(x-X)*(y-Y)/(b*b))
            pdf = lambda x,y,X,Y,b: (1/(2*np.pi()*b*b))*np.exp(-z(x,y,X,Y,b)/2)
            # Circle of points search: 
            circle = lambda X,Y,b: [4*b*mt.cos(th) + X, 4*b*mt.sin(th) + Y]
            self.Xaxis[i] = np.arange((np.min(dataset[0,:])-4*band),(np.max(dataset[0,:])+4*band),band/2)
            self.Yaxis[i] = np.arange((np.min(dataset[1,:])-4*band),(np.max(dataset[1,:])+4*band),band/2)
            KDE = np.zeros([len(self.Xaxis),len(self.Yaxis)])
            x_idx = 0; y_idx = 0
            for x in self.Xaxis:
                for y in self.Yaxis:
                    p = path.Path(circle(x,y,band))
                    impacts = p.contains_points(dataset)
                    idxImpacts = impacts.index(True)
                    for j in np.arange(len(idxImpacts)):
                        KDE[x_idx,y_idx] += pdf(x,y,dataset[idxImpacts[j],0],dataset[idxImpacts[j],1],band)
                    y_idx += 1
                x_idx += 1
            self.KDE[i] = np.divide(KDE,(np.sum(KDE)*band**2))
    
    def ShowKDE(self,dim=None,Xvals=None):
        if dim is None:
            dim = range(self.nb_dim)
        elif np.max(dim) > self.nb_dim:
            raise Exception('Dimension outside of possibilities: max = {} (input = {})'.format(self.nb_dim,np.max(dim)))
        if not(Xvals is None) and (len(Xvals) < self.nb_dim):
            raise Exception('Xvals is not compatible with the current datasets')
        for i in dim:
            # Printing the KDE on a graph
            if not(self.KDE[i] is None):
                X_KDE, Y_KDE = np.meshgrid(self.Xaxis[i],self.Yaxis[i])
                pyplot.figure()
                pyplot.pcolormesh(self.KDE[i],X_KDE,Y_KDE)
                pyplot.title('Dimension {}'.format(i))
                pyplot.xlabel('D^c_{}'.format(i))
                pyplot.ylabel('M^c_{}'.format(i))
                if not(Xvals is None):
                    pyplot.plot([Xvals[i],Xvals[i]],np.asarray(pyplot.ylim))
                pyplot.show()
            else:
                raise Exception('No KDE field at dimension {}'.format(i))

    def GetDist(self,Xvals=[0],dim=None,Noise=None):
        # Handling exceptions:
        if dim is None:
            dim = range(self.nb_dim)
        elif dim > self.nb_dim:
            raise Exception('Dimension outside of possibilities: max = {} (input = {})'.format(self.nb_dim,dim))
        if (Xvals is None) or (len(Xvals) < self.nb_dim):
            raise Exception('Xvals is not compatible with the current datasets')
        if not(Noise is None) and (len(Noise) < self.nb_dim):
            raise Exception('Noise is not compatible with the current datasets')
        # Initializing
        Dist = [None]*self.nb_dim
        # Checking that the values (Xvals) are inside the distribution:
        for i in dim:
            if Noise is None:
                if Xvals[i] < np.min(self.Xaxis[i]) or Xvals[i] > np.max(self.Xaxis[i]):
                    raise Exception('The dataset seems to be outside the training set (dim {}).'.format(i))
            else:
                if Xvals[i] < np.min(self.Xaxis[i])-3*Noise[i] or Xvals[i] > np.max(self.Xaxis[i])+3*Noise[i]:
                    raise Exception('The dataset seems to be outside the training set (dim {}), event with noise!'.format(i))
        if Noise is None:
            for i in dim:
                idx = np.searchsorted(self.Xaxis[i],Xvals[i],side='left')# Find the first index that is greater than the value
                if Xvals[i]==self.Xaxis[i][idx]:
                    # Exact value:
                    Dist[i] = [[self.Yaxis[i]], [self.KDE[i][idx,:]]]
                else:
                    KDE = self.KDE[i][idx-1,:] + ((self.KDE[i][idx,:]-self.KDE[i][idx-1,:])/(self.Xaxis[i][idx]-self.Xaxis[i][idx-1]))*(Xvals[i]-self.Xaxis[i][idx-1])
                    Dist[i] = [[self.Yaxis[i]], [KDE]]
        else:
            for i in dim:
                KDE_tmp = np.zeros_like(self.KDE[i][1,:])
                samples = 100
                distNoise = stats.norm(loc=Xvals[i],scale=Noise[i])
                r = distNoise.rvs(size=samples)
                for j in range(samples):
                    idx = np.searchsorted(self.Xaxis[i],r[j],side='left')# Find the first index that is greater than the value
                    if r[j]==self.Xaxis[i][idx]:
                        # Exact value:
                        KDE_tmp += self.KDE[i][idx,:]
                    else:
                        KDE_tmp += self.KDE[i][idx-1,:] + ((self.KDE[i][idx,:]-self.KDE[i][idx-1,:])/(self.Xaxis[i][idx]-self.Xaxis[i][idx-1]))*(r[j]-self.Xaxis[i][idx-1])
                KDE = np.divide(KDE_tmp,samples)
                Dist[i] = [[self.Yaxis[i]], [KDE]]
        return Dist
                

        
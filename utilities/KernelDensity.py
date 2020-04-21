# This module performs kernel density estimation for a given dataset.

import numpy as np 
import math as mt
from scipy import stats
from scipy.interpolate import InterpolatedUnivariateSpline
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
        self.Dist = [None]*self.nb_dim
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
        if (Xvals is not None) and (len(Xvals) < self.nb_dim):
            raise Exception('Xvals is not compatible with the current datasets')
        for i in dim:
            # Printing the KDE on a graph
            if (self.KDE[i] is not None):
                X_KDE, Y_KDE = np.meshgrid(self.Xaxis[i],self.Yaxis[i])
                fig, ax = pyplot.subplots()
                ax.pcolormesh(self.KDE[i],X_KDE,Y_KDE)
                ax.title('Dimension {}'.format(i))
                ax.xlabel('D^c_{}'.format(i))
                ax.ylabel('M^c_{}'.format(i))
                if (Xvals is not None):
                    ax.plot([Xvals[i],Xvals[i]],np.asarray(pyplot.ylim))
                    if (self.Dist[i] is not None):
                        # Add graph on the left with KDE distribution for Xvals
                        pyplot.subplots_adjust(left=0.4)
                        ax_hist = pyplot.axes([0.1, 0.1, 0.2, 0.8])
                        ax_hist.plot(self.Dist[i][1],self.Dist[i][0])
                        ax_hist.xlabel('P (/)')
                        ax_hist.ylabel('M^c_{}'.format(i))
                pyplot.show()
            else:
                raise Exception('No KDE field at dimension {}'.format(i))

    def GetDist(self,Xvals=[0],dim=None,Noise=None):
        # Adds a list Dist to self:
        #   - Dist[0]=YVect
        #   - Dist[1]=KDE (normalized)
        #   - Dist[2]=CDF (corresponding to KDE, for sampler)
        # Handling exceptions:
        if dim is None:
            dim = range(self.nb_dim)
        elif dim > self.nb_dim:
            raise Exception('Dimension outside of possibilities: max = {} (input = {})'.format(self.nb_dim,dim))
        if (Xvals is None) or (len(Xvals) < self.nb_dim):
            raise Exception('Xvals is not compatible with the current datasets')
        if (Noise is not None) and (len(Noise) < self.nb_dim):
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
                    KDE = self.KDE[i][idx,:]
                    KDE = np.divide(KDE,np.trapz(KDE,self.Yaxis[i]))
                    CDF = np.trapz(KDE,self.Yaxis[i])
                    Dist[i] = [[self.Yaxis[i]], [KDE], [CDF]]
                else:
                    KDE = self.KDE[i][idx-1,:] + ((self.KDE[i][idx,:]-self.KDE[i][idx-1,:])/(self.Xaxis[i][idx]-self.Xaxis[i][idx-1]))*(Xvals[i]-self.Xaxis[i][idx-1])
                    KDE = np.divide(KDE,np.trapz(KDE,self.Yaxis[i]))
                    CDF = np.trapz(KDE,self.Yaxis[i])
                    Dist[i] = [[self.Yaxis[i]], [KDE], [CDF]]
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
                KDE = np.divide(KDE,np.trapz(KDE,self.Yaxis[i]))
                CDF = np.trapz(KDE,self.Yaxis[i])
                Dist[i] = [[self.Yaxis[i]], [KDE], [CDF]]
        self.Dist = Dist
    
    def SampleKDE(self,nbSample=1000,dim=None):
        if self.Dist[0] is None:
            raise Exception('Sample first the distribution!')
        if dim is None:
            dim = range(self.nb_dim)
        elif dim > self.nb_dim:
            raise Exception('Dimension outside of possibilities: max = {} (input = {})'.format(self.nb_dim,dim))
        distUnif = stats.uniform()
        Samples = np.asarray(distUnif.rvs(nbSample*len(dim))).reshape(nbSample,len(dim))
        for i in range(len(dim)):
            InvCDFs = InterpolatedUnivariateSpline(self.Dist[dim[i]][2],self.Dist[dim[i]][0],bbox=[0, 1])
            Samples[:,i] = InvCDFs(Samples[:,i])
        return Samples

        
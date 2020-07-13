# This module performs kernel density estimation for a given dataset.

import numpy as np 
import math as mt
from scipy import stats
from scipy.interpolate import interp1d
from matplotlib import path
from matplotlib import pyplot
from itertools import groupby

class KDE:
    def __init__(self,X,Y):
        tmp = X.shape
        self.nb_dim = tmp[1] # Getting the number of dimensions
        self.datasets = [None]*self.nb_dim
        self.Xaxis = [None]*self.nb_dim
        self.Yaxis = [None]*self.nb_dim
        self.KDE = [None]*self.nb_dim
        self.Dist = [None]*self.nb_dim
        for i in range(self.nb_dim):
            self.datasets[i] = np.column_stack((X[:,i],Y[:,i]))
    
    def KernelDensity(self,dim=None,XTrue=None,NoiseError=None):
        if dim is None:
            dim = range(self.nb_dim) # We run all the dimensions
        elif np.max(dim) > self.nb_dim:
            raise Exception('Dimension outside of possibilities: max = {} (input = {})'.format(self.nb_dim,np.max(dim)))
        if XTrue is not None and len(XTrue) != self.nb_dim:
            raise Exception('XTrue is not compatible with the dimensions given.')
        if XTrue is None and NoiseError is not None:
            NoiseError is None
        for i in dim:
            # Initializing
            dataset = self.datasets[i]
            L = dataset.shape
            L = L[0] # Only keeping the length of the dataset

            # Defining the bands to test and choose the optimal one
            bandPossible = np.logspace(-5,0,100)
            meanD, meanH = np.mean(dataset,axis=0)
            nbTh = 50
            th = np.linspace(0,2*np.pi,nbTh)
            for j in bandPossible:
                #circleOK = np.zeros((nbTh,2))
                #for k in range(nbTh):
                circleOK = np.column_stack(((3*j*np.cos(th) + meanD), (3*j*np.sin(th) + meanH)))# [j*3*mt.cos(th[k]) + meanD, j*3*mt.sin(th[k]) + meanH]
                p = path.Path(circleOK)
                inside = p.contains_points(dataset)
                if np.sum(inside) > np.max([L*0.05, 100]):# 5% seems to work fine
                    break
            band = j
            # Defing some lambda functions
            # 2D gaussian pdf:
            # TODO: Vectorize this computation:
            # z = lambda x,y,X,Y,b: ((x-X)**2)/(b**2) + ((y-Y)**2)/(b**2) - (2*(x-X)*(y-Y)/(b*b))
            # pdf = lambda x,y,X,Y,b: (1/(2*np.pi*b*b))*np.exp(-z(x,y,X,Y,b)/2)
            # Circle of points search: 
            # circle = lambda X,Y,b: np.column_stack(((4*b*np.cos(th) + X), (4*b*np.sin(th) + Y)))
            if XTrue is None:
                self.Xaxis[i] = np.arange((np.min(dataset[:,0])-4*band),(np.max(dataset[:,0])+4*band),band/2)
                self.Yaxis[i] = np.arange((np.min(dataset[:,1])-4*band),(np.max(dataset[:,1])+4*band),band/2)
                KDE = np.zeros((len(self.Xaxis[i]),len(self.Yaxis[i])))
                x_idx = 0
                for x in self.Xaxis[i]:
                    y_idx = 0
                    for y in self.Yaxis[i]:
                        # p = path.Path(circle(x,y,band))
                        probaLarge = 4
                        impacts = np.logical_and(np.logical_and(np.greater(dataset[:,0],x-probaLarge*band), np.less(dataset[:,0],x+probaLarge*band)), np.logical_and( np.greater(dataset[:,1],y-probaLarge*band), np.less(dataset[:,1],y+probaLarge*band)))#p.contains_points(dataset)
                        if np.sum(impacts)>0:
                            idxImpacts = np.where(impacts)
                            idxImpacts = idxImpacts[0]
                            # Test for vectorization:
                            z = np.divide(np.power(np.subtract(x,dataset[idxImpacts,0]),2), band**2) + np.divide(np.power(np.subtract(y,dataset[idxImpacts,1]),2), band**2) - np.divide(np.multiply(np.multiply(np.subtract(x,dataset[idxImpacts,0]), np.subtract(y,dataset[idxImpacts,1])), 2), band*band)
                            pdf = np.multiply(1/(2*np.pi*band*band), np.exp(-z))
                            KDE[x_idx,y_idx] += np.sum(pdf)
                            # End- test for vectorization
                            # for j in np.arange(len(idxImpacts)):
                            #     KDE[x_idx,y_idx] += pdf(x,y,dataset[idxImpacts[j],0],dataset[idxImpacts[j],1],band)
                            y_idx += 1
                        else:
                            y_idx += 1                    
                    x_idx += 1
                self.KDE[i] = KDE #np.divide(KDE,(np.sum(KDE)*band**2))
            else:
                self.Xaxis[i] = np.asarray(XTrue[i])
                self.Yaxis[i] = np.arange((np.min(dataset[:,1])-4*band),(np.max(dataset[:,1])+4*band),band/2)
                KDE = np.zeros((1,len(self.Yaxis[i])))
                bandY = band
                if (NoiseError is not None) and NoiseError[i]> band:
                    bandX = NoiseError[i]
                else:
                    bandX = band
                x_idx = 0
                x = self.Xaxis[i]
                y_idx = 0
                for y in self.Yaxis[i]:
                    # p = path.Path(circle(x,y,band))
                    probaLarge = 4
                    impacts = np.logical_and(np.logical_and(np.greater(dataset[:,0],x-probaLarge*band), np.less(dataset[:,0],x+probaLarge*band)), np.logical_and( np.greater(dataset[:,1],y-probaLarge*band), np.less(dataset[:,1],y+probaLarge*band)))#p.contains_points(dataset)
                    if np.sum(impacts)>0:
                        idxImpacts = np.where(impacts)
                        idxImpacts = idxImpacts[0]
                        # Test for vectorization:
                        z = np.divide(np.power(np.subtract(x,dataset[idxImpacts,0]),2), bandX**2) + np.divide(np.power(np.subtract(y,dataset[idxImpacts,1]),2), bandY**2) - np.divide(np.multiply(np.multiply(np.subtract(x,dataset[idxImpacts,0]), np.subtract(y,dataset[idxImpacts,1])), 2), bandX*bandY)
                        pdf = np.multiply(1/(2*np.pi*bandX*bandY), np.exp(-z))
                        KDE[x_idx,y_idx] += np.sum(pdf)
                        # End- test for vectorization
                        # for j in np.arange(len(idxImpacts)):
                        #     KDE[x_idx,y_idx] += pdf(x,y,dataset[idxImpacts[j],0],dataset[idxImpacts[j],1],band)
                        y_idx += 1
                    else:
                        y_idx += 1                    
                self.KDE[i] = KDE #np.divide(KDE,(np.sum(KDE)*band**2))
                KDE = np.divide(KDE,np.trapz(KDE,self.Yaxis[i]))
                CDF = np.cumsum(np.divide(KDE,np.sum(KDE)))
                Dist = [[self.Yaxis[i]], [KDE], [CDF]]
                self.Dist[i] = Dist
    
    def ShowKDE(self,dim=None,Xvals=None):
        if dim is None:
            dim = range(self.nb_dim)
        elif np.max(dim) > self.nb_dim:
            raise Exception('Dimension outside of possibilities: max = {} (input = {})'.format(self.nb_dim,np.max(dim)))
        if (Xvals is not None):
            Xvals = np.squeeze(Xvals)
            if len(Xvals) < self.nb_dim:
                raise Exception('Xvals is not compatible with the current datasets')
        for i in dim:
            # Printing the KDE on a graph
            if (self.KDE[i] is not None):
                X_KDE, Y_KDE = np.meshgrid(self.Xaxis[i],self.Yaxis[i])
                _, ax = pyplot.subplots()
                ax.pcolormesh(X_KDE,Y_KDE,np.transpose(self.KDE[i]))
                ax.set_title('Dimension {}'.format(str(i+1)))
                ax.set_xlabel('$D^c_{}$'.format(str(i+1)))
                ax.set_ylabel('$M^c_{}$'.format(str(i+1)))
                if (Xvals is not None):
                    ax.plot([Xvals[i],Xvals[i]],np.asarray(ax.get_ylim()),'r')
                    if (self.Dist[i] is not None):
                        # Add graph on the left with KDE distribution for Xvals
                        pyplot.subplots_adjust(left=0.4)
                        ax_hist = pyplot.axes([0.1, 0.1, 0.15, 0.8])
                        ax_hist.plot(np.squeeze(self.Dist[i][1]),np.squeeze(self.Dist[i][0]),'k')
                        ax_hist.set_xlabel('P (/)')
                        ax_hist.set_ylabel('$M^c_{}$'.format(str(i+1)))
                pyplot.show(block=False)
            else:
                raise Exception('No KDE field at dimension {}'.format(i))
        pyplot.show()

    def GetDist(self,Xvals=[0],dim=None,Noise=None):
        # Adds a list Dist to self:
        #   - Dist[0]=YVect
        #   - Dist[1]=KDE (normalized)
        #   - Dist[2]=CDF (corresponding to KDE, for sampler)
        # Handling exceptions:
        Xvals = Xvals[0]
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
                    CDF = np.cumsum(np.divide(KDE,np.sum(KDE)))
                    Dist[i] = [[self.Yaxis[i]], [KDE], [CDF]]
                else:
                    KDE = self.KDE[i][idx-1,:] + ((self.KDE[i][idx,:]-self.KDE[i][idx-1,:])/(self.Xaxis[i][idx]-self.Xaxis[i][idx-1]))*(Xvals[i]-self.Xaxis[i][idx-1])
                    KDE = np.divide(KDE,np.trapz(KDE,self.Yaxis[i]))
                    CDF = np.cumsum(np.divide(KDE,np.sum(KDE)))
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
                CDF = np.cumsum(np.divide(KDE,np.sum(KDE)))
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
            x = np.asarray(self.Dist[dim[i]][2]) # CDF
            difCDF = np.append((np.diff(x)>0),[False])
            x = x[:,difCDF].squeeze()
            y = np.asarray(self.Dist[dim[i]][0]) # cdf axis
            y = y[:,difCDF].squeeze()
            # pyplot.plot(x,y,'ro')
            # pyplot.show(block=False)
            InvCDFs = interp1d(x,y,bounds_error=False,fill_value=(np.min(y),np.max(y)))
            Samples[:,i] = InvCDFs(Samples[:,i])
        return Samples

        
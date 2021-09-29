# This module performs kernel density estimation for a given dataset.

import numpy as np 
import math as mt
from scipy import stats
from scipy.interpolate import interp1d
from scipy.special import erfcinv
from matplotlib import path
from matplotlib import pyplot
from pathos import multiprocessing as mp 
from pathos import pools as pp
from functools import partial

def ParallelKernel(inputs):
    '''Computation of the kernel density estimation simplified for parallel computation.
    Do not use standalone.
    '''
    dataset = inputs[0]
    XTrue = inputs[1]
    NoiseError = inputs[2]
    # Initializing
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
        Xaxis = np.arange((np.min(dataset[:,0])-4*band),(np.max(dataset[:,0])+4*band),band/2)
        Yaxis = np.arange((np.min(dataset[:,1])-4*band),(np.max(dataset[:,1])+4*band),band/2)
        KDE = np.zeros((len(Xaxis),len(Yaxis)))
        x_idx = 0
        for x in Xaxis:
            y_idx = 0
            for y in Yaxis:
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
        output = [KDE, Xaxis, Yaxis]
        return output
    else:
        Xaxis = np.asarray(XTrue)
        Yaxis = np.arange((np.min(dataset[:,1])-4*band),(np.max(dataset[:,1])+4*band),band/2)
        KDE = np.zeros((1,len(Yaxis)))
        bandY = band
        if (NoiseError is not None) and NoiseError> band:
            bandX = NoiseError
        else:
            bandX = band
        x_idx = 0
        x = Xaxis
        y_idx = 0
        for y in Yaxis:
            # p = path.Path(circle(x,y,band))
            probaLarge = 4
            impacts = np.logical_and(np.logical_and(np.greater(dataset[:,0],x-probaLarge*bandX), np.less(dataset[:,0],x+probaLarge*bandX)), np.logical_and(np.greater(dataset[:,1],y-probaLarge*band), np.less(dataset[:,1],y+probaLarge*band)))#p.contains_points(dataset)
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
        KDEinit = KDE #np.divide(KDE,(np.sum(KDE)*band**2))
        KDE = np.divide(KDE,np.trapz(KDE,Yaxis))
        CDF = np.cumsum(np.divide(KDE,np.sum(KDE)))
        Dist = [[Yaxis], [KDE], [CDF]]
        output = [KDEinit, Xaxis, Yaxis, Dist]
        return output



class KDE:
    '''KDE is a class object that contains all the informations about the kernel density estimation.
    To initialize, it requieres 2 arguments:
        - X (np.array): the X values of the datasets
        - Y (np.array): the Y values of the datasets
    The data contained in the class are:
        - nb_dim (int): the number of dimensions that exist
        - datasets (list): a list of [X Y] datasets of lenght nb_dim
        - Xaxis (list): a list of np.array containing the X-axis along which KDE is computed
        - Yaxis (list): a list of np.array containing the Y-axis along which KDE is computed
        - KDE (list): a list of np.array containing the computed kernel density estimations
        - Dist (list): a list of elements describing the distributions for a given X value.
    
    The class has several methods:
        - KernelDensity: Computed KDE
        - ShowKDE: Show graphs with the KDE approximated CCA space
        - GetDist: Retreive the Distribution from KDE for a given X value
        - SampleKDE: Sample models from the distribution and return them
    '''
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
    
    def KernelDensity(self,dim=None,XTrue=None,NoiseError=None,RemoveOutlier=False, Parallelization=[False, None],verbose:bool=False):
        '''KERNELDENSITY computes the kernel density estimation for a given dataset.
        
        It can run without arguments (default) but several arguments are optional:
            - dim (list): list of the dimensions to compute (default=None, meaning all)
            - XTrue (list): list of true values for X (reduce the computation time 
                            but decrease reusability) (default=None)
            - NoiseError (list): list of parameters describing the error model (see
                                 dedicated function) (default=None)
            - RemoveOutlier (bool): remove the outliers (True) or not (False). An 
                                    outlier is defined as a value greater than 3 
                                    times the scaled median absolute value away from 
                                    the median. (default=False)
            - Parallelization (list): parallelization instructions
                    o [False, _]: no parallel runs (default)
                    o [True, None]: parallel runs without pool provided
                    o [True, pool]: parallel runs with pool (defined bypathos.pools) 
                                    provided 
        
        If XTrue is provided, the distribution will be computed only for this particular
        value. Otherwise, the full space (X,Y) will be computed.
        '''
        if dim is None:
            dim = range(self.nb_dim) # We run all the dimensions
        elif np.max(dim) > self.nb_dim:
            raise Exception('Dimension outside of possibilities: max = {} (input = {})'.format(self.nb_dim,np.max(dim)))
        if XTrue is not None and len(XTrue) != self.nb_dim:
            raise Exception('XTrue is not compatible with the dimensions given.')
        if XTrue is None and NoiseError is not None:
            NoiseError is None
        # Throwing error if the point is too close to the limits of the domain for one dimension (would not work)
        if XTrue is not None:
            percentileLimit = 0.01
            if NoiseError is None:
                for i in range(len(XTrue)):
                    dataset = self.datasets[i]
                    if XTrue[i] < np.quantile(dataset[:,0],percentileLimit) or XTrue[i] > np.quantile(dataset[:,0],1-percentileLimit):
                        raise Exception('The dataset is outside of the distribution in the reduced space for dimension {}'.format(i+1))
            else:
                for i in range(len(XTrue)):
                    dataset = self.datasets[i]
                    if XTrue[i]+2*NoiseError[i] < np.quantile(dataset[:,0],percentileLimit) or XTrue[i]-2*NoiseError[i] > np.quantile(dataset[:,0],1-percentileLimit):
                        raise Exception('The dataset is outside of the distribution in the reduced space for dimension {}. Even with noise taken into account!'.format(i+1))
        if Parallelization[0]:
            FuncPara = partial(ParallelKernel)
            # Parallel computing:
            if Parallelization[1] is not None:
                pool = Parallelization[1]
                # pool.restart()
            else:
                pool = pp.ProcessPool(np.min([len(dim), mp.cpu_count()]))# Create the parallel pool with at most the number of dimensions
            inKernel = []
            for i in dim:
                if (XTrue is not None) and (NoiseError is not None):
                    inKernel.append([self.datasets[i], XTrue[i], NoiseError[i]])
                elif XTrue is not None:
                    inKernel.append([self.datasets[i], XTrue[i], None])
                elif NoiseError is not None:
                    inKernel.append([self.datasets[i], [None], NoiseError[i]])
                else:
                    inKernel.append([self.datasets[i], None, None])
            outputs = pool.map(FuncPara,inKernel)
            # pool.close()
            # pool.join()
            if verbose:
                print('Parallel passed!')
            idx = 0
            for i in dim:
                self.KDE[i] = outputs[idx][0]
                self.Xaxis[i] = outputs[idx][1]
                self.Yaxis[i] = outputs[idx][2]
                if XTrue is not None:
                    self.Dist[i] = outputs[idx][3]
                idx += 1
        else:
            for i in dim:
                # Initializing
                dataset = self.datasets[i]
                L = dataset.shape
                L = L[0] # Only keeping the length of the dataset

                # Remove outliers:
                if RemoveOutlier: # Modification for graph only:
                    c = -1/(mt.sqrt(2)*erfcinv(3/2))
                    isoutlierX = np.greater(np.abs(dataset[:,0]),3*c*np.median(np.abs(dataset[:,0]-np.median(dataset[:,0]))))
                    isoutlierY = np.greater(np.abs(dataset[:,1]),3*c*np.median(np.abs(dataset[:,1]-np.median(dataset[:,1]))))
                    isoutlier = np.logical_or(isoutlierX,isoutlierY)
                    if any(isoutlier):
                        dataset = np.delete(dataset,np.where(isoutlier),0)
                        LNew = dataset.shape
                        LNew = LNew[0] # Only keeping the length of the dataset
                        print('{} outlier removed!'.format(L-LNew))
                        L = LNew

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
                        impacts = np.logical_and(np.logical_and(np.greater(dataset[:,0],x-probaLarge*bandX), np.less(dataset[:,0],x+probaLarge*bandX)), np.logical_and( np.greater(dataset[:,1],y-probaLarge*band), np.less(dataset[:,1],y+probaLarge*band)))#p.contains_points(dataset)
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
        '''SHOWKDE is a method that displays the KDE approximated CCA space.
        By default, it shows 1 graph per dimension. Optional arguments are:
            - dim (list): list of the specific dimensions to display
            - Xvals (list): list of the true X values. Will be displayed in red.
        
        If the distributions are already computed for a given value of X, they 
        will also be displayed.
        '''
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
                        ax_hist.set_ylim(ax.get_ylim())
                pyplot.show(block=False)
            else:
                raise Exception('No KDE field at dimension {}'.format(i))
        pyplot.show(block=False)

    def GetDist(self,Xvals=[0],dim=None,Noise=None,verbose:bool=False):
        '''GETDIST is a method that extracts the distribution for a given X value from the KDE.

        It takes as argument:
            - Xvals (np.array): the true X values where to extract
            - dim (list):  list of the specific dimensions to compute (optional, default is all)
            - Noise (list): list of the impact of noise for the different dimensions (optional, 
                            default is None)
        
        It produces the element Dist in the class with, for each dimension,:
            - Dist[0]=YVect
            - Dist[1]=KDE (normalized)
            - Dist[2]=CDF (corresponding to KDE, for sampler)
        '''
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
                    raise Exception('The dataset seems to be outside the training set (dim {}), even with noise!'.format(i))
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
            if verbose:
                print('Noise:',Noise)
            for i in dim:
                KDE_tmp = np.zeros_like(self.KDE[i][1,:])
                samples = 100
                distNoise = stats.norm(loc=Xvals[i],scale=Noise[i])
                r = distNoise.rvs(size=samples)
                for j in range(samples):
                    idx = np.searchsorted(self.Xaxis[i],r[j],side='left')# Find the first index that is greater than the value
                    if idx >= len(self.Xaxis[i]):
                        pass
                    elif r[j]==self.Xaxis[i][idx]:
                        # Exact value:
                        KDE_tmp += self.KDE[i][idx,:]
                    else:
                        KDE_tmp += self.KDE[i][idx-1,:] + ((self.KDE[i][idx,:]-self.KDE[i][idx-1,:])/(self.Xaxis[i][idx]-self.Xaxis[i][idx-1]))*(r[j]-self.Xaxis[i][idx-1])
                KDE = np.divide(KDE_tmp,samples)
                KDE = np.divide(KDE,np.trapz(KDE,self.Yaxis[i]))
                CDF = np.cumsum(np.divide(KDE,np.sum(KDE)))
                Dist[i] = [[self.Yaxis[i]], [KDE], [CDF]]
        self.Dist = Dist
    
    def SampleKDE(self,nbSample:int=1000,dim=None):
        '''SAMPLEKDE is a method that samples values from the KDE. 
        
        It takes as arguments:
            - nbSamples (int): the number of samples to return (default=1000)
            - dim (list): the dimensions to sample (defalut is all)

        It returns a np.array with samples extracted from the distributions through 
        the inverse sampler method (uniform distribution to any distribution).
        '''
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

        
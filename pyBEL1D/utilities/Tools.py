# class cond:
#     def __init__(self):
#         self.function = None

def isalambda(v):# From: https://stackoverflow.com/questions/3655842/how-can-i-test-whether-a-variable-holds-a-lambda
  LAMBDA = lambda:0
  return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

def Sampling(prior:list,conditions=None,nbModels:int=1000):
    '''SAMPLING is a function that samples models from a gievn prior model space.

    It takes as arguments:
        - prior (list): a list of scipy stats distributions describing the prior
                        model space
        - condtions (optional - callable lambda function): a function that returns
                o True if the sampled model respects the conditions
                o False otherwise
        - nbModels (int): the number of models to sample (default=1000)

    It returns the sampled models in a numpy array
    '''
    import numpy as np
    # Checking that prior is a list:
    if not(type(prior) is list):
        raise Exception('prior should be a List of distributions. prior is a {}'.format(type(prior)))
    # Initializing variables
    nbParam = len(prior)
    Models = np.zeros([nbModels,nbParam])
    # Sampling the models:
    if conditions is None:
        for i in range(nbParam):
            Models[:,i]=prior[i].rvs(size=nbModels)
    else:
        # Checking that the conditions are in a list
        if not(isalambda(conditions)):
            raise Exception('conditions should be a lambda. conditions is a {}'.format(type(conditions)))
        # Sampling the models:
        achieved = False
        modelsOK = 0
        while not(achieved):
            for i in range(nbParam):
                Models[modelsOK:,i]=prior[i].rvs(size=(nbModels-modelsOK))
            keep = np.ones((nbModels,))
            for i in range(nbModels-modelsOK):
                keep[modelsOK+i] = conditions(Models[modelsOK+i,:])
            indexKeep = np.where(keep)
            modelsOK = np.shape(indexKeep)[1]
            tmp = np.zeros([nbModels,nbParam])
            tmp[range(modelsOK),:] = np.squeeze(Models[indexKeep,:])
            Models = tmp
            if modelsOK == nbModels:
                achieved = True
    # Return the sampled models
    return Models

def PropagateNoise(POSTBEL,NoiseLevel=None, DatasetIn=None):
    '''PROPAGATENOISE is a function that computes the impact of noise on the PCA scores
    for a given problem (POSTBEL) and a given noise level (NoiseLevel).

    The arguments are:
        - POSTBEL (POSTBEL): a POSTBEL class object that has been initialized.
        - NoiseLevel (list): a list of parameters for the noise model (depending on TypeMod)
            o TypeMod=="sNMR": list with 1 value for the standard deviation
            o TypeMod=="DC": list with a given standard deviation for all the values 
                             of the dataset.
            o TypeMod=="General": list with a given standard deviation for all the values 
                                  of the dataset.
    
    It returns NoiseLevel, a np.array containing the noise propagated in the CCA space.
    '''
    import numpy as np
    TypeMod = POSTBEL.MODPARAM.method
    
    implemented_noise_models = ['sNMR', 'DC']
    if not TypeMod in implemented_noise_models:
        TypeMod = 'General'
        print('Warning - no noise propagation for this method defined...')
        print('falling back to general case...')
    
    dim = POSTBEL.CCA.x_scores_.shape[1] # Number of dimensions for noise propagation
    dimD = POSTBEL.PCA["Data"].n_components_
    # NoiseLevel = [0]*dim
    if TypeMod == "sNMR": # Modeling Gaussian NoiseLevel (noise is an int)
        if not(isinstance(NoiseLevel,int)):
            print('NoiseLevel must be an integer for sNMR NoiseLevel propagation! Converted to default value of 10 nV!')
            NoiseLevel = 10 # in nV
        NoiseLevel *= 1e-9 # Convert to Volts
        # Propagating NoiseLevel:
        nbTest = int(np.ceil(POSTBEL.nbModels/10)) 
        COV_diff = np.zeros((nbTest,dimD,dimD))
        index = np.random.permutation(np.arange(POSTBEL.nbModels))
        index = index[:nbTest] # Selecting a set of random models to compute the noise propagation
        data = POSTBEL.FORWARD[index,:] 
        dataNoisy = data + NoiseLevel*np.random.randn(nbTest,POSTBEL.FORWARD.shape[1])
        scoreData = POSTBEL.PCA['Data'].transform(data)
        scoreDataNoisy = POSTBEL.PCA['Data'].transform(dataNoisy)
        err_f = scoreData-scoreDataNoisy
        Cf = np.cov(err_f.T)
        # for i in range(nbTest):
        #    COV_diff[i,:,:] = np.cov(np.transpose(np.squeeze([[scoreData[i,:]],[scoreDataNoisy[i,:]]])))
        # Cf = np.squeeze(np.mean(COV_diff,axis=0))
        Cc = np.dot(POSTBEL.CCA.x_loadings_.T,np.dot(Cf,POSTBEL.CCA.x_loadings_))#POSTBEL.CCA.x_loadings_.T*Cf*POSTBEL.CCA.x_loadings_
        NoiseLevel = np.diag(Cc)
    elif TypeMod == "DC":
        # if not(isinstance(NoiseLevel,list)):
        #     print('NoiseLevel must be in the form of a list! Converted to default value!')
        #     NoiseLevel = [0.05, 100]
        # Propagating NoiseLevel:
        # PCA_Propag = True
        # if PCA_Propag:
        if not(isinstance(NoiseLevel,list)) and not(isinstance(NoiseLevel,np.ndarray)):
            raise Exception('NoiseLevel is not a list or a numpy array!')
        if len(NoiseLevel)!=POSTBEL.FORWARD.shape[1]:
            raise Exception('Wrong length for NoiseLevel list.')
        nbTest = int(np.ceil(POSTBEL.nbModels/10)) #10
        COV_diff = np.zeros((nbTest,dimD,dimD))
        index = np.random.permutation(np.arange(POSTBEL.nbModels))
        index = index[:nbTest] # Selecting a set of random models to compute the noise propagation
        data = POSTBEL.FORWARD[index,:] 
        dataNoisy = data + np.multiply(np.repeat(np.random.randn(nbTest,1),data.shape[1],axis=1),np.repeat(np.reshape(NoiseLevel,(1,np.shape(data)[1])),data.shape[0],axis=0)) # np.divide((NoiseLevel[0]*data*1000 + np.divide(1,POSTBEL.MODPARAM.forwardFun["Axis"])/NoiseLevel[1]),1000)# The error model is in Frequency, not periods
        #dataNoisy = data + np.multiply(np.random.randn(nbTest,data.shape[1]),np.repeat(np.reshape(NoiseLevel,(1,np.shape(data)[1])),data.shape[0],axis=0)) # np.divide((NoiseLevel[0]*data*1000 + np.divide(1,POSTBEL.MODPARAM.forwardFun["Axis"])/NoiseLevel[1]),1000)# The error model is in Frequency, not periods
        scoreData = POSTBEL.PCA['Data'].transform(data)
        scoreDataNoisy = POSTBEL.PCA['Data'].transform(dataNoisy)
        err_f = scoreData-scoreDataNoisy
        Cf = np.cov(err_f.T)
        # for i in range(nbTest):
        #     COV_diff[i,:,:] = np.cov(np.transpose(np.squeeze([[scoreData[i,:]],[scoreDataNoisy[i,:]]])))
        # Cf = np.squeeze(np.mean(COV_diff,axis=0))
        Cc = np.dot(POSTBEL.CCA.x_loadings_.T,np.dot(Cf,POSTBEL.CCA.x_loadings_))
        NoiseLevel = np.diag(Cc)
        # else:
        #     try:
        #         Dataset = POSTBEL.DATA['True']
        #     except:
        #         Dataset = DatasetIn
        #         if Dataset is None:
        #             raise Exception('The Dataset is not given for the current computation!')
        #     nbTest = 1000#int(np.ceil(POSTBEL.nbModels/10))
        #     nbDim = POSTBEL.CCA.x_scores_.shape[1]
        #     #CCA_Data = np.zeros((nbTest,nbDim))
        #     Dataset_Noisy = np.repeat(Dataset, nbTest, axis=0) + np.multiply(np.repeat(np.random.randn(nbTest,1),Dataset.shape[1],axis=1),np.repeat(np.reshape(NoiseLevel,(1,np.shape(Dataset)[1])),Dataset.shape[0],axis=0))
        #     d_obs_h = POSTBEL.PCA['Data'].transform(Dataset_Noisy)
        #     CCA_Data = POSTBEL.CCA.transform(d_obs_h)
        #     NoiseLevel = np.var(CCA_Data,axis=0)
    elif TypeMod == "General":
        if not(isinstance(NoiseLevel,list)) and not(isinstance(NoiseLevel,np.ndarray)):
            raise Exception('NoiseLevel is not a list!')
        if len(NoiseLevel)!=POSTBEL.FORWARD.shape[1]:
            raise Exception('Wrong length for NoiseLevel list.')
        nbTest = int(np.ceil(POSTBEL.nbModels/10)) 
        COV_diff = np.zeros((nbTest,dimD,dimD))
        index = np.random.permutation(np.arange(POSTBEL.nbModels))
        index = index[:nbTest] # Selecting a set of random models to compute the noise propagation
        data = POSTBEL.FORWARD[index,:] 
        dataNoisy = data + np.multiply(np.repeat(np.random.randn(nbTest,1),data.shape[1],axis=1),np.repeat(np.reshape(NoiseLevel,(1,np.shape(data)[1])),data.shape[0],axis=0)) # np.divide((NoiseLevel[0]*data*1000 + np.divide(1,POSTBEL.MODPARAM.forwardFun["Axis"])/NoiseLevel[1]),1000)# The error model is in Frequency, not periods
        scoreData = POSTBEL.PCA['Data'].transform(data)
        scoreDataNoisy = POSTBEL.PCA['Data'].transform(dataNoisy)
        err_f = scoreData-scoreDataNoisy
        Cf = np.cov(err_f.T)
        # for i in range(nbTest):
        #    COV_diff[i,:,:] = np.cov(np.transpose(np.squeeze([[scoreData[i,:]],[scoreDataNoisy[i,:]]])))
        # Cf = np.squeeze(np.mean(COV_diff,axis=0))
        Cc = np.dot(POSTBEL.CCA.x_loadings_.T,np.dot(Cf,POSTBEL.CCA.x_loadings_))#POSTBEL.CCA.x_loadings_.T*Cf*POSTBEL.CCA.x_loadings_
        NoiseLevel = np.diag(Cc)
    else:
        raise RuntimeWarning('No noise propagation defined for the given method!')
    return NoiseLevel

nSamplesConverge = 1000
def ConvergeTest(SamplesA, SamplesB, tol=5e-3):
    ''' CONVERGETEST is a function that returns the mean Wasserstein distance between 
    two sets of N-dimensional datapoints.
    
    It takes as arguments:
        - SamplesA (np.array): the base samples
        - SamplesB (np.array): the new samples
        - tol (float): the tolerance on the distance (default=5e-3)
    
    It returns:
        - diverge (bool): True if the samples diverge, False otherwise
        - distance (float): the mean Wasserstein distance

    '''
    import numpy as np 
    from scipy import stats
    from sklearn.preprocessing import normalize
    nbDim = np.size(SamplesA,axis=1)
    if np.size(SamplesB,axis=1) != nbDim:
        raise Exception('SamplesA and SamplesB must have the same number of features!')
    # SamplesA is the base and SamplesB is compared to it!
    # Subsampling 1000 models to compare each distributions (gain of time)
    if len(SamplesA) > nSamplesConverge and len(SamplesB) > nSamplesConverge:
        idxKeepA = np.random.choice(np.arange(len(SamplesA)), nSamplesConverge, replace=False)
        SamplesA = SamplesA[idxKeepA,:]
        idxKeepB = np.random.choice(np.arange(len(SamplesB)), nSamplesConverge, replace=False)
        SamplesB = SamplesB[idxKeepB,:]
    SamplesANorm, norms = normalize(SamplesA,axis=0,return_norm=True)
    SamplesBNorm = SamplesB/norms
    # SamplesANorm = SamplesA
    # SamplesBNorm = SamplesB
    distance = np.zeros(nbDim)
    # nbVal = 0
    Entropy = False
    Wasserstein = False
    KStest = True
    for i in range(nbDim):
        # Check if the 1D distributions are similar
        # nbVal += 1
        # Convert the samples into probabilities:
        if Entropy:
            minBins = np.min([np.min(SamplesANorm,axis=0),np.min(SamplesBNorm,axis=0)],axis=0)
            maxBins = np.max([np.max(SamplesANorm,axis=0),np.max(SamplesBNorm,axis=0)],axis=0)
            binsCompute = np.linspace(minBins[i],maxBins[i],num=int(len(SamplesANorm[:,i])/10))
            pdf1,_ = np.histogram(SamplesANorm[:,i],bins=binsCompute)
            pdf2,_ = np.histogram(SamplesBNorm[:,i],bins=binsCompute)
            while (pdf1==0).any() or (pdf2==0).any():
                idx0_1 = np.ravel(np.equal(pdf1,0).nonzero())
                idx0_2 = np.ravel(np.equal(pdf2,0).nonzero())
                if idx0_1.size: # There is (at least) a zero value in the first pdf
                    while idx0_1.size:
                        if idx0_1[0] == 0:
                            binsCompute = np.delete(binsCompute,1)
                            idx0_1 = np.delete(idx0_1,0)
                            idx0_1 = idx0_1 - 1
                        else:
                            binsCompute = np.delete(binsCompute,idx0_1)
                            idx0_1 = np.array([])
                    pdf1,_ = np.histogram(SamplesANorm[:,i],bins=binsCompute)
                    pdf2,_ = np.histogram(SamplesBNorm[:,i],bins=binsCompute)
                elif idx0_2.size: # There is (at least) a zero value in the first pdf
                    while idx0_2.size:
                        if idx0_2[0] == 0:
                            binsCompute = np.delete(binsCompute,1)
                            idx0_2 = np.delete(idx0_2,0)
                            idx0_2 = idx0_2 - 1
                        else:
                            binsCompute = np.delete(binsCompute,idx0_2)
                            idx0_2 = np.array([])
                    pdf1,_ = np.histogram(SamplesANorm[:,i],bins=binsCompute)
                    pdf2,_ = np.histogram(SamplesBNorm[:,i],bins=binsCompute)
            distance[i] = stats.entropy(pdf1,pdf2)
        if Wasserstein:
            distance[i] = stats.wasserstein_distance(SamplesANorm[:,i],SamplesBNorm[:,i]) # Return wasserstein distance between distributions --> "Small" = converged
        if KStest:
            D, pvalue =  stats.ks_2samp(SamplesANorm[:,i],SamplesBNorm[:,i])
            distance[i] = D
    
    # if not(KStest):
    distance = np.max(distance)
    if distance <= tol:
        diverge = False
    else:
        diverge = True
    # else:
    #     distance = np.min(distance)
    #     if distance >= tol:
    #         diverge = False
    #     else:
    #         diverge = True
        
    return diverge, distance
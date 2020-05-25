class cond:
    def __init__(self):
        self.function = None

def isalambda(v):# From: https://stackoverflow.com/questions/3655842/how-can-i-test-whether-a-variable-holds-a-lambda
  LAMBDA = lambda:0
  return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

def Sampling(prior,conditions=None,nbModels=1000):
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

def PropagateNoise(POSTBEL,NoiseLevel=None):
    import numpy as np 
    TypeMod = POSTBEL.MODPARAM.method
    dim = POSTBEL.CCA.x_scores_.shape[1] # Number of dimensions for noise propagation
    dimD = POSTBEL.PCA["Data"].n_components_
    Noise = [0]*dim
    if TypeMod is "sNMR": # Modeling Gaussian Noise (noise is an int)
        if not(isinstance(NoiseLevel,int)):
            print('Noise must be an integer for sNMR Noise propagation! Converted to default value of 10 nV!')
            NoiseLevel = 10 # in nV
        NoiseLevel *= 1e-9 # Convert to Volts
        # Propagating Noise:
        nbTest = int(np.ceil(POSTBEL.nbModels/10)) 
        COV_diff = np.zeros((nbTest,dimD,dimD))
        index = np.random.permutation(np.arange(POSTBEL.nbModels))
        index = index[:nbTest] # Selecting a set of random models to compute the noise propagation
        data = POSTBEL.FORWARD_PRIOR[index,:] 
        dataNoisy = data + NoiseLevel*np.random.randn(nbTest,POSTBEL.FORWARD_PRIOR.shape[1])
        scoreData = POSTBEL.PCA['Data'].transform(data)
        scoreDataNoisy = POSTBEL.PCA['Data'].transform(dataNoisy)
        for i in range(nbTest):
           COV_diff[i,:,:] = np.cov(np.transpose(np.squeeze([[scoreData[i,:]],[scoreDataNoisy[i,:]]])))
        Cf = np.squeeze(np.max(COV_diff,axis=0))
        Cc = POSTBEL.CCA.x_loadings_.T*Cf*POSTBEL.CCA.x_loadings_
        Noise = np.diag(Cc)
    elif TypeMod is "DC":
        if not(isinstance(NoiseLevel,list)):
            print('Noise must be in the form of a list! Converted to default value!')
            NoiseLevel = [0.05, 100]
        # Propagating Noise:
        nbTest = int(np.ceil(POSTBEL.nbModels/10)) 
        COV_diff = np.zeros((nbTest,dimD,dimD))
        index = np.random.permutation(np.arange(POSTBEL.nbModels))
        index = index[:nbTest] # Selecting a set of random models to compute the noise propagation
        data = POSTBEL.FORWARD_PRIOR[index,:] 
        dataNoisy = data + NoiseLevel[0]*(np.random.randn(nbTest,1)*data + POSTBEL.MODPARAM.forwardFun["Axis"]/NoiseLevel[1])
        scoreData = POSTBEL.PCA['Data'].transform(data)
        scoreDataNoisy = POSTBEL.PCA['Data'].transform(dataNoisy)
        for i in range(nbTest):
           COV_diff[i,:,:] = np.cov(np.transpose(np.squeeze([[scoreData[i,:]],[scoreDataNoisy[i,:]]])))
        Cf = np.squeeze(np.max(COV_diff,axis=0))
        Cc = POSTBEL.CCA.x_loadings_.T*Cf*POSTBEL.CCA.x_loadings_
        Noise = np.diag(Cc)
    else:
        raise RuntimeWarning('No noise propagation defined for the gievn method!')
    return Noise

def ConvergeTest(SamplesA, SamplesB,nbClassDim=10):
    import numpy as np 
    from scipy import stats
    # 0) Find the ranges of each axis:
    mins = np.min(np.append(SamplesA, SamplesB, axis=0),axis=0)
    maxs = np.max(np.append(SamplesA, SamplesB, axis=0),axis=0)
    nbDim = np.size(SamplesA,axis=1)
    if np.size(SamplesB,axis=1) != nbDim:
        raise Exception('SamplesA and SamplesB must have the same number of features!')
    # TODO: Manage memory! The use of memory for the operations is HUGE!
    #
    # Instead of comparing the whole distributions (impossible for memory management!), 
    # we will compare the distributions in 2D spaces and analyse all the combinations 
    # We thus have to analyse 'np.cumsum(np.arange(1,nbDim))' combinations with memory
    # needs closer to acheivable results (with 100 values per axis, we get per combination 
    # (100x100)xnp.unit64).
    # The complete analysis would requier (100**nbDim)xnp.uint64, impossible for large
    # dimension models.
    #
    # 1) Convert the samples array to Discrete probabilities in n-dimensions
    for i in range(nbDim):
        for j in range(nbDim):
            if i==j:
                # Check if the 1D distributions are similar
                distance = stats.wasserstein_distance(SamplesA[:,i],SamplesB[:,j]) # Return wasserstein distance between distributions --> "Small" = converged
            elif i < j:
                # Check if the 2D combinations are similar: DOI 10.1007/s10851-014-0506-3
                # Build classes of the first dimension to infer the second dimension histogram

                test = 0
    # 2) Compute the KL divernence:
    diverge = 0
    return diverge
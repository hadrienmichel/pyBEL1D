class cond:
    def __init__(self):
        self.function = None

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
        if not(type(conditions) is list):
            raise Exception('conditions should be a List of conditions. conditions is a {}'.format(type(conditions)))
        # Sampling the models:
        achieved = False
        modelsOK = 0
        while not(achieved):
            for i in range(nbParam):
                Models[modelsOK:-1,i]=prior[i].rvs(size=(nbModels-modelsOK))
            keep = np.ones([1,nbModels])
            for i in range(len(conditions)):
                keep[modelsOK:-1] = keep[modelsOK:-1] and conditions[i].compute(Models[modelsOK:-1,:])
            indexKeep = np.where(keep)
            modelsOK = len(indexKeep)
            tmp = np.zeros([nbModels,nbParam])
            tmp[range(modelsOK),:] = Models[indexKeep,:]
            Models = tmp
            if modelsOK == nbModels:
                achieved = True
    # Return the sampled models
    return Models

def PropagateNoise(POSTBEL,NoiseModel):
    Noise = 0
    return Noise
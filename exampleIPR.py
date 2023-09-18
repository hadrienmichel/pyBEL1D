# Import the usefull libraries:
from pyBEL1D import BEL1D
import numpy as np

# Create a benchmark sNMR model:
SyntheticBenchmarkSNMR = np.asarray([25, 25, 0.05, 0.25, 0.10, 0.1, 0.2, 0.05])

# Create a suitable prior model space for the the dataset:
priorSNMRBenchmark = np.asarray([[1.0, 50.0, 0.01, 0.50, 0.005, 0.500],
                                 [1.0, 50.0, 0.01, 0.50, 0.005, 0.500], 
                                 [1.0, 50.0, 0.01, 0.50, 0.005, 0.500]])

# Load the forward modelling matrix (from MRSMatlab)
KernelBenchmark = "Data/sNMR/MRS2021.mrsk"
TimingsBenchmark = np.arange(0.005, 0.5, 0.002)

# Create the modelset class object (names of variables, forward model, etc.)
ModelSet = BEL1D.MODELSET().SNMR(prior=priorSNMRBenchmark, 
    Kernel=KernelBenchmark, Timing= TimingsBenchmark)

# Create the benchmark dataset (with added noise)
Dataset = ModelSet.forwardFun["Fun"](SyntheticBenchmarkSNMR)
Noise = np.mean(Dataset)/5
NoisyDatasetBenchmark = Dataset + np.random.randn(len(Dataset))*Noise

# Run pyBEL1D:
#   - Prebel = Last training phase
#   - Postbel = Results from pyBEL1D
#   - PrebelInit = Initial training on the initail prior
Prebel, Postbel, PrebelInit = BEL1D.IPR(MODEL=ModelSet, 
    Dataset=NoisyDatasetBenchmark, NoiseEstimate=Noise*1e9)

# Show a graph with the resulting distributions (automated fucntions):
Postbel.ShowPostModels(TrueModel=SyntheticBenchmarkSNMR, RMSE=True)

BEL1D.pyplot.show()
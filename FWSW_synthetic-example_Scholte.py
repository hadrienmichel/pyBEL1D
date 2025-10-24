'''
FWSW: Full wavefield surface wave analysis
Here we show the application on a real dataset (download dataset from github).
'''
import matplotlib.pyplot as plt

if __name__=="__main__": # To prevent recomputation when in parallel
    #########################################################################################
    ###           Import the different libraries that are used in the script              ###
    #########################################################################################
    ## Common libraries:
    import numpy as np                              # For matrix-like operations and storage
    import os                                       # For files structures and read/write operations
    from os import listdir                          # To retreive elements from a folder
    from os.path import isfile, join                # Common files operations
    from matplotlib import pyplot                   # For graphics on post-processing
    import time                                     # For simple timing measurements


    ## Libraries for parallel computing:
    from pathos import multiprocessing as mp        # Multiprocessing utilities (get CPU cores info)
    from pathos import pools as pp                  # Building the pool to use for computations

    ## BEL1D requiered libraries:
    from scipy import stats                         # To build the prior model space
    from pyBEL1D import BEL1D                       # The main code for BEL1D
    from pyBEL1D.utilities import Tools             # For further post-processing
    from pyBEL1D.utilities.Tools import multiPngs   # For saving the figures as png

    ## Forward modelling code:
    # try:
    #     from pysurf96 import surf96                     # Code for the forward modelling of dispersion curves
    # except:
    #     pass
    #########################################################################################
    ###                    Flags for the different computation possible                   ###
    #########################################################################################
    '''
    For reproductibility of the results, we can fix the random seed.
    To fix the random seed, set RamdomSeed to False. Otherwise, the
    seed will be provided by the operating system.
    Note that the results exposed in the publication are performed
    under Windows 10 running python 3.7.6 (numpy=1.16.5, scikit-
    learn=0.23.1 and scipy=1.5.0).
    We observed that the random function does not necesseraly produce
    exactly the same results under other environments (and python 
    versions)!
    '''
    RandomSeed = False          # If True, use true random seed, else (False), fixed for reproductibility (seed=0)
    '''
    Some input parameters, to obtain some results or others.
    Eventhough the computations are relativelly fast, producing the 
    different graphs might be very cumbersome (matplotlib produces 
    nice figures, but is very slow).
    '''
    Graphs = True               # Obtain all the graphs?
    ParallelComputing = False   # Use parallel computing whenever possible?
    FWImage = True  # Projection of dispersion images
    verbose = True              # Output all the details about the current progress of the computations
    stats = True                # Parameter for the computation/retrun of statistics along with the iterations.
    #########################################################################################
    ###                            Initializing the parallel pool                          ###
    #########################################################################################
    if ParallelComputing:
        pool = pp.ProcessPool(mp.cpu_count()) # Create the parallel pool with at most the number of dimensions
        ppComp = [True, pool]
    else:
        ppComp = [False, None] # No parallel computing

    #########################################################################################
    ###                     FW image on synthetic and real data                           ###
    #########################################################################################
    if FWImage:

        ## PRIORS
        # uniform priors are structured as follows:
        # prior = np.array([[th1_min, th1_max, Vs1_min, Vs1_max, Vp1_min, Vp1_max], #rho1_min, rho1_max, Qalphas1_min, Qalphas2_max, Qbetas1_min, Qbetas1_max
        #                   [th2_min, th2_max, Vs2_min, ...],
        #                   [th3_min, ...],
        #                   ...
        #                   ])

        # MSEISFK example
        # prior = np.array([[0.01, 0.1, 0., 0.5, 1., 2., 0.5, 1.5],
        #                     [0.1, 0.25, 1.5, 2.5, 3.0, 4., 1., 2.],
        #                     [0.02, 0.1, 1., 1.5, 1., 2., 1.5, 2.5],
        #                     [0, 0, 1., 2., 2., 3., 2., 3.]])
        prior_Q = np.array([[0.01, 0.1, 0., 0.5, 1., 2., 0.5, 1.5, 2500, 3500, 0, 0],
                            [0.1, 0.25, 1.5, 2.5, 3.0, 4., 1., 2., 1500, 2500, 500, 1500],
                            [0.02, 0.1, 1., 1.5, 1., 2., 1.5, 2.5, 500, 1500, 350, 1150],
                            [0, 0, 1., 2., 2., 3., 2., 3., 500, 1500, 350, 1150]])


        ## MODELS
        ## FIELD MODELS

        # MSEISFK example
        Thickness = np.asarray([0.034, 0.2, 0.05])
        Vs = np.asarray([0, 2., 1.2, 1.4])
        Vp = np.asarray([1.43, 3.5, 1.3, 2.3])
        rho = np.asarray([1., 1.5, 2., 2.3])
        Qalphas = np.asarray([3000., 2000., 1000., 1000.])
        Qbetas = np.asarray([0., 1000., 800., 800.])
        ## Qbetas = np.asarray([0., 100, 100., 100.])
        nLayer = 4

        model = np.hstack((Thickness, Vs, Vp, rho, Qalphas, Qbetas))


        ## TEST RUNS
        ## ENVIRONMENT COMPOSTI FOR DC_FW_IMAGE

        from composti.src.sourcefunction import SourceFunctionGenerator
        from composti.src.utils import create_frequencyvector, create_timevector, convert_freq_to_time
        from composti.src import reflectivityCPP
        import scipy.interpolate as interp
        from swprocess import Masw, Sensor1C, Source, Array1D
        from swprocess.wavefieldtransforms import SlantStack, FK, FDBF, PhaseShift
        from matplotlib.pyplot import GridSpec
        from scipy.interpolate import RegularGridInterpolator

        path = YOURPATH


        # Create COMPOSTI options:
        options = np.zeros(4, dtype=np.int32)
        options[0] = 1  # Compute the direct wave
        options[1] = 1  # Compute multiple reflections
        options[2] = 0  # Use frequency windowing
        options[3] = 1  # Return type: 0 - displacement, 1 - velocity, 2 - acceleration


        # RUNS - SYNTHETIC OR REAL DATA


        #########
        ### IF SYNTHETIC DATA:
        # Min-max frequency for dispersion image plot
        fmin, fmax = 5, 100
        nf = 0.5
        # Selection of trial velocities (velocity in m/s) with minimum, maximum, number of steps, and space {"linear", "log"}.
        vmin, vmax, nvel, vspace = 1000, 3500, 1000, "linear"

        source_sw = Source(x=0, y=0, z=0)
        N = 120
        T = 0.0001
        Tacq = N * T
        dx = 10
        array = 990
        xReceivers = np.arange(10, array + dx, dx).astype(np.float64)  # position of traces (offset in x-direction)
        fMax = 100
        fMaxImage = 100
        vMax = 3500  # maximum frequency used to calculate dispersion image
        freq, dt = create_frequencyvector(T_end=Tacq, f_max_requested=fMax, f_min_requested=5)
        source_generator = SourceFunctionGenerator(freq)
        # # Ricker source wavelet of 20 Hz, multiple peak frequencies are also possible
        source = source_generator.Ricker(1, 20)
        fdbf_weighting = "sqrt"
        fdbf_steering = "cylindrical"
        settings = Masw.create_settings_dict(fmin=fmin, fmax=fmax, vmin=vmin, vmax=vmax, nvel=nvel, vspace=vspace,
                                             weighting=fdbf_weighting, steering=fdbf_steering)


        ##########

        # Underwater acquisition parameters for MSEISFK:
        # model to layers
        layers = BEL1D.model_to_layers(thicknesses=Thickness, vp=Vp, vs=Vs, rho=rho, qp=Qalphas, qs=Qbetas)
        print("layers:", layers)

        # underwater source and receiver depths
        source_depth = 0.01  # km
        # source_depth = BEL1D.calc_source_depth(model, nLayer)
        print("source_depth:", source_depth)
        # receiver_depth = 0.01  # km
        receiver_depth = BEL1D.calc_receiver_depth(model, nLayer)
        print("receiver_depth:", receiver_depth)

        q_artif = 0    # Q value for artificial wave attenuation (if <= 0, no artificial attenuation; optional parameter for blunting the normal mode spikes)

        # frequency range for mseisfk
        freq_range = (fmin, fmax, nf)
        print("freq_range:", freq_range)
        frequency = np.arange(fmin, fmax + nf, nf)

        # from velocity to slowness for mseisfk
        vel_range = np.linspace(vmin, vmax, nvel)
        slowness_range_kms = 1000 / vel_range
        slow_start = np.min(slowness_range_kms)
        slow_end = np.max(slowness_range_kms)
        slow_step = abs((slow_end - slow_start) / (nvel - 1))
        # slow_step = 0.002
        slow_range = (slow_start, slow_end, slow_step)
        print("slow_range:", slow_range)
        slowness = np.arange(slow_start, slow_end + slow_step, slow_step)

        msfk_input = BEL1D.write_msfk_input(source_depth=source_depth, receiver_depth=receiver_depth,
                                            freq_range=freq_range, slow_range=slow_range,
                                            q_artif=q_artif, layers=layers, filename='msfk08.inp')

        # Run the simulation, creating ms.fz (vertical/pressure component) and ms.fr (radial component)
        msfk_res = BEL1D.run_msfk(cwd=path)
        # Print execution result
        print(msfk_res.stdout)
        print(msfk_res.stderr)

        ############
        # IF UNDERWATER ACQUISITION:
        # Read and use the output
        fz = BEL1D.read_spectrum_file(path + 'ms.fz')
        fr = BEL1D.read_spectrum_file(path + 'ms.fr')

        # Normalize the data (cut off extra rows/cols if needed)
        fr_ = fr[1:, 1:]
        print("np.max(fr_):", np.max(fr_))
        fr_norm = fr_ / np.max(fr_)

        # Ensure dimensions match
        if fr_norm.shape != (len(slowness), len(frequency)):
            print("Shape mismatch — adjusting slowness and frequency to match data matrix.")
            slowness = slowness[:fr_norm.shape[0]]
            frequency = frequency[:fr_norm.shape[1]]

        # Create interpolator
        interp_func = RegularGridInterpolator(
            (slowness, frequency), fr_norm, bounds_error=False, fill_value=0.0)

        # New velocity grid
        velocity_new = np.linspace(vmin, vmax, nvel)  # m/s
        slowness_new = 1000 / velocity_new  # s/km
        print(f"Slowness domain: {slowness.min()} - {slowness.max()}")
        print(f"Slowness_new domain: {slowness_new.min()} - {slowness_new.max()}")

        # Meshgrid for evaluation
        S_new, F_new = np.meshgrid(slowness_new, frequency, indexing='ij')
        points = np.stack([S_new.ravel(), F_new.ravel()], axis=-1)

        # Interpolated data
        fr_interp = RegularGridInterpolator((slowness, frequency), fr_norm, bounds_error=False, fill_value=0.0)
        fr_velocity = interp_func(points).reshape(S_new.shape)
        # print(fr_velocity)

        Upf = fr_velocity

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.imshow(Upf, aspect='auto', origin='lower',
                  extent=(frequency[0], frequency[-1], velocity_new[0], velocity_new[-1]),
                  interpolation='bilinear', cmap='gist_rainbow')
        ax.set_xlabel('Frequency [Hz]', fontsize=14)
        ax.set_ylabel('Phase velocity [m/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.tight_layout()
        plt.show(block=True)

        dataSyn = Upf.flatten()
        np.savetxt('dataSyn_mseisfk.txt', dataSyn)

        #########


        dataModel, fSyn, vSyn = BEL1D.ForwardMSEISFK(model=model, nLayer=nLayer, fmin=fmin, fmax=fmax,
                                                     nf=nf, vmin=vmin, vmax=vmax, nvel=nvel, path=path,
                                                     source_depth=source_depth, receiver_depth=receiver_depth,
                                                     q_artif=q_artif, showIm=True, returnAxis=True, return_raw=False)

        # # for MSEISFK return_raw=False and propagate_noise=False
        ModelsetImage = BEL1D.MODELSET.DC_FW_image_bis(prior=prior_Q, priorDist='Uniform', priorBound=None, fMaxCalc=fMax,
                                                       xReceivers=xReceivers, source_sw=source_sw,
                                                       Tacq=Tacq, settingsSW=settings, rho_fixed=False, rho_val=2000, Q=True, Qalphas_fixed=False,
                                                       propagate_noise=False, waterlayer=True, model=model, nLayer=nLayer, fmin=fmin, fmax=fmax,
                                                       nf=nf, vmin=vmin, vmax=vmax, nvel=nvel, path=path,
                                                       source_depth=source_depth, receiver_depth=receiver_depth,
                                                       q_artif=q_artif, returnAxis=True)



        np.savetxt('fSyn.txt', fSyn) # needed for plots in read_postbel.py
        np.savetxt('vSyn.txt', vSyn) # needed for plots in read_postbel.py
        np.savetxt('dataModel.txt', dataModel)
        # np.savetxt('u_z_time_cpp_Levin.txt', u_z_time_cpp_Levin)

        # fSyn = np.loadtxt('fSyn.txt')
        # vSyn = np.loadtxt('vSyn.txt')

        # if forward model already run (comment ForwardFWIMAGE):
        # dataModel, u_z_time_cpp_Levin = ModelsetImage.forwardFun['Fun'](model)

        ###########
        # IF SYNTHETIC:
        dataSyn = dataModel #+ np.abs(np.random.rand(len(dataModel))*1e-1)
        ###########




        # # RUN PREBEL AND POSTBEL
        Prebel = BEL1D.PREBEL(MODPARAM=ModelsetImage, nbModels=10000)      # activate if not prebel saved

        # # We then run the prebel operations:
        Prebel.run(verbose=verbose)       # activate if not prebel saved


        BEL1D.SavePREBEL(Prebel, Filename='prebel_model')     # activate if not prebel saved
        # path_prebel = 'YOURPATH'
        # Prebel = BEL1D.LoadPREBEL(path_prebel + 'prebel_model.prebel')



        # Then, since we know the dataset, we can initialize the "post-bel" operations:
        Postbel = BEL1D.POSTBEL(Prebel)

        # Run the operations:
        Postbel.run(Dataset=dataSyn, nbSamples=10000, verbose=verbose) #, NoiseModel=1e-12)

        Postbel.ShowPostModels(TrueModel=model, RMSE=False)
        Postbel.ShowPostModels(TrueModel=model, RMSE=True)


        BEL1D.SavePOSTBEL(Postbel, Filename='postbel_model')
        BEL1D.SaveSamples(Postbel, Data=True, Filename='postbel_samples')

        # # Optional plots
        ## For FW images prior and datasets, showing the mean prior
        prior_data = np.reshape(Prebel.FORWARD, (Prebel.nbModels, len(vSyn), len(fSyn)))
        mean = np.mean(prior_data, axis=0)
        std = np.std(prior_data, axis=0)
        # vSyn = np.divide(1, pSyn)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(mean, aspect='auto', origin='lower', extent=(fSyn[0], fSyn[-1], vSyn[0], vSyn[-1]),
                  interpolation='bilinear')
        ax.set_xlabel('Frequency [Hz]', fontsize=14)
        ax.set_ylabel('Phase velocity [m/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(std, aspect='auto', origin='lower', extent=(fSyn[0], fSyn[-1], vSyn[0], vSyn[-1]),
                  interpolation='bilinear')
        ax.set_xlabel('Frequency [Hz]', fontsize=14)
        ax.set_ylabel('Phase velocity [m/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.tight_layout()


        Postbel.KDE.ShowKDE(
            Xvals=Postbel.CCA.transform(Postbel.PCA['Data'].transform(np.reshape(dataSyn, (1, -1)))))
        pyplot.show()

        post_data = np.reshape(Postbel.SAMPLESDATA, (Postbel.nbSamples, len(vSyn), len(fSyn)))
        mean = np.mean(post_data, axis=0)
        std = np.std(post_data, axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(mean, aspect='auto', origin='lower', extent=(fSyn[0], fSyn[-1], vSyn[0], vSyn[-1]),
                  interpolation='bilinear')
        ax.set_xlabel('Frequency [Hz]', fontsize=14)
        ax.set_ylabel('Phase velocity [m/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(std, aspect='auto', origin='lower', extent=(fSyn[0], fSyn[-1], vSyn[0], vSyn[-1]),
                  interpolation='bilinear')
        ax.set_xlabel('Frequency [Hz]', fontsize=14)
        ax.set_ylabel('Phase velocity [m/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.tight_layout()


    #########################################################################################
    ###                               Closing the parallel pool                           ###
    #########################################################################################
    if ParallelComputing:
        pool.terminate()

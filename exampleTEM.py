"""
extensive test script for synthetic TEM data


TODO:
    [] stream line to fit to examples of other methods
    [] add conductivity boolean switch
    [] check mapping to simpeg mesh with interpolation again
    ...

@author: lukas
"""
# %% import necessary modules
import os
import time
import numpy as np # For the initialization of the parameters

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


from pyBEL1D import BEL1D
from pyBEL1D.utilities import Tools # For further post-processing

from TEM_frwrd.empymod_frwrd import empymod_frwrd

# from TEM_frwrd.BEL1D_TEM_tools import parse_TEMfastFile
from TEM_frwrd.BEL1D_TEM_tools import multipage
from TEM_frwrd.BEL1D_TEM_tools import save_all
# from TEM_frwrd.BEL1D_TEM_tools import mtrxMdl2vec
from TEM_frwrd.BEL1D_TEM_tools import get_stduniform_fromPrior
from TEM_frwrd.BEL1D_TEM_tools import reshape_model
from TEM_frwrd.BEL1D_TEM_tools import mdl2steps
# from TEM_frwrd.BEL1D_TEM_tools import plot_mdl_memima
from TEM_frwrd.BEL1D_TEM_tools import plot_prior_space
# from TEM_frwrd.BEL1D_TEM_tools import plot_frwrdComp
from TEM_frwrd.BEL1D_TEM_tools import plot_fit
from TEM_frwrd.BEL1D_TEM_tools import query_yes_no






# %% -------------------------------------------------------------------------
# %% directions and basic settings
version = 'v0'
nbIter = 10  # set to None for no Iteration

nbModPre = 3000  # low number for quicker testing
nbModPos = 500
total_frwrds = nbModPre + nbModPos


savepath_figs = f'./results_tem/syndata/{version}/'
if not os.path.exists(savepath_figs):
    os.makedirs(savepath_figs)

savepath_csv = f'./results_tem/syndata/{version}/csv/'
if not os.path.exists(savepath_csv):
    os.makedirs(savepath_csv)


# %% -------------------------------------------------------------------------
# %% setup BEL1D, define prior
# Parameters for the tested model para0 = thk, para1 = res
# para0lay1, para0lay2; para1lay1, para1lay2, para1lay3
# modelTrue = np.asarray([5, 10, 20, 5, 50])  # 3 layer case; true model
modelTrue = np.asarray([2.5, 5, 12, 5,
                        25, 15, 5, 30, 60]) # 5 layer case, last 5 rho of 5 layers

# priorTEM = np.array([[1, 10, 5, 40], [5, 25, 1, 15], [0, 0, 25, 75]])  # 3 layer case prior space
priorTEM = np.array([[1, 10, 5, 40],[1, 10, 1, 30], [5, 25, 1, 15],
                     [1, 15, 10, 55], [0, 0, 40, 75]])

header = ['thk_min', 'thk_max', 'rho_min', 'rho_max', ]
np.savetxt(savepath_csv + f'prior_{version}.csv', priorTEM,
           fmt='%.1f,%.1f,%.1f,%.1f', comments='', delimiter=',',
           header=','.join(header))

prior_elems = priorTEM.shape[0]* priorTEM.shape[1]
nLayers = priorTEM.shape[0]
nParams = int(priorTEM.shape[1] / 2)


# %% -------------------------------------------------------------------------
# %% Simpeg setup
# CHECK noise estimation of frwrd solver correct forwarding? what does bel1d need?
# frwrd solver with clean data and adds error model later, or does it need the estimated noise added to the syntetic data?
setup_simpeg = {'coredepth': 80,               # max depth of simpeg core mesh
                'csz': 1,                      # core cell size in z, x size is multiplied by 3
                'relerr': 0.00001,             # keep like this for clean data
                'abserr': 1e-16}               # noise will be added after calculating clean data set

coredepth = setup_simpeg['coredepth']
csz = setup_simpeg['csz']
relerr = 0.003
abserr = 1e-11
bottom2plot = coredepth

device = 'TEMfast'
setup_device = {"timekey": 3,
                "txloop": 12.5,
                "rxloop": 12.5,
                "current": 1,                          # current key, either 1 or 4
                "filter_powerline": 50}


forward = simpeg_frwrd(setup_device=setup_device,
                       setup_simpeg=setup_simpeg,
                       device=device,
                       nlayer=nLayers, nparam=nParams)
times_rx = forward.times_rx

bottom2plot = coredepth
stdTrue = get_stduniform_fromPrior(priorTEM)


# %% -------------------------------------------------------------------------
# %% plot prior space
xtick_spacing_major = 10
xtick_spacing_minor = 1
ytick_spacing_major = 10
ytick_spacing_minor = 1

_, ax = plot_prior_space(priorTEM)

if not modelTrue is None:
    true_mdl = reshape_model(modelTrue, nLayers, nParams)
    true_cum, true2plt = mdl2steps(true_mdl, extend_bot=25, cum_thk=True)
    ax.plot(true2plt[:,1],true2plt[:,0], 'dodgerblue', ls='-',
            marker='.')
    lines = [ax.lines[-2], ax.lines[-1]]
    leg_labels = ['prior mean', 'true model', 'prior space']
else:
    lines = [ax.lines[-1]]
    leg_labels = ['prior mean', 'prior space']

ax.xaxis.set_major_locator(MultipleLocator(xtick_spacing_major))
# ax.xaxis.set_major_formatter('{x:.1f}')
ax.xaxis.set_minor_locator(MultipleLocator(xtick_spacing_minor))

ax.yaxis.set_major_locator(MultipleLocator(ytick_spacing_major))
# ax.yaxis.set_major_formatter('{x:.1f}')
ax.yaxis.set_minor_locator(MultipleLocator(ytick_spacing_minor))

ax.grid(which='major', color='lightgray', linestyle='-')
ax.grid(which='minor', color='lightgray', linestyle=':')
ax.set_xlim(0, 80)
ax.set_ylim(80, -1)
# legend control
lines.append(Line2D([0], [0], linestyle="none",
                    marker="s", alpha=1, ms=10,
                    mfc="lightgrey", mec='lightgrey'))
ax.legend(lines, leg_labels,
             loc='lower left', fancybox=True, framealpha=1,
             facecolor='white', frameon=True, edgecolor='k')
plt.savefig(savepath_figs + f'prior_space_{version}.png', dpi=150)
# plt.close('all')

if nbIter is not None:
    print(f'\n...about to start {nbIter} iterations of bel1d')
    print(f'\n...each iteration uses: {total_frwrds} frwrd runs (prebel + postbel)')
    if query_yes_no('Proceed? ', default='no'):
        means = np.zeros((nbIter,len(modelTrue)))
        stds = np.zeros((nbIter,len(modelTrue)))
        timings = np.zeros((nbIter,))
        ws_distances = np.zeros((nbIter,))
        rms_datas = np.zeros((nbIter,2))
        rms_rhoas = np.zeros((nbIter,2))
        data_comps = np.zeros((len(times_rx), nbIter+2))  # + 2 for Timings and Dataset_pre
        rhoa_comps = np.zeros((len(times_rx), nbIter+2))  # + 2 for Timings and Dataset_pre
        start = time.time()

        parallel = False
        pool = None#pp.ProcessPool(mp.cpu_count())
        diverge = True
        distancePrevious = 1e10
        MixingUpper = 0
        MixingLower = 1

        for idxIter in range(nbIter):
            if idxIter == 0: # Initialization
                # %% prebel --------------------------------------------------
                # To first declare the parameters, we call the constructor MODELSET().TEM() with the right parameters
                TestCase = BEL1D.MODELSET().TEM(prior=priorTEM, timing=None,
                                                device_sttngs=setup_device,
                                                simpeg_sttngs=setup_simpeg)
                # Then, we build the "pre-bel" operations using the PREBEL function
                PrebelIter = BEL1D.PREBEL(TestCase, nbModels=nbModPre)
                # We then run the prebel operations:
                PrebelIter.run(Parallelization=[parallel,pool])
                PrebelIter.KDE.ShowKDE()  # show kernel density estimator

                ModLastIter = PrebelIter.MODELS

                # %% simulate data -------------------------------------------
                # Then, we generate the synthetic benchmark dataset (with noise)
                simdata_clean = PrebelIter.MODPARAM.forwardFun["Fun"](modelTrue)
                print('\n - simulated data:')
                print(simdata_clean)

                data_comps[:,0] = times_rx
                rhoa_comps[:,0] = times_rx

                # TODO fix noise estimation for TEM [DONE]
                np.random.seed(42)
                rndm = np.random.randn(len(simdata_clean))
                noise_calc_rand = (relerr * np.abs(simdata_clean) +
                                   abserr) * rndm
                simdata_noisy = noise_calc_rand + simdata_clean
                data_comps[:,1] = simdata_noisy

                print('\n\n------------------------------------------------------')
                print(f'Starting iteration: {idxIter+1}')
                print('------------------------------------------------------\n')

                PostbelTest = BEL1D.POSTBEL(PrebelIter)
                PostbelTest.run(simdata_noisy,
                                nbSamples=nbModPos,
                                NoiseModel=noise_calc_rand)#NoiseModel=[0.005,100])
                means[idxIter,:], stds[idxIter,:] = PostbelTest.GetStats()
                end = time.time()
                timings[idxIter] = (end-start) / 60  # to min

                dim_str = ['it{:02d}_dim{:03d}'.format(idxIter+1, dim+1) for dim in range(0,len(modelTrue))]
                add_str = ['it{:02d}_post01_Corr'.format(idxIter+1),
                           'it{:02d}_post02_MDLvis'.format(idxIter+1),
                           'it{:02d}_post03_dataFIT'.format(idxIter+1)]
                filenames = dim_str + add_str

            else:
                # %% prebel iter ---------------------------------------------
                ModLastIter = PostbelTest.SAMPLES
                # Here, we will use the POSTBEL2PREBEL function that adds the POSTBEL from previous iteration to the prior (Iterative prior resampling)
                # However, the computations are longer with a lot of models, thus you can opt-in for the "simplified" option which randomely select up to 10 times the numbers of models
                MixingUpper += 1
                MixingLower += 1
                Mixing = MixingUpper/MixingLower
                PrebelIter = BEL1D.PREBEL.POSTBEL2PREBEL(PREBEL=PrebelIter,
                                                         POSTBEL=PostbelTest,
                                                         Dataset=simdata_noisy,
                                                         NoiseModel=noise_calc_rand,
                                                         Parallelization=[parallel,pool],
                                                         # Simplified=True,
                                                         # nbMax=nbModPre,
                                                         MixingRatio=Mixing)

                print('\n\n------------------------------------------------------')
                print(f'Starting iteration: {idxIter+1}')
                print('------------------------------------------------------\n')
                PostbelTest = BEL1D.POSTBEL(PrebelIter)
                PostbelTest.run(simdata_noisy,
                                nbSamples=nbModPos,
                                NoiseModel=noise_calc_rand)
                means[idxIter,:], stds[idxIter,:] = PostbelTest.GetStats()
                end = time.time()
                timings[idxIter] = (end-start) / 60  # to min

                filenames = ['it{:02d}_post01_Corr'.format(idxIter+1),
                             'it{:02d}_post02_MDLvis'.format(idxIter+1),
                             'it{:02d}_post03_dataFIT'.format(idxIter+1)]

            # %% check divergence --------------------------------------------
            diverge, ws_distances[idxIter] = Tools.ConvergeTest(SamplesA=ModLastIter,
                                                            SamplesB=PostbelTest.SAMPLES,
                                                            tol=1e-5)
            print('KS distance: {}'.format(ws_distances[idxIter]))
            if not(diverge) or (abs((distancePrevious-ws_distances[idxIter])/distancePrevious)*100<1):
                # Convergence acheived if:
                # 1) Distance below threshold
                # 2) Distance does not vary significantly (less than 2.5%)
                print('Model has converged at iter {}!'.format(idxIter+1))
                break
            rt_min = timings[idxIter]  # calc runtime in min
            start = time.time()
            PostbelTest.ShowPostCorr(TrueModel=modelTrue,OtherMethod=PrebelIter.MODELS)
            PostbelTest.ShowPostModels(TrueModel=modelTrue,
                                       RMSE=True, Parallelization=[parallel,pool])

            # %% calc fit and RMS for each iteration -------------------------
            (fig0, ax0,
             predicted, rms_sig, pre_rhoa,
             post_rhoa, rms_rhoa) = plot_fit(forward, simdata_noisy,
                                             means[idxIter])

            # print('rms: {:20.10f} '.format(rms_sig) +
                  # 'of iteration: {:02d}'.format(idxIter+1))
            rms_datas[idxIter,:] = rms_sig
            rms_rhoas[idxIter,:] = rms_rhoa

            data_comps[:,idxIter+2] = predicted
            rhoa_comps[:,1] = pre_rhoa
            rhoa_comps[:,idxIter+2] = post_rhoa

            # plot means of solved model to post model comp
            figs = [plt.figure(n) for n in plt.get_fignums()]
            ax = figs[-2].axes

            rshpd_mdl = reshape_model(means[idxIter], nLayers, nParams)
            cum_thk = np.cumsum(rshpd_mdl[:,0])
            rshpd_mdl[:,0] = cum_thk  # calc cumulative thickness for plot
            rshpd_mdl[-1, 0] = bottom2plot
            thk = np.r_[0, rshpd_mdl[:,0].repeat(2, 0)[:-1]]
            model2plot = np.column_stack((thk, rshpd_mdl[:,1:].repeat(2, 0)))
            for i in range(0,nParams-1):
                print(i)
                ax[i].plot(model2plot[:,i+1], model2plot[:,0],
                           color='grey', ls='--', lw=2,
                           label='mean solved mdl')
                ax[i].legend(loc='best')

            # %% save plots from bel here...
            save_all(savepath_figs, filenames,
                     file_ext='.png', figs=None, dpi=150)
            plt.close('all')

            # save empty txt file with name containing nbIter and runtime
            progress_file = 'doneIT{:02d}_rt{:.3f}min.info'.format(idxIter+1, rt_min)
            with open(savepath_csv + progress_file, 'w') as fp:
                pass

        timings = timings[:idxIter+1]
        means = means[:idxIter+1,:]
        stds = stds[:idxIter+1,:]
        paramnames = PostbelTest.MODPARAM.paramNames["NamesS"] # For the legend of the futur graphs

        fig1, ax1 = plt.subplots(1,1)
        ax1.plot(np.arange(len(timings)), timings, '.-')
        ax1.set_ylabel('Computation Time [min]')
        ax1.set_xlabel('Iteration nb.')
        plt.show()

        fig2, ax2 = plt.subplots(1,1)
        ax2.plot(np.arange(len(timings)), np.divide(means,modelTrue), '.-')
        ax2.set_ylabel('Normalized means [/]')
        ax2.set_xlabel('Iteration nb.')
        ax2.legend(paramnames)
        plt.show()

        fig3, ax3 = plt.subplots(1,1)
        ax3.plot(np.arange(len(timings)), np.divide(stds,stdTrue), '.-')
        ax3.set_ylabel('Normalized standard deviations [/]')
        ax3.set_xlabel('Iteration nb.')
        ax3.legend(paramnames)
        plt.show()

        fig4, ax4 = plt.subplots(1,1)
        ax4.plot(np.arange(len(timings)), rms_datas[:,1], '.-')
        ax4.set_ylabel('misfit rRMS - signal (%)')
        ax4.set_xlabel('Iteration nb.')
        plt.show()

        fig5, ax5 = plt.subplots(1,1)
        ax5.plot(np.arange(len(timings)), rms_rhoas[:,1], '.-')
        ax5.set_ylabel('misfit rRMS - rhoa (%)')
        ax5.set_xlabel('Iteration nb.')
        plt.show()

        filenames = ['trs_comp_time', 'trs_norm_means',
                     'trs_norm_stds', 'trs_rms_sig', 'trs_rms_rhoa']
        save_all(savepath_figs, filenames,
                  file_ext='.png', figs=None, dpi=150)

        means_hdr = ';'.join(paramnames) + ';timing(min);' + 'ws_dist;' + 'arms_sig;'+ 'arms_rhoa'
        means_export = np.column_stack((means, timings, ws_distances, rms_datas[:,0], rms_rhoas[:,0]))
        means_arr_cols = means_export.shape[1] - 1
        np.savetxt(savepath_csv + f'means_{version}_nIter{nbIter}_nMdls{nbModPre}.csv', means_export,
                   header=means_hdr, comments='', delimiter=';')

        std_hdr = ';'.join(paramnames) + ';timing(min);' + 'ws_dist;' + 'rrms_sig;'+ 'rrms_rhoa'
        stds_export = np.column_stack((stds, timings, ws_distances, rms_datas[:,1], rms_rhoas[:,1]))
        stds_arr_cols = stds_export.shape[1] - 1
        np.savetxt(savepath_csv + f'stds_{version}_nIter{nbIter}_nMdls{nbModPre}.csv', stds_export,
                    header=std_hdr, comments='', delimiter=';')

        iter_datanames = ['dataPost_{:02d}'.format(iters) for iters in range(0,nbIter)]
        datafit_header = 'times;dataPre;' + ';'.join(iter_datanames)
        fid_data_comps = savepath_csv + f'datafit_{version}_nIter{nbIter}_nMdls{nbModPre}.csv'
        np.savetxt(fid_data_comps, data_comps,
                   header=datafit_header, comments='', delimiter=';')

        iter_datanames = ['rhoaPost_{:02d}'.format(iters) for iters in range(0,nbIter)]
        datafit_header = 'times;rhoaPre;' + ';'.join(iter_datanames)
        fid_rhoa_comps = savepath_csv + f'rhoafit_{version}_nIter{nbIter}_nMdls{nbModPre}.csv'
        np.savetxt(fid_rhoa_comps, rhoa_comps,
                   header=datafit_header, comments='', delimiter=';')


# %% -------------------------------------------------------------------------
# %% no iterations...
else:
    print(f'\n...about to start {total_frwrds} frwrd runs (prebel + postbel)')
    if query_yes_no('Proceed? ', default='no'):
        # %% prebel ----------------------------------------------------------
        t0 = time.time()
        # To first declare the parameters, we call the constructor MODELSET().SNMR() with the right parameters
        TestCase = BEL1D.MODELSET().TEM(prior=priorTEM, timing=None,
                                        device_sttngs=setup_device,
                                        simpeg_sttngs=setup_simpeg)
        # Then, we build the "pre-bel" operations using the PREBEL function
        Prebel = BEL1D.PREBEL(TestCase, nbModels=nbModPre)
        # We then run the prebel operations:
        Prebel.run()
        # You can observe the relationship using:
        Prebel.KDE.ShowKDE()
        t1 = time.time()


        # %% simulate data ---------------------------------------------------
        # Then, we generate the synthetic benchmark dataset (with noise)
        simdata_clean = Prebel.MODPARAM.forwardFun["Fun"](modelTrue)

        # TODO fix noise estimation for TEM [DONE]
        np.random.seed(42)
        rndm = np.random.randn(len(simdata_clean))
        noise_calc_rand = (relerr * np.abs(simdata_clean) +
                           abserr) * rndm
        simdata_noisy = noise_calc_rand + simdata_clean


        # %% postbel ---------------------------------------------------------
        # Then, since we know the dataset, we can initialize the "post-bel" operations:
        t2 = time.time()
        time_prebel = time.strftime('%H:%M:%S', time.gmtime(t1-t0))
        print('\n #################################################################################')
        print(f"runtime of prebel operations:    {time_prebel} \n")
        print("starting postbel operations...\n\n")

        Postbel = BEL1D.POSTBEL(Prebel)
        # Run the operations:
        # Postbel.run(Dataset=simdata_noisy, nbSamples=1000, NoiseModel=10) noise propagation not yet operational for tem
        Postbel.run(Dataset=simdata_noisy, nbSamples=nbModPos, NoiseModel=noise_calc_rand)
        # All the operations are done, now, you just need to analyze the results (or run the iteration process - see next example)
        # Show the models parameters uncorrelated:
        Postbel.ShowPost(TrueModel=modelTrue)
        # Show the models parameters correlated with also the prior samples (Prebel.MODELS):
        Postbel.ShowPostCorr(TrueModel=modelTrue, OtherMethod=Prebel.MODELS)
        # Show the depth distributions of the parameters with the RMSE
        Postbel.ShowPostModels(TrueModel=modelTrue, RMSE=True)
        # Get key statistics
        means, stds = Postbel.GetStats()

        t3 = time.time()
        time_postbel = time.strftime('%H:%M:%S', time.gmtime(t3-t2))
        print('\n #################################################################################')
        print(f"runtime of postbel operations:    {time_postbel}\n")

        # %% save results ----------------------------------------------------
        time_total = time.strftime('%H:%M:%S', time.gmtime(t3-t0))

        log_runtime = f'# runtime: total, prebel, postbel\n# {time_total}, {time_prebel}, {time_postbel}\n'
        header = ('thk_mdlTrue,par1_mdlTrue,' +
                  'thk_means,par1_means,' +
                  'thk_stds,par1_stds')

        result_array = np.hstack((reshape_model(modelTrue, nLayers, nParams),
                                  reshape_model(means, nLayers, nParams),
                                  reshape_model(stds, nLayers, nParams)))
        res_arr_cols = result_array.shape[1] - 1
        np.savetxt(savepath_figs + f'result_{nbModPre}.csv', result_array,
                   header=log_runtime+header, comments='',
                   fmt='%.3f,'*res_arr_cols + '%.3f')

        (figp, axp,
         predicted, rms_sig, pre_rhoa,
         post_rhoa, rms_rhoa) = plot_fit(forward, simdata_noisy, mdl_means_post=means)

        dim_str = [f'dimension{dim}' for dim in range(0,nLayers+nParams)]
        ph_str = [f'posthist{dim}' for dim in range(0,nLayers+nParams)]
        add_str = ['post_mdl_space', 'post_model_vis', 'post_data_fit']
        filenames = dim_str + ph_str + add_str

        save_all(savepath_figs, filenames,
                 file_ext='.png', figs=None, dpi=150)
        multipage(savepath_figs + f'bel1d_syndata_{version}.pdf', dpi=100)

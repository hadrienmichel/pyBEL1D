# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:18:06 2021

additional tools specifically for BEL1D TEM usage

@author: lukas
"""
# %% import modules
import os
import sys
import time
import numpy as np # For the initialization of the parameters
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from matplotlib.backends.backend_pdf import PdfPages
from scipy.constants import mu_0


# %% collection of functions
def parse_TEMfastFile(filename, path):
    """
    read a .tem file and return as a pd.dataframe !!Convert first!!

    Parameters
    ----------
    filename : string
        name of file including extension.
    path : string
        path to file - either absolute or relative.

    Returns
    -------
    rawData : pd.DataFrame
        contains all data in one file.
    nLogs : int
        number of soundings in the .tem data file.
    indices_hdr : pd.DataFrame
        Contains the indices where the header block of 
        each sounding starts and ends.
    indices_dat : pd.DataFrame
        Contains the indices where the data block of 
        each sounding starts and ends..

    """
    headerLines = 8
    # properties = generate_props('TEMfast')

    # fName = filename[:-4] if filename[-4:] == '_txt' else filename
    fin = path + '/' + filename

    # Start of file reading
    myCols = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    rawData = pd.read_csv(fin,
                          names=myCols,
                          sep='\\t',
                          engine="python")
    rawData = rawData[~pd.isnull(rawData).all(1)].fillna('')
    lengthData = len(rawData.c1)

    # create start and end indices of header and data lines
    start_hdr = np.asarray(np.where(rawData.loc[:]['c1'] ==
                                    'TEM-FAST 48 HPC/S2  Date:'))
    nLogs = np.size(start_hdr)
    start_hdr = np.reshape(start_hdr, (np.size(start_hdr),))
    end_hdr = start_hdr + headerLines

    start_dat = end_hdr
    end_dat = np.copy(start_hdr)
    end_dat = np.delete(end_dat, 0)
    end_dat = np.append(end_dat, lengthData)

    # create new dataframe which contains all indices
    indices_hdr = pd.DataFrame({'start': start_hdr, 'end': end_hdr},
                               columns=['start', 'end'])
    indices_dat = pd.DataFrame({'start': start_dat, 'end': end_dat},
                               columns=['start', 'end'])

    return rawData, nLogs, indices_hdr, indices_dat


def multipage(filename, figs=None, dpi=200):
    """
    function to save all plots to multipage pdf
    https://stackoverflow.com/questions/26368876/saving-all-open-matplotlib-figures-in-one-file-at-once

    Parameters
    ----------
    filename : string
        name of the the desired pdf file including the path and file extension.
    figs : list, optional
        list of instances of figure objects. If none automatically retrieves currently opened figs.
        The default is None.
    dpi : int, optional
        dpi of saved figures. The default is 200.

    Returns
    -------
    None.

    """
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf', dpi=dpi)
    pp.close()


def save_all(savepath, filenames,
             file_ext='.png', figs=None, dpi=150):
    """
    function to save all open figures to individual files

    Parameters
    ----------
    savepath : string
        path where the figures should be saved.
    filename : string
        name of the the desired pdf file including the file extension.
    file_ext : string, optional
        extension of the saved files. The default is '.png'.
    figs : list, optional
        list of instances of figure objects. If none automatically retrieves currently opened figs.
        The default is None.
    dpi : int, optional
        dpi of saved figures. The default is 200.

    Returns
    -------
    None.

    """
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        print(len(figs), '... number of opened figs.')
    for id, fig in enumerate(figs):
        try:
            fig.savefig(savepath + filenames[id] + file_ext, dpi=dpi)
        except IndexError:
            fig.savefig(savepath + f'unknownfig_{id}' + file_ext, dpi=dpi)


def mtrxMdl2vec(mtrx):
    """
    reshapes a multicolumn model to a 1D vector assuming the structure
    as required by bel1d

    Parameters
    ----------
    mtrx : 2D - np.array
        array containing parameter values in the rows and different params in columns.
        uses thk of each individual layer in such structure that:
            thk_lay_0,     param1_lay_0,    param2_lay_0,   ....  param_n_lay_0
            thk_lay_1,     param1_lay_1,    param2_lay_1,   ....  param_n_lay_1
            .              .                .               ....  .            
            .              .                .               ....  .            
            thk_lay_n-1,   param1_lay_n-1,  param2_lay_n-1, .... param_n_lay_n-1
            0,             param1_lay_n,    param2_lay_n,   .... param_n-1_lay_n
         

    Returns
    -------
    mtrx_1D : np.array (1D)
        1D array (or vector) containing the same info as mtrx
        but reshaped to:
            thk_lay_0
            thk_lay_1
            .
            .
            thk_lay_n-1 
            param1_lay_0
            param1_lay_1
            .
            .
            param1_lay_n
            .
            .
            .
            param_n_lay_0
            param_n_lay_1
            .
            .
            param_n_lay_n
    """
    nLayers = mtrx.shape[0]
    nParams = mtrx.shape[1]
    for par in range(nParams):
        if par == 0:
            mtrx_1D = mtrx[:-1,0]
        else:
            mtrx_1D = np.hstack((mtrx_1D, mtrx[:,par]))
    return mtrx_1D


def get_stduniform_fromPrior(prior_space):
    """
    

    Parameters
    ----------
    prior_space : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    prior_elems = prior_space.shape[0]* prior_space.shape[1]
    stdUniform = lambda a,b: (b-a)/np.sqrt(prior_elems)
    nLayers = prior_space.shape[0]
    nParams = int(prior_space.shape[1] / 2)
    stdTrue = []
    for par in range(nParams):
        par *= 2
        for lay in range(nLayers):
            if prior_space[lay,par] == 0:
                pass
            else:
                stdTrue.append(stdUniform(prior_space[lay,par],
                                          prior_space[lay,par+1]))
    return np.asarray(stdTrue)


def reshape_model(model, nLayer, nParam):
    """
    function to reshape a 1D bel1d style model to a n-D model containing as
    many rows as layers and as many columns as parameters (thk + nParams)

    Parameters
    ----------
    model : np.array
        bel1D vector model:
            thk_lay_0
            thk_lay_1
            .
            .
            thk_lay_n-1 
            param1_lay_0
            param1_lay_1
            .
            .
            param1_lay_n
            .
            .
            .
            param_n_lay_0
            param_n_lay_1
            .
            .
            param_n_lay_n
    nLayer : int
        number of layers in the model.
    nParam : int
        number of parameters in the model, thk also counts!!

    Returns
    -------
    model : np.array
        n-D array with the model params.

    """
    mdlRSHP = np.zeros((nLayer,nParam))
    i = 0
    for col in range(0, nParam):
        for row in range(0, nLayer):
            if col == 0 and row == nLayer-1:
                pass
            else:
                mdlRSHP[row, col] = model[i]
                i += 1
    model = mdlRSHP
    return model


def mdl2steps(mdl, extend_bot=25, cum_thk=True):
    """
    

    Parameters
    ----------
    mdl : TYPE
        DESCRIPTION.
    extend_bot : TYPE, optional
        DESCRIPTION. The default is 25.
    cum_thk : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    mdl_cum : TYPE
        DESCRIPTION.
    model2plot : TYPE
        DESCRIPTION.

    """
    mdl_cum = mdl.copy()
    
    if cum_thk:
        cum_thk = np.cumsum(mdl[:,0]).copy()
        mdl_cum[:,0] = cum_thk  # calc cumulative thickness for plot

    mdl_cum[-1, 0] = mdl_cum[-2, 0] + extend_bot

    thk = np.r_[0, mdl_cum[:,0].repeat(2, 0)[:-1]]
    model2plot = np.column_stack((thk, mdl_cum[:,1:].repeat(2, 0)))
    
    return mdl_cum, model2plot


def plot_mdl_memima(mdl_mean, mdl_min, mdl_max, extend_bot=25, ax=None):
    """
    function to plot the min and max of a model space.
    mainly intended for bugfixing the plot_prior space function

    Parameters
    ----------
    mdl_mean : np.array
        means of prior space.
    mdl_min : np.array
        mins of prior space.
    mdl_max : np.array
        maxs of prior space.
    extend_bot : int, optional
        extension of bottom layer. The default is 25.
    ax : pyplot axis object, optional
        for plotting to an already existing axis. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    prior_min_cum, min2plot = mdl2steps(mdl_min,
                                        extend_bot=extend_bot, cum_thk=True)
    prior_max_cum, max2plot = mdl2steps(mdl_max,
                                        extend_bot=extend_bot, cum_thk=True)
    prior_mea_cum, mean2plot = mdl2steps(mdl_mean,
                                         extend_bot=extend_bot, cum_thk=True)
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
    
    ax.plot(mean2plot[:,1], mean2plot[:,0],
            color='k', ls='-', lw=2,
            zorder=10)
    ax.plot(min2plot[:,1], min2plot[:,0],
            color='c', ls='--', lw=2,
            zorder=5)
    ax.plot(max2plot[:,1], max2plot[:,0],
            color='m', ls=':', lw=2,
            zorder=5)
    ax.invert_yaxis()
    if ax is None:
        return fig, ax
    else:
        return ax


def plot_prior_space(prior, show_patchcorners=False):
    """
    function to visualize the prior model space
    could be also used to visualize the solved means and stds
    
    use with caution - might be buggy still...
        TODO - fix possible bugs!!
    
    Parameters
    ----------
    prior : np.array
        prior model vector.
    show_patchcorners : boolean, optional
        switch to decide whether to show the corners in the plot.
        useful for debugging. The default is False.

    Returns
    -------
    fig_pr : pyplot figure object
        figure object.
    ax_pr : pyplot axis object
        1 subplot

    """
    
    prior_min = np.column_stack((prior[:,0], prior[:,2]))
    prior_max = np.column_stack((prior[:,1], prior[:,3]))
    prior_mean = np.mean(np.array([prior_min, prior_max]), axis=0)
    
    prior_min_cum, min2plot = mdl2steps(prior_min,
                                        extend_bot=25, cum_thk=True)
    prior_max_cum, max2plot = mdl2steps(prior_max,
                                        extend_bot=25, cum_thk=True)
    prior_mea_cum, mean2plot = mdl2steps(prior_mean,
                                         extend_bot=25,cum_thk=True)
    
    fig_pr, ax_pr = plt.subplots(1,1, figsize=(6,9))
    
    k = 0
    patches_min = []
    for prior_cum in [prior_min_cum, prior_max_cum]:
        print(prior_cum)
        for i, thk in enumerate(prior_cum[:,0]):
            print('layerID', i)
            print('curr_thickness', thk)
            r_rli_min = prior_min_cum[i,1]
            r_rli_max = prior_max_cum[i,1]
            print('minmax_rho', r_rli_min, r_rli_max)
            
            color = 'r' if k % 2 == 0 else 'g'
            mrkr = '.' if k % 2 == 0 else 'o'
            mfc = None if k % 2 == 0 else 'None'
        
            if i == 0:
                thk_top = 0
            else:
                thk_top = prior_min_cum[i-1,0]
            corners_xy = [[r_rli_min, thk_top],
                          [r_rli_max, thk_top],
                          [r_rli_max, thk],
                          [r_rli_min, thk],
                          ]
            print(corners_xy)
            if show_patchcorners:
                ax_pr.plot(np.asarray(corners_xy)[:,0],
                            np.asarray(corners_xy)[:,1],
                            mrkr, color=color,
                            mfc=mfc)
                alpha = 0.3
            else:
                alpha = 1
            patches_min.append(Polygon(corners_xy,
                            color='lightgrey', alpha=alpha,
                            zorder=0))
        k += 1

    p = PatchCollection(patches_min, match_original=True)
    ax_pr.add_collection(p)
    ax_pr.plot(mean2plot[:,1], mean2plot[:,0],
               'k.--', label='prior mean')
    ax_pr.invert_yaxis()
    
    ax_pr.set_xlabel(r'$\rho$ ($\Omega$m)')
    ax_pr.set_ylabel('Depth (m)')
    return fig_pr, ax_pr
    

def plot_frwrdComp(obs_data_norm, frwrd_response, frwrd_times, frwrd_rhoa,
                   filter_times, show_rawdata=True, **kwargs):
    """

    Parameters
    ----------
    obs_data_norm : pd.DataFrame
        Normalized to V/m² with columns:
        ['channel', 'time', 'signal', 'err', 'rhoa'].
    frwrd_response : np.array (V/m²)
        response from forward calc.
    frwrd_times : np.array (s)
        times at which the frwrd response was calculated.
    filter_times : tuple (minT, maxT) (s)
        time range at which the data where filtered.
    **kwargs : mpl kwargs
        for setting markersize, linewidth, etc...

    Returns
    -------
    ax : pyplot axis object
        1 row, 2 columns - signal, app. Res.

    """
    minT = filter_times[0]
    maxT = filter_times[1]
    
    obs_sub = obs_data_norm[(obs_data_norm.time>minT) &
                            (obs_data_norm.time<maxT)]
    
    
    fig, ax = plt.subplots(nrows=1, ncols=2,
                           figsize=(12, 6),
                           sharex=True)

    if show_rawdata:
        ax[0].loglog(obs_data_norm.time, obs_data_norm.signal,
                     ':o', alpha=0.7,
                     color='k', label='raw data observed',
                     **kwargs)
    
    ax[0].loglog(obs_sub.time, obs_sub.signal,
                 ':o', alpha=0.7,
                  color='g', label='data observed (filtered)',
                  **kwargs)
    
    ax[0].loglog(frwrd_times, 
                 frwrd_response,
                  ':d', color='b', 
                  label='forward response SimPEG',
                  **kwargs)
    
    ax[0].set_title('measured signal vs. frwrd sol')
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('dbz/dt (V/m²)')
    ax[0].set_xlim((1e-6, 1e-3))
    ax[0].legend(loc='best')

    if show_rawdata:
        ax[1].loglog(obs_data_norm.time, obs_data_norm.rhoa,
                     ':o', alpha=0.7,
                     color='k', label='raw rhoa observed',
                     **kwargs)
    
    ax[1].loglog(obs_sub.time, obs_sub.rhoa,
                 ':o', alpha=0.7,
                  color='g', label='rhoa observed (filtered)',
                  **kwargs)
    
    ax[1].loglog(frwrd_times, 
                 frwrd_rhoa,
                  ':d', color='b', 
                  label='forward response SimPEG',
                  **kwargs)
    
    ax[1].set_title('apparent resistivity after christiansen paper...')
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel(r'$\rho_a$ ($\Omega$m)')
    # ax[1].set_ylim((min(frwrd_rhoa)*0.8, max(frwrd_rhoa)*1.2))
    ax[1].legend(loc='best')

    return fig, ax


def plot_fit(forward, data_pre, mdl_means_post, **kwargs):
    """
    function to plot the data fit after finishing a bel1d run

    Parameters
    ----------
    forward : simpeg forward operator
        DESCRIPTION.
    data_pre : np.array (float)
        tem data to solve for (V/m²).
    mdl_means_post : np.array
        solved models.
    **kwargs : matplotlib kwargs

    Returns
    -------
    fig : pyplot figure object
        figure object.
    ax : pyplot axis object
        1 row, 2 columns - signal, app. Res.
    post_sig : np.array (float)
        tem data calculated from mdl means post (V/m²).
    rms_sig : np.array (float)
        contains the absolute and relative rms of the misfit between observed and calculated data.
    pre_rhoa : np.array (float)
        tem data to solve for in terms of app. res - calculated here (Ohmm).
    post_rhoa : np.array (float)
        tem data calculated from mdl means post (Ohmm).
    rms_rhoa : np.array (float)
        contains the absolute and relative rms of the misfit between observed and calculated rhoa.

    """
    rx_times = forward.times_rx
    print('###############################################################')
    print('Post data fit plotting.... at receiver times:')
    print(rx_times)
    forward.calc_response(mdl_means_post)
    post_sig = forward.response
    print('reponse...')
    print(post_sig)
    
    # calc RMS of misfit
    misfit_sig = post_sig - data_pre
    misfit_sig_norm = misfit_sig / np.mean((post_sig, data_pre), axis=0)
    # misfit_sig_norm = np.mean((post_sig, data_pre), axis=0)/misfit_sig
    arms_sig = np.sqrt(np.mean(misfit_sig**2))
    rrms_sig = np.sqrt(np.mean(misfit_sig_norm**2))*100
    rms_sig = np.r_[arms_sig, rrms_sig]
    
    fig, ax = plt.subplots(nrows=1, ncols=2,
                           figsize=(12, 6),
                           sharex=True)
    
    ax[0].loglog(rx_times, data_pre,
                 'd', color='g',
                 label='synthetic data pre',
                  **kwargs)
    
    ax[0].loglog(rx_times, post_sig,
                  '-', alpha=0.7,
                  color='k', 
                  label='fitted data post',
                  **kwargs)
    
    ax[0].set_title(('abs RMS: {:10.3e} V/m² \n'.format(arms_sig) + 
                     'rel RMS: {:6.2f} % '.format(rrms_sig)))
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('dbz/dt (V/m²)')
    ax[0].set_xlim((1e-6, 1e-3))
    ax[0].legend(loc='best')
    # rhoa
    turns = 1
    Rx_area = forward.setup_device['rxloop']**2
    M = (forward.setup_device['current'] *
         forward.setup_device['txloop']**2 * turns)
    pre_rhoa = ((1 / np.pi) *
              (M / (20 * (data_pre)))**(2/3) *
              (mu_0 / (rx_times))**(5/3)
              )
    post_rhoa = ((1 / np.pi) *
              (M / (20 * (post_sig)))**(2/3) *
              (mu_0 / (rx_times))**(5/3)
              )
    # calc RMS of misfit
    misfit_rhoa = post_rhoa - pre_rhoa
    misfit_rhoa_norm = misfit_rhoa / np.mean((post_rhoa, pre_rhoa), axis=0)
    # misfit_rhoa_norm = np.mean((post_rhoa, pre_rhoa), axis=0) / misfit_rhoa
    arms_rhoa = np.sqrt(np.mean(misfit_rhoa**2))
    rrms_rhoa = np.sqrt(np.mean(misfit_rhoa_norm**2))*100
    rms_rhoa = np.r_[arms_rhoa, rrms_rhoa]
    
    ax[1].loglog(rx_times, pre_rhoa,
                 'd', color='g', label='rhoa pre',
                 **kwargs)

    ax[1].loglog(rx_times, post_rhoa,
                  '-', alpha=0.7,
                  color='k', label='fitted rhoa post',
                  **kwargs)

    ax[1].set_title(('abs RMS: {:6.2f} ($\Omega$m) \n'.format(arms_rhoa) +
                     'rel RMS: {:6.2f} % '.format(rrms_rhoa)))
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel(r'$\rho_a$ ($\Omega$m)')
    ax[1].yaxis.set_label_position('right')
    ax[1].yaxis.tick_right()
    # ax[1].set_ylim((min(post_rhoa)*0.8, max(post_rhoa)*1.2))
    ax[1].legend(loc='best')
    return fig, ax, post_sig, rms_sig, pre_rhoa, post_rhoa, rms_rhoa


def query_yes_no(question, default='no'):
    """
    yes no query for terminal usage
    from: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    
    Parameters
    ----------
    question : string
        query to ask the user.
    default : string, optional
        default answer. The default is 'no'.

    Raises
    ------
    ValueError
        if the expected variations of yes/no are not in the answer...

    Returns
    -------
    none

    """
    from distutils.util import strtobool
    if default is None:
        prompt = " (y/n)? "
    elif default == 'yes':
        prompt = " ([y]/n)? "
    elif default == 'no':
        prompt = " (y/[n])? "
    else:
        raise ValueError(f"Unknown setting '{default}' for default.")

    while True:
        try:
            resp = input(question + prompt).strip().lower()
            if default is not None and resp == '':
                return default == 'yes'
            else:
                return strtobool(resp)
        except ValueError:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


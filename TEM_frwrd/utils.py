#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:21:05 2022

utility function for empymod_frwrd_ip

@author: laigner
"""

# %% modules
import logging
import numpy as np
import pandas as pd

from scipy.constants import epsilon_0
from scipy.special import roots_legendre
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline

from scipy.constants import mu_0


# %% logging ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)
# logger.setLevel(logging.DEBUG)


# %% general ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def version_to_numeric(version):
    part0 = int(version.split('.')[0]) * 10000
    part1 = int(version.split('.')[1]) * 100
    part2 = int(version.split('.')[2]) * 1
    
    version_numeric = part0 + part1 + part2
    
    return version_numeric


def getR_fromSquare(a):
    return np.sqrt((a*a) / np.pi)


def scaling(signal):
    signal_min = np.min(np.abs(signal))
    signal_max = np.max(np.abs(signal))
    s = np.abs(signal_max / (10 * (np.log10(signal_max) - np.log10(signal_min))))
    return s


def arsinh(signal):
    s = scaling(signal)
    return np.log((signal/s) + np.sqrt((signal/s)**2 + 1))


def kth_root(x, k):
    """
    kth root, returns only real roots

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if k % 2 != 0:
        res = np.power(np.abs(x),1./k)
        return res*np.sign(x)
    else:
        return np.power(np.abs(x),1./k)


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


def simulate_error(relerr, abserr, data):
    np.random.seed(42)
    rndm = np.random.randn(len(data))

    rand_error_abs = (relerr * np.abs(data) +
                  abserr) * rndm

    return rand_error_abs


# %% forward preparations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_TEMFAST_rampdata(location, current_key='1A'):

    column_list = ['cable', 'side', 'turns', 'ramp_off']

    if current_key == '1A':
        if location == 'donauinsel':
            ramp_data_array = np.array([[  6. ,  1.5 , 1. , 0.15],
                                        [ 12. ,  3.  , 1. , 0.23],
                                        [ 25. ,  6.25, 1. , 0.4 ],
                                        [ 50. , 12.5 , 1. , 0.8 ],
                                        [100. , 25.  , 1. , 1.3 ]])

        elif location == 'salzlacken':
            ramp_data_array = np.array([[  8. ,  2.  , 1.  ,  0.21],
                                        [ 25. ,  6.25, 1.  ,  0.44],
                                        [ 50. , 12.5 , 1.  ,  0.8 ],
                                        [100. , 25.  , 1.  ,  1.3 ]])

        elif location == 'hengstberg':
            ramp_data_array = np.array([[ 25. ,  6.25 , 1. ,  0.44],
                                        [ 50. ,  12.5 , 1. ,  0.82]])

        elif location == 'sonnblick':
            ramp_data_array = np.array([[ 50. ,  12.5 , 1. ,  1.0],
                                        [100. ,  25.  , 1. ,  2.5],
                                        [200. ,  50.  , 1. ,  4.2]])

        else:
            raise ValueError('location of ramp data measurements is not available ...')

    elif current_key == '4A':
        if location == 'donauinsel':
            ramp_data_array = np.array([[  6. ,  1.5  , 1. ,  0.17],
                                        [ 25. ,  6.25 , 1. ,  0.45],
                                        [ 50. ,  12.5 , 1. ,  0.95],
                                        [100. ,  25.  , 1. ,  1.5],
                                        [400. , 100.  , 1. , 10.0]])

        elif location == 'salzlacken':
            ramp_data_array = np.array([[  8. ,  2.  , 1. ,  0.21],
                                        [ 25. ,  6.25, 1. ,  0.5 ],
                                        [ 50. , 12.5 , 1. ,  0.95],
                                        [100. , 25.  , 1. ,  1.5 ],
                                        [200. , 50.  , 1. ,  4.3 ],
                                        [400. ,100.  , 1. , 10.0]])

        elif location == 'hengstberg':
            ramp_data_array = np.array([[ 25. ,  6.25 , 1. ,  0.48],
                                        [ 50. ,  12.5 , 1. ,  0.98]])

        elif location == 'sonnblick':
            ramp_data_array = np.array([[ 50. ,  12.5 , 1. ,  1.15],
                                        [100. ,  25.  , 1. ,  2.70],
                                        [200. ,  50.  , 1. ,  5.10]])

        else:
            raise ValueError('location of ramp data measurements is not available ...')

    ramp_data = pd.DataFrame(ramp_data_array, columns=column_list)

    return ramp_data


def get_TEMFAST_timegates():
    
    tg_raw = np.array([[1.00000e+00, 3.60000e+00, 4.60000e+00, 4.06000e+00, 1.00000e+00],
       [2.00000e+00, 4.60000e+00, 5.60000e+00, 5.07000e+00, 1.00000e+00],
       [3.00000e+00, 5.60000e+00, 6.60000e+00, 6.07000e+00, 1.00000e+00],
       [4.00000e+00, 6.60000e+00, 7.60000e+00, 7.08000e+00, 1.00000e+00],
       [5.00000e+00, 7.60000e+00, 9.60000e+00, 8.52000e+00, 2.00000e+00],
       [6.00000e+00, 9.60000e+00, 1.16000e+01, 1.05300e+01, 2.00000e+00],
       [7.00000e+00, 1.16000e+01, 1.36000e+01, 1.25500e+01, 2.00000e+00],
       [8.00000e+00, 1.36000e+01, 1.56000e+01, 1.45600e+01, 2.00000e+00],
       [9.00000e+00, 1.56000e+01, 1.96000e+01, 1.74400e+01, 4.00000e+00],
       [1.00000e+01, 1.96000e+01, 2.36000e+01, 2.14600e+01, 4.00000e+00],
       [1.10000e+01, 2.36000e+01, 2.76000e+01, 2.54900e+01, 4.00000e+00],
       [1.20000e+01, 2.76000e+01, 3.16000e+01, 2.95000e+01, 4.00000e+00],
       [1.30000e+01, 3.16000e+01, 3.96000e+01, 3.52800e+01, 8.00000e+00],
       [1.40000e+01, 3.96000e+01, 4.76000e+01, 4.33000e+01, 8.00000e+00],
       [1.50000e+01, 4.76000e+01, 5.56000e+01, 5.14000e+01, 8.00000e+00],
       [1.60000e+01, 5.56000e+01, 6.36000e+01, 5.94100e+01, 8.00000e+00],
       [1.70000e+01, 6.36000e+01, 7.96000e+01, 7.16000e+01, 1.60000e+01],
       [1.80000e+01, 7.96000e+01, 9.56000e+01, 8.76000e+01, 1.60000e+01],
       [1.90000e+01, 9.56000e+01, 1.11600e+02, 1.03600e+02, 1.60000e+01],
       [2.00000e+01, 1.11600e+02, 1.27600e+02, 1.19600e+02, 1.60000e+01],
       [2.10000e+01, 1.27600e+02, 1.59600e+02, 1.43600e+02, 3.20000e+01],
       [2.20000e+01, 1.59600e+02, 1.91600e+02, 1.75600e+02, 3.20000e+01],
       [2.30000e+01, 1.91600e+02, 2.23600e+02, 2.07600e+02, 3.20000e+01],
       [2.40000e+01, 2.23600e+02, 2.55600e+02, 2.39600e+02, 3.20000e+01],
       [2.50000e+01, 2.55600e+02, 3.19600e+02, 2.85000e+02, 6.40000e+01],
       [2.60000e+01, 3.19600e+02, 3.83600e+02, 3.50000e+02, 6.40000e+01],
       [2.70000e+01, 3.83600e+02, 4.47600e+02, 4.14000e+02, 6.40000e+01],
       [2.80000e+01, 4.47600e+02, 5.11600e+02, 4.78000e+02, 6.40000e+01],
       [2.90000e+01, 5.11600e+02, 6.39600e+02, 5.70000e+02, 1.28000e+02],
       [3.00000e+01, 6.39600e+02, 7.67600e+02, 6.99000e+02, 1.28000e+02],
       [3.10000e+01, 7.67600e+02, 8.95600e+02, 8.28000e+02, 1.28000e+02],
       [3.20000e+01, 8.95600e+02, 1.02360e+03, 9.56000e+02, 1.28000e+02],
       [3.30000e+01, 1.02360e+03, 1.27960e+03, 1.15200e+03, 2.56000e+02],
       [3.40000e+01, 1.27960e+03, 1.53560e+03, 1.40800e+03, 2.56000e+02],
       [3.50000e+01, 1.53560e+03, 1.79160e+03, 1.66400e+03, 2.56000e+02],
       [3.60000e+01, 1.79160e+03, 2.04760e+03, 1.92000e+03, 2.56000e+02],
       [3.70000e+01, 2.04760e+03, 2.55960e+03, 2.30400e+03, 5.12000e+02],
       [3.80000e+01, 2.55960e+03, 3.07160e+03, 2.81600e+03, 5.12000e+02],
       [3.90000e+01, 3.07160e+03, 3.58360e+03, 3.32800e+03, 5.12000e+02],
       [4.00000e+01, 3.58360e+03, 4.09560e+03, 3.84000e+03, 5.12000e+02],
       [4.10000e+01, 4.09560e+03, 5.11960e+03, 4.60800e+03, 1.02400e+03],
       [4.20000e+01, 5.11960e+03, 6.14360e+03, 5.63200e+03, 1.02400e+03],
       [4.30000e+01, 6.14360e+03, 7.16760e+03, 6.65600e+03, 1.02400e+03],
       [4.40000e+01, 7.16760e+03, 8.19160e+03, 7.68000e+03, 1.02400e+03],
       [4.50000e+01, 8.19160e+03, 1.02396e+04, 9.21600e+03, 2.04800e+03],
       [4.60000e+01, 1.02396e+04, 1.22876e+04, 1.12640e+04, 2.04800e+03],
       [4.70000e+01, 1.22876e+04, 1.43356e+04, 1.33120e+04, 2.04800e+03],
       [4.80000e+01, 1.43356e+04, 1.63836e+04, 1.53600e+04, 2.04800e+03]])

    return pd.DataFrame(tg_raw, columns=['id', 'startT', 'endT', 'centerT', 'deltaT'])


def get_time(times, wf_times):
    """Additional time for ramp.

    Because of the arbitrary waveform, we need to compute some times before and
    after the actually wanted times for interpolation of the waveform.

    Some implementation details: The actual times here don't really matter. We
    create a vector of time.size+2, so it is similar to the input times and
    accounts that it will require a bit earlier and a bit later times. Really
    important are only the minimum and maximum times. The Fourier DLF, with
    `pts_per_dec=-1`, computes times from minimum to at least the maximum,
    where the actual spacing is defined by the filter spacing. It subsequently
    interpolates to the wanted times. Afterwards, we interpolate those again to
    compute the actual waveform response.

    Note: We could first call `waveform`, and get the actually required times
          from there. This would make this function obsolete. It would also
          avoid the double interpolation, first in `empymod.model.time` for the
          Fourier DLF with `pts_per_dec=-1`, and second in `waveform`. Doable.
          Probably not or marginally faster. And the code would become much
          less readable.

    Parameters
    ----------
    times : ndarray
        Desired times

    wf_times : ndarray
        Waveform times

    Returns
    -------
    time_req : ndarray
        Required times
    """
    tmin = np.log10(max(times.min()-wf_times.max(), 1e-10))
    tmax = np.log10(times.max()-wf_times.min())
    return np.logspace(tmin, tmax, times.size+2)


def waveform(times, resp, times_wanted, wave_time, wave_amp, nquad=3):
    """Apply a source waveform to the signal.

    Parameters
    ----------
    times : ndarray
        Times of computed input response; should start before and end after
        `times_wanted`.

    resp : ndarray
        EM-response corresponding to `times`.

    times_wanted : ndarray
        Wanted times. Rx-times at which the decay is observed

    wave_time : ndarray
        Time steps of the wave. i.e current pulse

    wave_amp : ndarray
        Amplitudes of the wave corresponding to `wave_time`, usually
        in the range of [0, 1].

    nquad : int
        Number of Gauss-Legendre points for the integration. Default is 3.

    Returns
    -------
    resp_wanted : ndarray
        EM field for `times_wanted`.

    """

    # Interpolate on log.
    PP = iuSpline(np.log10(times), resp)

    # Wave time steps.
    dt = np.diff(wave_time)
    dI = np.diff(wave_amp)
    dIdt = dI/dt

    # Gauss-Legendre Quadrature; 3 is generally good enough.
    # (Roots/weights could be cached.)
    g_x, g_w = roots_legendre(nquad)

    # Pre-allocate output.
    resp_wanted = np.zeros_like(times_wanted)

    # Loop over wave segments.
    for i, cdIdt in enumerate(dIdt):

        # We only have to consider segments with a change of current.
        if cdIdt == 0.0:
            continue

        # If wanted time is before a wave element, ignore it.
        ind_a = wave_time[i] < times_wanted
        if ind_a.sum() == 0:
            continue

        # If wanted time is within a wave element, we cut the element.
        ind_b = wave_time[i+1] > times_wanted[ind_a]

        # Start and end for this wave-segment for all times.
        ta = times_wanted[ind_a]-wave_time[i]
        tb = times_wanted[ind_a]-wave_time[i+1]
        tb[ind_b] = 0.0  # Cut elements

        # Gauss-Legendre for this wave segment. See
        # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
        # for the change of interval, which makes this a bit more complex.
        logt = np.log10(np.outer((tb-ta)/2, g_x)+(ta+tb)[:, None]/2)
        fact = (tb-ta)/2*cdIdt
        resp_wanted[ind_a] += fact*np.sum(np.array(PP(logt)*g_w), axis=1)

    return resp_wanted


# %% CC type models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TODO double check movement to universal_tools in main utils library!!
# def PEM_res(rho0, m, tau, c, f):
#     """
#     Pelton et al. model
#     in terms of resistivity

#     Parameters
#     ----------
#     rho0 : float (Ohmm)
#         DC or low frequency limit of resistivity.
#     m : float ()
#         chargeability (0-1).
#     tau : float (s)
#         relaxation time.
#     c : float
#         dispersion coefficient (0-1).
#     f : np.array (float)
#         frequencies at which the model should be evaluated.

#     Returns
#     -------
#     complex_res : np.array
#         complex resistivity at the given frequencies.

#     """

#     iotc = (2j*np.pi*f * tau)**c
#     complex_res = (rho0 * (1 - m*(1 - 1/(1 + iotc))))

#     return complex_res


# def PEM_fia_con0(sig_0, m, tau_sig, c, f):
#     """
#     from fiandaca et al. (2018)
#     Formula 3 (erroneous sign in manuscript version)

#     Parameters
#     ----------
#     sig_0 : float
#         DC conductivity.
#     m : float (0 - 1)
#         chargeability.
#     tau : float
#         relaxation time (s).
#     c : float (0 - 1)
#         dispersion coefficient.
#     f : float, array-like, or single value
#         frequency(ies) at which the complex conductivity should be calculated.

#     Returns
#     -------
#     complex_con : complex, array-like, or single value, depending on f
#         complex conductivity.

#     """
#     iotc = (2j*np.pi*f * tau_sig)**c
#     complex_con = sig_0 * (1 + (m / (1 - m)) * (1 - (1 / (1 + (iotc))))) # correct version!!
#     # complex_con = sig_0 * (1 - (m / (1 - m)) * (1 - (1 / (1 + (iotc)))))  wrong in paper!!

#     return complex_con



# def cole_cole(inp, p_dict):
#     """Cole and Cole (1941).
#     code from: https://empymod.emsig.xyz/en/stable/examples/time_domain/cole_cole_ip.html#sphx-glr-examples-time-domain-cole-cole-ip-py
#     """

#     # Compute complex conductivity from Cole-Cole
#     iotc = np.outer(2j*np.pi*p_dict['freq'], inp['tau'])**inp['c']
#     condH = inp['cond_8'] + (inp['cond_0']-inp['cond_8']) / (1 + iotc)
#     condV = condH/p_dict['aniso']**2

#     # Add electric permittivity contribution
#     etaH = condH + 1j*p_dict['etaH'].imag
#     etaV = condV + 1j*p_dict['etaV'].imag

#     return etaH, etaV



# def pelton_et_al(inp, p_dict):
#     """ Pelton et al. (1978).
#     code from: https://empymod.emsig.xyz/en/stable/examples/time_domain/cole_cole_ip.html#sphx-glr-examples-time-domain-cole-cole-ip-py
#     """

#     # Compute complex resistivity from Pelton et al.
#     # print('\n   shape: p_dict["freq"]\n', p_dict['freq'].shape)
#     iotc = np.outer(2j*np.pi*p_dict['freq'], inp['tau'])**inp['c']

#     # print('\n   shape: inp["rho_0"]\n', inp["rho_0"].shape)
#     # print('\n   shape: iotc\n', iotc.shape)
#     rhoH = inp['rho_0'] * (1 - inp['m']*(1 - 1/(1 + iotc)))
#     rhoV = rhoH*p_dict['aniso']**2

#     # Add electric permittivity contribution
#     etaH = 1/rhoH + 1j*p_dict['etaH'].imag
#     etaV = 1/rhoV + 1j*p_dict['etaV'].imag

#     return etaH, etaV



# def mpa_model(inp, p_dict):
#     """
#     maximum phase angle model (Fiandaca et al 2018)
#     Formula 8 - 11, appendix A.1 - A.08

#     Parameters
#     ----------
#     inp : dictionary
#         dictionary containing the cole-cole parameters:
#             'rho_0' - DC resistivity
#             'phi_max' - maximum phase angle, peak value of the phase of complex res (rad).
#             'tau_phi' - relaxation time, specific for mpa model, see Formula 10 (s).
#             'c' - dispersion coefficient
#     p_dict : dictionary
#         additional dictionary with empymod specific parameters.

#     Returns
#     -------
#     etaH, etaV : dtype??


#     """
#     # obtain chargeability and tau from mpa model
#     m, tau_rho = get_m_taur_MPA(inp['phi_max'], inp['tau_phi'], inp['c'], verbose=True)

#     iotc = np.outer(2j*np.pi*p_dict['freq'], tau_rho)**inp['c']
#     # Compute complex resistivity
#     rhoH = inp['rho_0'] * (1 - m*(1 - 1/(1 + iotc)))
#     rhoV = rhoH*p_dict['aniso']**2

#     # Add electric permittivity contribution
#     etaH = 1/rhoH + 1j*p_dict['etaH'].imag
#     etaV = 1/rhoV + 1j*p_dict['etaV'].imag

#     return etaH, etaV


# # TODO: Test and finish function!!
# # def cc_eps(inp, p_dict):
# #     """
# #     Mudler et al. (2020) - after Zorin and Ageev, 2017 with HF EM part - dielectric permittivity
# #     """

# #     # Compute complex permittivity
# #     iotc = np.outer(2j*np.pi*p_dict['freq'], inp['tau'])**inp['c']
# #     iwe0rhoDC = np.outer(2j*np.pi*p_dict['freq'], epsilon_0, inp["rho_DC"])
# #     eta_c_r = inp["epsilon_hf"] + ((["epsilon_DC"] - ["epsilon_HF"]) / (1 + iotc)) + (1 / iwe0rhoDC)

# #     etaH = eta_c_r
# #     etaV = eta_c_r

# #     return etaH, etaV


# def cc_con_koz(inp, p_dict):
#     """
#     compl. con from Kozhevnikov, Antonov (2012 - JaGP)
#     using perm0 and perm8 - Formula 5
#     # TODO: Test and finish method!!
#     """

#     # Compute complex permittivity,
#     # print('\n   shape: p_dict["freq"]\n', p_dict['freq'].shape)
#     io = 2j * np.pi * p_dict['freq'] ## i*omega --> from frequency to angular frequency
#     iotc = np.outer(io, inp['tau'])**inp['c']

#     # print('\n   shape: inp["sigma_0"]\n', inp["sigma_0"].shape)
#     # print('\n   shape: iotc\n', iotc.shape)
#     # print('\n   shape: io\n', io.shape)
#     # print('1st term:', (inp["sigma_0"] + np.outer(io, epsilon_0)).shape)
#     # print('2nd term:', (inp["epsilon_8"] + ((inp["epsilon_s"] - inp["epsilon_8"]) / (1 + iotc))).shape)
#     etaH = (inp["sigma_0"] + np.outer(io, epsilon_0) *
#             (inp["epsilon_8"] + ((inp["epsilon_s"] - inp["epsilon_8"]) / (1 + iotc)))
#            )

#     etaV = etaH * p_dict['aniso']**2

#     return etaH, etaV





# # %% functions for MPA
# def CC_MPA(rho_0, phi_max, tau_phi, c, f):
#     """
#     maximum phase angle model (after Fiandaca et al, 2018)
#     Formula 8 - 11, appendix A.1 - A.08

#     Parameters
#     ----------
#     rho_0 : float
#         DC resistivity.
#     phi_max : float
#         maximum phase angle, peak value of the phase of complex res (rad).
#     tau_phi : float
#         relaxation time, specific for mpa model, see Formula 10 (s).
#     c : float (0 - 1)
#         dispersion coefficient.
#     f : float, array-like, or single value
#         frequency(ies) at which the complex conductivity should be calculated.

#     Returns
#     -------
#     complex_res : complex, array-like, or single value, depending on f
#         complex resistivity at the given frequencies.

#     """
#     m, tau_rho = get_m_taur_MPA(phi_max, tau_phi, c, verbose=True)
#     iotc = (2j*np.pi*f * tau_rho)**c
#     complex_res = rho_0 * (1 + (m / ((1 - m)) * (1 - (1 / (1 + (iotc)*(1 - m))))))

#     return complex_res


# def get_m_taur_MPA(phi_max, tau_phi, c, verbose=True):
#     """
#     function to obtain the classical cc params from the mpa ones:
#         uses an iterative approach and stops once the difference between
#         two consecutive m values equals 0
#         (after Fiandaca et al, 2018), appendix A.1 - A.08

#     Parameters
#     ----------
#     phi_max : float
#         maximum phase angle, peak value of the phase of complex res (rad).
#     tau_phi : float
#         relaxation time, specific for mpa model, see Formula 10 (s).
#     c : float (0 - 1)
#         dispersion coefficient.

#     Raises
#     ------
#     ValueError
#         in case the iteration doesn't converge after 100 iters.

#     Returns
#     -------
#     m : float ()
#         chargeability (0-1).
#     tau_rho : float (s)
#         relaxation time.

#     """
#     n_iters = 10000
#     th = 1e-8

#     if (type(phi_max) == type(c)) and (type(tau_phi) == type(c)):
#         if hasattr(phi_max, '__len__') and (not isinstance(phi_max, str)):
#             m, tau_rho = np.zeros_like(phi_max), np.zeros_like(tau_phi)

#             phi_max_s = np.copy(phi_max)

#             if any(phi_max == 0):
#                 logger.info('encountered phi_max == 0, assuming no-IP effect, setting m also to 0')
#                 mask = phi_max == 0
#                 phi_max = phi_max
#             else:
#                 start_id = 0

#             for i, phi_max in enumerate(phi_max_s):
#                 mns = []
#                 tau_rs = []
#                 areal = []
#                 bimag = []
#                 delta_ms = []

#                 for n in range(0, n_iters):
#                     if n == 0:
#                         mns.append(0)
#                         tau_rs.append(mpa_get_tau_rho(m=mns[n],
#                                                       tau_phi=tau_phi[i],
#                                                       c=c[i]))
#                         areal.append(mpa_get_a(tau_rs[n], tau_phi[i], c[i]))
#                         bimag.append(mpa_get_b(tau_rs[n], tau_phi[i], c[i]))
#                         mns.append(mpa_get_m(a=areal[n],
#                                               b=bimag[n],
#                                               phi_max=phi_max[i]))
#                         delta_ms.append(mpa_get_deltam(mn=mns[n+1], mp=mns[n]))
#                     else:
#                         tau_rs.append(mpa_get_tau_rho(m=mns[n],
#                                                       tau_phi=tau_phi[i],
#                                                       c=c[i]))
#                         areal.append(mpa_get_a(tau_rs[n], tau_phi[i], c[i]))
#                         bimag.append(mpa_get_b(tau_rs[n], tau_phi[i], c[i]))
#                         mns.append(mpa_get_m(a=areal[n],
#                                               b=bimag[n],
#                                               phi_max=phi_max[i]))
#                         delta_ms.append(mpa_get_deltam(mn=mns[n+1], mp=mns[n]))
#                         logger.info('delta_m: ', delta_ms[n])
#                         if delta_ms[n] <= th:  # stop if the difference is below th
#                             if verbose:
#                                 logger.info(f'iteration converged after {n} iters')
#                                 logger.info('solved m:', mns[-1])
#                                 logger.info('solved tau_rho:', tau_rs[-1])
#                             m[i] = mns[-1]
#                             tau_rho[i] = tau_rs[-1]
#                             break

#                 #for n in range(0, n_iters):
#                     #if n == 0:
#                         #mns.append(0)
#                         #tau_rs.append(mpa_get_tau_rho(m=mns[n],
#                                                       #tau_phi=tau_phi[i],
#                                                       #c=c[i]))
#                         #areal.append(mpa_get_a(tau_rs[n], tau_phi[i], c[i]))
#                         #bimag.append(mpa_get_b(tau_rs[n], tau_phi[i], c[i]))
#                         #mns.append(mpa_get_m(a=areal[n],
#                                               #b=bimag[n],
#                                               #phi_max=phi_max[i]))
#                         #delta_ms.append(mpa_get_deltam(mn=mns[n+1], mp=mns[n]))
#                     #else:
#                         #tau_rs.append(mpa_get_tau_rho(m=mns[n],
#                                                       #tau_phi=tau_phi[i],
#                                                       #c=c[i]))
#                         #areal.append(mpa_get_a(tau_rs[n], tau_phi[i], c[i]))
#                         #bimag.append(mpa_get_b(tau_rs[n], tau_phi[i], c[i]))
#                         #mns.append(mpa_get_m(a=areal[n],
#                                               #b=bimag[n],
#                                               #phi_max=phi_max[i]))
#                         #delta_ms.append(mpa_get_deltam(mn=mns[n+1], mp=mns[n]))
#                         #logger.info('delta_m: ', delta_ms[n])
#                         #if delta_ms[n] <= th:  # stop if the difference is below th
#                             #if verbose:
#                                 #logger.info(f'iteration converged after {n} iters')
#                                 #logger.info('solved m:', mns[-1])
#                                 #logger.info('solved tau_rho:', tau_rs[-1])
#                             #m[i] = mns[-1]
#                             #tau_rho[i] = tau_rs[-1]
#                             #break

#                 if delta_ms[n] > th:
#                     raise ValueError(f'the iterations did not converge after {n_iters} iterations, please check input!')

#         else:
#             #mns = np.zeros(n_iters)
#             #tau_rs = np.zeros(n_iters)
#             #areal = np.zeros(n_iters)
#             #bimag = np.zeros(n_iters)
#             #delta_ms = np.zeros(n_iters)
            
#             #for n in range(1, n_iters):
#                 #tau_rs[n] = mpa_get_tau_rho(m=mns[n],
#                                             #tau_phi=tau_phi,
#                                             #c=c)
#                 #areal[n] = mpa_get_a(tau_rs[n], tau_phi, c)
#                 #bimag[n] = mpa_get_b(tau_rs[n], tau_phi, c)
#                 #mns[n] = mpa_get_m(a=areal[n], b=bimag[n], phi_max=phi_max)
#                 #delta_ms[n] = mpa_get_deltam(mn=mns[n], mp=mns[n-1])
#                 #logger.info('delta_m: ', delta_ms[n])

#                 #if delta_ms[n] <= th:  # stop if the difference is below 1e-9
#                     #if verbose:
#                         #logger.info(f'iteration converged after {n} iters')
#                         #logger.info('solved m:', mns[n])
#                         #logger.info('solved tau_rho:', tau_rs[n])
#                     #m = mns[n]
#                     #tau_rho = tau_rs[n]
#                     #break

#             mns = []
#             tau_rs = []
#             areal = []
#             bimag = []
#             delta_ms = []

#             for n in range(0, n_iters):
#                 if n == 0:
#                     mns.append(0)
#                     tau_rs.append(mpa_get_tau_rho(m=mns[n],
#                                                 tau_phi=tau_phi,
#                                                 c=c))
#                     areal.append(mpa_get_a(tau_rs[n], tau_phi, c))
#                     bimag.append(mpa_get_b(tau_rs[n], tau_phi, c))
#                     mns.append(mpa_get_m(a=areal[n],
#                                         b=bimag[n],
#                                         phi_max=phi_max))
#                     delta_ms.append(mpa_get_deltam(mn=mns[n+1], mp=mns[n]))
#                 else:
#                     tau_rs.append(mpa_get_tau_rho(m=mns[n],
#                                                 tau_phi=tau_phi,
#                                                 c=c))
#                     areal.append(mpa_get_a(tau_rs[n], tau_phi, c))
#                     bimag.append(mpa_get_b(tau_rs[n], tau_phi, c))
#                     mns.append(mpa_get_m(a=areal[n],
#                                         b=bimag[n],
#                                         phi_max=phi_max))
#                     delta_ms.append(mpa_get_deltam(mn=mns[n+1], mp=mns[n]))
#                     print('delta_m: ', delta_ms[n])
#                     if delta_ms[n] <= th:  # stop if the difference is below 1e-9
#                         if verbose:
#                             print(f'iteration converged after {n} iters')
#                             print('solved m:', mns[-1])
#                             print('solved tau_rho:', tau_rs[-1])
                            
#                         m = mns[-1]
#                         tau_rho = tau_rs[-1]
#                         break

#                 if delta_ms[n] > th:
#                     raise ValueError(f'the iterations did not converge after {n_iters} iterations, please check input!')

#     else:
#         raise TypeError('please make sure that all 3 input params are of the same dtype!!')

#     return m, tau_rho


# def get_tauphi_from_ts(m, tau_sig, c):
#     """
#     after fiandaca et al (2018), Formula 10
#     obtain tau_phi from cole-cole conductivity formula (clasic CC)

#     Parameters
#     ----------
#     m : float ()
#         chargeability (0-1).
#     tau_sig : float (s)
#         relaxation time.
#     c : float (0 - 1)
#         dispersion coefficient.

#     Returns
#     -------
#     tau_phi : float
#         relaxation time, specific for mpa model, see Formula 10 (s).

#     """
#     tau_phi = tau_sig * (1 - m)**(-1/(2*c))
#     return tau_phi


# def get_tauphi_from_tr(m, tau_rho, c):
#     """
#     after fiandaca et al (2018), Formula 10
#     obtain tau_phi from cole-cole resisitvity formula (PEM)

#     Parameters
#     ----------
#     m : float ()
#         chargeability (0-1).
#     tau_rho : float (s)
#         relaxation time.
#     c : float (0 - 1)
#         dispersion coefficient.

#     Returns
#     -------
#     tau_phi : float
#         relaxation time, specific for mpa model, see Formula 10 (s).

#     """
#     tau_phi = tau_rho * (1 - m)**(1/(2*c))
#     return tau_phi


# def get_phimax_from_CCC(sig_0, m, tau_sig, c):
#     """
#     after fiandaca et al (2018), Formula 9
#     obtain phi_max from cole-cole resisitvity formula (PEM)

#     Parameters
#     ----------
#     sig_0 : float
#         DC conductivity.
#     m : float ()
#         chargeability (0-1).
#     tau_sig : float (s)
#         relaxation time.
#     c : float (0 - 1)
#         dispersion coefficient.

#     Returns
#     -------
#     phi_max : float
#         maximum phase angle, peak value of the phase of complex res (rad).

#     """
#     tau_phi = get_tauphi_from_ts(m, tau_sig, c)
#     cmplx_con = PEM_fia_con0(sig_0=sig_0, m=m, tau_sig=tau_sig,
#                              c=c, f=1 / (2*np.pi*tau_phi))
#     phi_max = np.arctan(np.imag(cmplx_con) / np.real(cmplx_con))
#     return phi_max


# def get_phimax_from_CCR(rho_0, m, tau_rho, c):
#     """


#     Parameters
#     ----------
#     rho_0 : float
#         DC resistivity.
#     m : float ()
#         chargeability (0-1).
#     tau_rho : float (s)
#         relaxation time.
#     c : float (0 - 1)
#         dispersion coefficient.

#     Returns
#     -------
#     phi_max : float
#         maximum phase angle, peak value of the phase of complex res (rad).

#     """
#     tau_phi = get_tauphi_from_tr(m, tau_rho, c)
#     cmplx_res = PEM_res(rho0=rho_0, m=m, tau=tau_rho,
#                         c=c, f=1 / (2*np.pi*tau_phi))
#     phi_max = -np.arctan(np.imag(cmplx_res) / np.real(cmplx_res))
#     return phi_max


# def mpa_get_deltam(mn, mp):
#     """
#     after Fiandaca et al. (2018), Appendix A.04

#     Parameters
#     ----------
#     mn : float
#         m of current iteration.
#     mp : TYPE
#         m of previous iteration.

#     Returns
#     -------
#     float
#         delta_m, difference between current and previous m.

#     """
#     return np.abs(mn - mp) / mn


# def mpa_get_tau_rho(m, tau_phi, c):
#     """
#     after Fiandaca et al. (2018), Appendix A.05
#     needs |1 - m|, otherwise values of m > 1 will result in nan!!

#     Parameters
#     ----------
#     m : float ()
#         chargeability (0-1).
#     tau_phi : float
#         relaxation time, specific for mpa model, see Formula 10 (s).
#     c : float (0 - 1)
#         dispersion coefficient.

#     Returns
#     -------
#     tau_rho : float (s)
#         relaxation time.

#     """
#     tau_rho = tau_phi * (abs(1 - m)**(-1/(2*c)))  # abs is essential here
#     return tau_rho


# def mpa_get_a(tau_rho, tau_phi, c):
#     """
#     after Fiandaca et al. (2018), Appendix A.06

#     Parameters
#     ----------
#     tau_rho : float (s)
#         relaxation time.
#     tau_phi : float
#         relaxation time, specific for mpa model, see Formula 10 (s).
#     c : float (0 - 1)
#         dispersion coefficient.

#     Returns
#     -------
#     a : float
#         real part of complex variable.

#     """
#     a = np.real(1 / (1 + (1j*(tau_rho / tau_phi))**c))
#     return a


# def mpa_get_b(tau_rho, tau_phi, c):
#     """
#     after Fiandaca et al. (2018), Appendix A.07

#     Parameters
#     ----------
#     tau_rho : float (s)
#         relaxation time.
#     tau_phi : float
#         relaxation time, specific for mpa model, see Formula 10 (s).
#     c : float (0 - 1)
#         dispersion coefficient.

#     Returns
#     -------
#     b : float
#         imaginary part of complex variable.

#     """
#     b = np.imag(1 / (1 + (1j*(tau_rho / tau_phi))**c))
#     return b


# def mpa_get_m(a, b, phi_max):
#     """
#     after Fiandaca et al. (2018), Appendix A.08

#     Parameters
#     ----------
#     a : float
#         real part of complex variable. see mpa_get_a
#     b : float
#         imaginary part of complex variable. see mpa_get_b
#     phi_max : float
#         maximum phase angle, peak value of the phase of complex res (rad).

#     Returns
#     -------
#     m : float ()
#         chargeability (0-1).

#     """
#     tan_phi = np.tan(-phi_max)
#     m = tan_phi / ((1 - a) * tan_phi + b)
#     return m




# old ramp data
    #if current_key == '1A':
        #if location == 'donauinsel':
            #ramp_data_array = np.array([[  6. ,  1.5 , 1. , 0.15],
                                        #[ 12. ,  3.  , 1. , 0.23],
                                        #[ 25. ,  6.25, 1. , 0.4 ],
                                        #[ 50. , 12.5 , 1. , 0.8 ],
                                        #[100. , 25.  , 1. , 1.3 ]])

        #elif location == 'salzlacken':
            #ramp_data_array = np.array([[  8. ,  2.  , 1.  ,  0.21],
                                        #[ 25. ,  6.25, 1.  ,  0.44],
                                        #[ 50. , 12.5 , 1.  ,  0.8 ],
                                        #[100. , 25.  , 1.  ,  1.3 ]])

        #elif location == 'hengstberg':
            #raise ValueError('1A ramp data not available for hengstberg')

        #elif location == 'sonnblick':
            #raise ValueError('1A ramp data not available for sonnblick')

        #else:
            #raise ValueError('location of ramp data measurements is not available ...')

    #elif current_key == '4A':
        #if location == 'donauinsel':
            #ramp_data_array = np.array([[  6. ,  1.5  , 1. ,  0.17],
                                        #[ 25. ,  6.25 , 1. ,  0.45],
                                        #[ 50. ,  12.5 , 1. ,  0.95],
                                        #[100. ,  25.  , 1. ,  1.5],
                                        #[400. , 100.  , 1. , 10.0]])

        #elif location == 'salzlacken':
            #ramp_data_array = np.array([[  8. ,  2.  , 1. ,  0.21],
                                        #[ 25. ,  6.25, 1. ,  0.5 ],
                                        #[ 50. , 12.5 , 1. ,  0.95],
                                        #[100. , 25.  , 1. ,  1.5 ],
                                        #[400. ,100.  , 1. , 10.0]])

        #elif location == 'hengstberg':
            #ramp_data_array = np.array([[  6. ,  1.5  , 1. ,  0.17],
                                        #[ 25. ,  6.25 , 1. ,  0.45],
                                        #[ 50. ,  12.5 , 1. ,  0.95],
                                        #[100. ,  25.  , 1. ,  1.5]])

        #elif location == 'sonnblick':
            #ramp_data_array = np.array([[100. ,  25.  , 1. ,  2.5],
                                        #[350. ,  75.  , 1. , 10.0]])

        #else:
            #raise ValueError('location of ramp data measurements is not available ...')

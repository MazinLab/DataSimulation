import glob
import inspect
import os
import sys

from astropy.coordinates import SkyCoord
import astropy.units as units
from PyAstronomy import pyasl
from astropy.convolution import Gaussian2DKernel, convolve
import numpy as np

"""
Modules

calc_pos_angle:  
    Calculate positon angle between a primary and secondary source.  
    Input: (['HH_ra_pri:MM_ra_pri:SS_ra_pri HH_dec_pri:MM_dec_pri:SS_dec_pri'], 
            ['HH_ra_sec:MM_ra_sec:SS_ra_sec HH_dec_sec:MM_dec_sec:SS_dec_sec']), 
    EX: (['00:42.5 +41:12:00'], 
         ['00:41.5 +41:12:00'])
    Returns position angle in degrees

calc_diffrac_lim
    Calculates lambda/D in pixels for a specific telescope and instrument.
    Input:  lambda sensitivity range of the instrument lambda_start, lambda_stop in nm
            diameter of telescope in meters
            milliarcsec/pixel or milliarcsec/lenslet
            mode = 'pixels' or 'mas' 
    Returns lambda/D in mas or pixels

diffrac_lim_kernel
    Convolve an input array with a Gaussian PSF whose FWHM is lambda/D in pixels
    Input:  The input 2D image, the diffraction limit (~lambda/D) 
    Returns the input image convolved with the diffraction-limited Gaussian PDF

"""
rad_to_mas = (180.0 / np.pi) * 3600 * 1000
rad_to_deg = (180.0 / np.pi)


def calc_pos_angle(RA_DEC_hms_pri=['00:42.5 +41:12:00'], RA_DEC_hms_sec=['00:41.5 +41:12:00']):
    decimaldeg_pri = SkyCoord(RA_DEC_hms_pri, unit=(units.hourangle, units.deg))
    decimaldeg_sec = SkyCoord(RA_DEC_hms_sec, unit=(units.hourangle, units.deg))
    pos_angle = pyasl.positionAngle(decimaldeg_pri.ra.degree, +decimaldeg_pri.dec.degree, decimaldeg_sec.ra.degree,
                                    +decimaldeg_sec.dec.degree)
    return pos_angle


def calc_diffrac_lim(lambda_start=1100, lambda_stop=1400, diameter=8.3, scale=16.4, mode='pixels'):
    lambda_telescope = np.median([lambda_start, lambda_stop])
    diameter *= 1e9
    diffrac_lim_mas = 1.22 * (lambda_telescope / diameter) * rad_to_mas
    if mode == 'mas':
        diffrac_lim = diffrac_lim_mas
    else:
        diffrac_lim = diffrac_lim_mas / scale
    return diffrac_lim


def diffrac_lim_kernel(input_image=input_image, diffrac_lim=2.68):
    [x_size, y_size] = np.shape(input_image)
    sigma = diffrac_lim / (2. * math.sqrt(2. * math.log(2.)))
    gaussian_2D_kernel = Gaussian2DKernel(sigma, x_size, y_size)
    convolved_kernel = convolve(input_image, gaussian_2D_kernel)
    return convolved_kernel


def PCA_2d(filename, ncomponents):

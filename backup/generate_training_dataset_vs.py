# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:51:51 2019

@author: Chuanzhen Hu
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import spectra_process.subpys as subpys
import time
################################################################################
# define function for creating mixed spectra
def spectra_generator(pures, noise_level):
    # randomize polynomial order
    index = np.random.permutation(np.arange(6))
    # create background
    bg = subpys.myploy(index[0]+1, pures.shape[1])
    bg = bg*np.random.rand(1)*10.
    for k in range(index[0]):
        if np.random.randn(1) <= 0:
            bg[k, :] = np.flip(bg[k, :])
    # generate random coefficients
    rand_coeffs = np.random.rand(1, index[0]+1+pures.shape[0]); # (0, 1)
    f = np.matmul(rand_coeffs, np.concatenate((pures, bg), axis=0))
    ################################
    mixed_spectra = f + noise_level*np.random.rand(1)*np.mean(bg)*np.random.randn(1, pures.shape[1])
    coeff = rand_coeffs[0, :(pures.shape[0])]
    
    return coeff, mixed_spectra
################################################################################
# define function for creating mixed spectra
def spectra_generator_constant(pures, noise_level, pow_val=5):
    # create background
    bg = subpys.myploy(pow_val, pures.shape[1])
    bg = bg*10.
    # generate random coefficients
    rand_coeffs = np.random.rand(1, pow_val+pures.shape[0]); # (0, 1)
    f = np.matmul(rand_coeffs, np.concatenate((pures, bg), axis=0))
    ################################
    mixed_spectra = f + noise_level*np.random.rand(1)*np.mean(bg)*np.random.randn(1, pures.shape[1])
    coeff = rand_coeffs
    
    return coeff, mixed_spectra
################################################################################
def main(unused_argv):
    # load pure spectra
    pures = sio.loadmat('./Spectra_data/pure_spectra.mat')['s']
    wn = sio.loadmat('./Spectra_data/wn.mat')['wn']
    
    # preprocesspure spectra -> subtract minima
    pures = pures - np.transpose(np.tile(np.min(pures, 1), (pures.shape[1], 1)))
    plt.figure(figsize=(10, 6))
    plt.plot(np.transpose(wn), np.transpose(pures))
    plt.title('pure sspectra')
    plt.xlabel('wavenumber (cm-1)')
    plt.ylabel('intensity (a.u.)')
    plt.xlim((np.min(wn), np.max(wn)))
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.transpose(wn), np.transpose(subpys.myploy(6, pures.shape[1])))
    plt.title('polynomial spectra')
    plt.xlabel('wavenumber (cm-1)')
    plt.ylabel('intensity (a.u.)')
    plt.xlim((np.min(wn), np.max(wn)))
    plt.show()
    ############################################################################
    ############################################################################
    random_index = np.int(1e6)
    noise_level = 0.5 #  0.5
    pow_val = 5
    version_choice = 1 # 1: random polynomial bgs; 2: constant polynomial bgs
    ############################################################################
    if version_choice == 1:
        coeff = np.zeros((random_index, pures.shape[0]))
        mixed_spectra = np.zeros((pures.shape[1], random_index))
        # create mixed spectra
        time_start = time.time()
        for ij in range(random_index):
            coeff[ij, :], mixed_spectra[:, ij] = spectra_generator(pures, noise_level)
        time_end = time.time()
    elif version_choice == 2:
        coeff = np.zeros((random_index, pures.shape[0]+pow_val))
        mixed_spectra = np.zeros((pures.shape[1], random_index))
        # create mixed spectra
        time_start = time.time()
        for ij in range(random_index):
            coeff[ij, :], mixed_spectra[:, ij] = spectra_generator_constant(pures, noise_level, pow_val)
        time_end = time.time()
    ############################################################################
    ############################################################################
    plt.figure(figsize=(10, 6))
    plt.plot(np.transpose(wn), mixed_spectra[:, :100])
    plt.title('polynomial sspectra')
    plt.xlabel('wavenumber (cm-1)')
    plt.ylabel('intensity (a.u.)')
    plt.xlim((np.min(wn), np.max(wn)))
    plt.show()
    
    np.save('./Spectra_data/X_train.npy', np.transpose(mixed_spectra))
    np.save('./Spectra_data/Y_train.npy', coeff)
    np.save('./Spectra_data/wn.npy', wn)
    print('Totally cost:', time_end-time_start, 's')
    
if __name__ == "__main__":
    main(0)
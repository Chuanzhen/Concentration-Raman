# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:51:51 2019

@author: Chuanzhen Hu
"""

import numpy as np
import matplotlib.pyplot as plt
import spectra_process.subpys as subpys
import time

################################################################################
# define function for creating mixed spectra
def spectra_generator(pures, noise_level, pow_val=5, bg_enhance=1.0):
    # randomize polynomial order
    index = np.random.permutation(np.arange(pow_val))
    # create background
    bg = subpys.myploy(index[0]+1, pures.shape[1])
    bg = bg*bg_enhance
    for k in range(bg.shape[0]):
        if np.random.randn(1) <= 0:
            bg[k, :] = np.flip(bg[k, :])
    # generate random coefficients
    rand_coeffs = np.random.rand(1, bg.shape[0]+pures.shape[0]); # (0, 1)
    ################################
    f = np.matmul(rand_coeffs, np.concatenate((pures, bg), axis=0))
    ################################
    mixed_spectra = f + noise_level*(np.random.rand(1)*np.mean(bg))*np.random.randn(1, pures.shape[1])
    coeff = rand_coeffs[0, :(pures.shape[0])]
    
    return coeff, mixed_spectra
################################################################################
def main(unused_argv):
    # load pure spectra
    data = np.load('./Spectra_data/my_measurement.npz')
    met = data['methanol']
    eth = data['ethanol']
    ace = data['acetonitrile']
    qtz = data['quartz']
    wn = data['wn'][:, 0]
    names = data['pure_names']

    exp_ch = -1
    pures = np.reshape(np.concatenate((met[exp_ch, :], eth[exp_ch, :], 
                                       ace[exp_ch, :], qtz[exp_ch, :])), (4, wn.shape[0]))
    pures = pures[[0, 1, 3], :]
    ############################################################################
    ############################################################################
    random_index = np.int(1e6)
    noise_level = 0.2#  0.2
    bg_enhance = 10.
    pow_val = 5 # 5, 2
    ############################################################################
    plt.figure(figsize=(10, 6))
    plt.plot(wn, np.transpose(pures[:-1, :]))
    plt.title('pure spectra', fontsize=15)
    plt.xlabel('wavenumber (cm-1)', fontsize=15)
    plt.ylabel('intensity (a.u.)', fontsize=15)
    plt.xlim((np.min(wn), np.max(wn)))
    plt.ylim((np.min(pures), np.max(pures)))
    plt.legend(names[:pures.shape[0]-1], loc=1)
    plt.savefig('./output/generated/pure_spectra.png', dpi=200)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    if noise_level == 0:
        plt.plot(wn, np.transpose(subpys.myploy(pow_val, pures.shape[1])*bg_enhance))
    else:
        plt.plot(wn, np.transpose(subpys.myploy(pow_val, pures.shape[1])*bg_enhance*noise_level))
    plt.title('polynomial spectra', fontsize=15)
    plt.xlabel('wavenumber (cm-1)', fontsize=15)
    plt.ylabel('intensity (a.u.)', fontsize=15)
    plt.xlim((np.min(wn), np.max(wn)))
    plt.savefig('./output/generated/polynomial_background.png', dpi=200)
    plt.show()
    ############################################################################
    coeff = np.zeros((random_index, pures.shape[0]))
    mixed_spectra = np.zeros((pures.shape[1], random_index))
    # create mixed spectra
    time_start = time.time()
    for ij in range(random_index):
        coeff[ij, :], mixed_spectra[:, ij] = spectra_generator(pures, noise_level, pow_val, bg_enhance)
    time_end = time.time()
    ############################################################################
    plt.figure(figsize=(10, 6))
    plt.plot(np.transpose(wn), mixed_spectra[:, :50], linewidth=0.5)
    plt.title('simulated mixed spectra', fontsize=15)
    plt.xlabel('wavenumber (cm-1)', fontsize=15)
    plt.ylabel('intensity (a.u.)', fontsize=15)
    plt.xlim((np.min(wn), np.max(wn)))
    plt.savefig('./output/generated/generated_mixed_spectra.png', dpi=200)
    plt.show()
    
    np.save('./spectra_data/X_train.npy', np.transpose(mixed_spectra))
    np.save('./spectra_data/Y_train.npy', coeff)
    print('Totally cost:', time_end-time_start, 's')
    print('Minima of mixture: %f'%np.min(mixed_spectra))

if __name__ == "__main__":
    main(0)
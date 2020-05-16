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
def spectra_generator(pures, noise_level, pow_val=5, poly_enhance=1.0):
    ################################################################
    # generate random spectra coefficients, range from 0 to 1
    rand_coeffs = np.random.rand(1, pures.shape[0])*poly_enhance
    # create simulated mixed spectra
    f = np.matmul(rand_coeffs, pures)
    ###############################################################
    if pow_val != 0:
        # randomize polynomial order
        index = np.random.permutation(np.arange(pow_val))
        # create background
        base = subpys.myploy(index[0]+1, pures.shape[1])
        base = base*poly_enhance
        for k in range(base.shape[0]):
            if np.random.randn(1) <= 0:
                base[k, :] = np.flip(base[k, :])
        ################################
        # create baseline
        base_coeffs = np.random.rand(1, base.shape[0])*2 - 1
        baseline = np.matmul(base_coeffs, base)
        ################################
        #base_concat = np.concatenate((pures, baseline), axis=0)
        ################################
        f = f + baseline       
    ###############################################################   
    qtz_bg = np.mean(pures[-1, :])*(1 - rand_coeffs[0, -1])
    nosie = noise_level*qtz_bg*np.random.randn(1, pures.shape[1])
    ################################
    mixed_spectra = f + nosie
    ################################
    coeff = rand_coeffs[0, :(pures.shape[0])]
#    coeff = coeff/np.sum(coeff) # normalization
    
    return coeff, mixed_spectra
################################################################################
def main(unused_argv):
    # load pure spectra
    data = np.load('./Spectra_data/my_measurement.npz')
    met = data['methanol_nobg']
    eth = data['ethanol_nobg']
    ace = data['acetonitrile_nobg']
    qtz = data['qtz_ss']
    wn = data['wn'][:, 0]
    names = data['pure_names']
    
    pures = np.reshape(np.concatenate((met, eth, ace, qtz)), (4, wn.shape[0]))
    pures = pures[[0, 1, 3], :]
    ############################################################################
    random_index = np.int(5e5)
    poly_enhance = 1.0 #10, 5
    noise_level = 0.02#  0.01
    pow_val = 6 # 6, 2
    # pures = pures/10
    ############################################################################
    fontsize_val = 20
    plt.figure(figsize=(10, 6))
    plt.plot(wn, np.transpose(pures[:, :]))
    plt.title('pure spectra', fontsize=fontsize_val)
    plt.xlabel('wavenumber (cm-1)', fontsize=fontsize_val)
    plt.ylabel('intensity (a.u.)', fontsize=fontsize_val)
    plt.xlim((np.min(wn), np.max(wn)))
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.ylim((np.min(pures), np.max(pures)))
    plt.legend(names[:pures.shape[0]-1], loc=1, fontsize=fontsize_val)
    plt.savefig('./output/generated/pure_spectra.png', dpi=200)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    if noise_level == 0:
        plt.plot(wn, np.transpose(subpys.myploy(pow_val, pures.shape[1])*poly_enhance))
    else:
        plt.plot(wn, np.transpose(subpys.myploy(pow_val, pures.shape[1])*poly_enhance))
    plt.title('polynomial backgrounds', fontsize=fontsize_val)
    plt.xlabel('wavenumber (cm-1)', fontsize=fontsize_val)
    plt.ylabel('intensity (a.u.)', fontsize=fontsize_val)
    plt.xlim((np.min(wn), np.max(wn)))
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.savefig('./output/generated/polynomial_background.png', dpi=200)
    plt.show()
    ############################################################################
    coeff = np.zeros((random_index, pures.shape[0]))
    mixed_spectra = np.zeros((pures.shape[1], random_index))
    # create mixed spectra
    time_start = time.time()
    for ij in range(random_index):
        coeff[ij, :], mixed_spectra[:, ij] = spectra_generator(pures, noise_level, pow_val, poly_enhance)
    time_end = time.time()
    ############################################################################
    mixture_minima = np.min(np.mean(mixed_spectra, axis=0))
#    mixed_spectra = mixed_spectra - np.min(mixed_spectra)
    ############################################################################
    plt.figure(figsize=(10, 6))
    plt.plot(np.transpose(wn), mixed_spectra[:, :100], linewidth=0.5)
    plt.title('simulated mixed spectra', fontsize=fontsize_val)
    plt.xlabel('wavenumber (cm-1)', fontsize=fontsize_val)
    plt.ylabel('intensity (a.u.)', fontsize=fontsize_val)
    plt.xlim((np.min(wn), np.max(wn)))
    #plt.ylim((0, 100))
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.savefig('./output/generated/generated_mixed_spectra.png', dpi=200)
    plt.show()
    
    np.save('./spectra_data/X_train.npy', np.transpose(mixed_spectra))
    np.save('./spectra_data/Y_train.npy', coeff[:, :(pures.shape[0]-1)])
    print('totally cost:', time_end-time_start, 's')
    print('minimized mean of mixture:', mixture_minima)
    print('minima of mixture:', np.min(mixed_spectra))
    print('maxima of mixture:', np.max(mixed_spectra))
    print('minima of peak height:', np.min(np.max(mixed_spectra, axis=0) - np.min(mixed_spectra, axis=0)))
    print('noise base:', np.mean(pures[-1, :])*noise_level*poly_enhance)

if __name__ == "__main__":
    main(0)
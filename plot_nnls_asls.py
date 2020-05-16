# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:55:49 2019

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import spectra_process.subpys as subpys
import scipy.optimize as optimize

# In[] load pure spectra
data = np.load('./Spectra_data/my_measurement.npz')
met = data['methanol']
eth = data['ethanol']
ace = data['acetonitrile']
wn = data['wn'][:, 0]
names = data['pure_names'][:3]
qtz = data['quartz']

met = met/np.max(met)
eth = eth/np.max(eth)
ace = ace/np.max(ace)
qtz = qtz/np.max(qtz)
pures = np.reshape(np.concatenate((met, eth, ace)), (3, wn.shape[0]))

plt.figure(figsize=(10, 6))
plt.plot(wn, met.T, 'r',
         wn, eth.T, 'g',
         wn, ace.T, 'b', linewidth=1.5)
plt.grid()
plt.legend(names, loc=1, fontsize=20)
plt.xlim(np.min(wn), np.max(wn))
plt.ylim(np.min(pures), np.max(pures))
plt.xlabel('wavenumber (cm-1)', fontsize=20)
plt.ylabel('intensity (a.u.)', fontsize=20)
plt.setp(plt.gca().get_xticklabels(), fontsize=15)
plt.setp(plt.gca().get_yticklabels(), fontsize=15)
plt.savefig('C:/Users/admin/Desktop/1.png', dpi=200)
plt.show()

# In[] generate random coeffs
coeffs = np.random.random((pures.shape[0], 100))
mixed = np.matmul(coeffs.T, pures)

plt.figure(figsize=(10, 6))
plt.plot(wn, mixed.T, linewidth=0.4)
plt.grid()
plt.xlim(np.min(wn), np.max(wn))
plt.ylim(np.min(mixed), np.max(mixed))
plt.xlabel('wavenumber (cm-1)', fontsize=20)
plt.ylabel('intensity (a.u.)', fontsize=20)
plt.setp(plt.gca().get_xticklabels(), fontsize=15)
plt.setp(plt.gca().get_yticklabels(), fontsize=15)
plt.savefig('C:/Users/admin/Desktop/2.png', dpi=200)
plt.show()

# In[] nnls
#poly_pures = np.concatenate((pures[:2, :], subpys.myploy(1, pures.shape[1])))
poly_pures = pures[:2, :]
nnls_coeffs = np.zeros(coeffs.shape)
asls_coeffs = np.zeros(coeffs.shape)
for ij in range(mixed.shape[0]):
    tmpCoeff = subpys.asls(poly_pures, np.reshape(mixed[ij, :], (1, len(wn))), 0.01)
    asls_coeffs[:2, ij] = tmpCoeff[:2]
    [tmpCoeff, residual] = optimize.nnls(poly_pures[:2, :].T, mixed[ij, :])
    nnls_coeffs[:2, ij] = tmpCoeff[:2]
# In[]
fontsize_val = 25
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.plot(coeffs[0, :], nnls_coeffs[0, :], 'ob',
         np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
plt.ylabel('concentrations by NNLS', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.title(names[0], fontsize=fontsize_val)
plt.subplot(222)
plt.plot(coeffs[1, :], nnls_coeffs[1, :], 'ob',
         np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
#    plt.xlabel('true concentrations', fontsize=fontsize_val)
#    plt.ylabel('concentrations by CNN', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.title(names[1], fontsize=fontsize_val)

plt.subplot(223)
plt.plot(coeffs[0, :], asls_coeffs[0, :], 'ob',
         np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
plt.xlabel('true concentrations', fontsize=fontsize_val)
plt.ylabel('concentrations by AsLS', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.subplot(224)
plt.plot(coeffs[1, :], asls_coeffs[1, :], 'ob',
         np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
plt.xlabel('true concentrations', fontsize=fontsize_val)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)

plt.savefig('C:/Users/admin/Desktop/3.png', dpi=200)
plt.show()
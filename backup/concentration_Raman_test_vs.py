# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:35:28 2019

@author: Chuanzhen Hu
"""
from tensorflow import keras
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import spectra_process.subpys as subpys
from sklearn import preprocessing
# In[] read pure_spectra and mixture, then smoothing
mixture = sio.loadmat('./Spectra_data/mixture.mat')['mx']
pures = sio.loadmat('./Spectra_data/pure_spectra.mat')['s']
wn = sio.loadmat('./Spectra_data/wn.mat')['wn']

#mixture = mixture - np.transpose(np.tile(np.transpose(np.min(mixture, axis=1)), (mixture.shape[1], 1)))
mss = np.transpose(subpys.whittaker_smooth(spectra=np.transpose(mixture), lmbda=0.5, d=3))

show_choice = 500
plt.figure(figsize=(10, 6))
plt.plot(np.transpose(wn), np.transpose(mixture[show_choice, :]),
         np.transpose(wn), np.transpose(mss[show_choice, :]))
plt.title('smoothed spectra')
plt.xlabel('wavenumber (cm-1)')
plt.ylabel('intensity (a.u.)')
plt.xlim((np.min(wn), np.max(wn)))
plt.show()
# In[] normalize
#mixture = mss
X_mean = np.load('./Spectra_data/X_scale_mean.npy')
X_std = np.load('./Spectra_data/X_scale_std.npy')
X = (mixture - X_mean)/X_std

#X_scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(mixture)
#X = X_scaler.transform(mixture)
# In[] load trained model and predict
model = keras.models.load_model('./RamanNet/regression_model.h5')
YPredict = model.predict(np.reshape(X, (X.shape[0], X.shape[1], 1)))
recovered = np.matmul(YPredict, pures)
# In[] least square fitting
ls_mss = mss-np.transpose(np.tile(np.transpose(np.min(mss, axis=1)), (mss.shape[1], 1)))
ls_coeffs = np.zeros(YPredict.shape)
poly_pures = np.concatenate((pures, subpys.myploy(2, pures.shape[1])))
#poly_pures = pures
for ij in range(ls_coeffs.shape[0]):
#    tmpCoeff = subpys.asls(poly_pures, ls_mss[ij, :], 0.01)
    [tmpCoeff, resnorm, residual] = subpys.lsqnonneg(poly_pures.T, ls_mss[ij, :])
    ls_coeffs[ij, :] = tmpCoeff[:pures.shape[0]]
ls_recovred = np.matmul(ls_coeffs, pures)
# In[] plot pre-processing
recovered_norm = recovered/np.transpose(np.tile(np.max(np.abs(recovered), axis=1), (mixture.shape[1], 1)))
YPredict_norm = YPredict/np.transpose(np.tile(np.sum(YPredict, axis=1), (YPredict.shape[1], 1)))
mixture_norm = mixture/np.transpose(np.tile(np.max(np.abs(mixture), axis=1), (mixture.shape[1], 1)))

ls_recovered_norm = ls_recovred/np.transpose(np.tile(np.max(np.abs(ls_recovred), axis=1), (ls_recovred.shape[1], 1)))
ls_coeffs_norm = ls_coeffs/np.transpose(np.tile(np.sum(ls_coeffs, axis=1), (ls_coeffs.shape[1], 1)))
# In[] plot result of CNN
plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.plot(np.transpose(np.arange(YPredict.shape[0])), YPredict)
plt.title('concentrations', fontsize=12)
plt.xlabel('measurements', fontsize=12)
plt.ylabel('coefficients', fontsize=12)
plt.xlim((0, YPredict.shape[0]-1))
plt.ylim((0, 1))

plt.subplot(223)
plt.plot(np.transpose(np.arange(YPredict_norm.shape[0])), YPredict_norm)
plt.title('relative concentrations', fontsize=12)
plt.xlabel('measurements', fontsize=12)
plt.ylabel('coefficients', fontsize=12)
plt.xlim((0, YPredict_norm.shape[0]-1))
plt.ylim((0.3, 0.7))

plt.subplot(222)
plt.plot(np.transpose(wn), np.transpose(recovered),
         np.transpose(wn), -np.transpose(mixture), linewidth=0.2)
plt.title('recovered spectra and raw spectra', fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.xlim((np.min(wn), np.max(wn)))

plt.subplot(224)
plt.plot(np.transpose(wn), np.transpose(recovered_norm),
         np.transpose(wn), -np.transpose(mixture_norm), linewidth=0.2)
plt.title('normalized recovered spectra and raw spectra', fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.xlim((np.min(wn), np.max(wn)))

plt.savefig('./RamanNet/output_cnn.png', dpi=300)
plt.show()
# In[] plot result of LS
plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.plot(np.transpose(np.arange(ls_coeffs.shape[0])), ls_coeffs)
plt.title('concentrations', fontsize=12)
plt.xlabel('measurements', fontsize=12)
plt.ylabel('coefficients', fontsize=12)
plt.xlim((0, ls_coeffs.shape[0]-1))
plt.ylim((0, 1))

plt.subplot(223)
plt.plot(np.transpose(np.arange(ls_coeffs_norm.shape[0])), ls_coeffs_norm)
plt.title('relative concentrations', fontsize=12)
plt.xlabel('measurements', fontsize=12)
plt.ylabel('coefficients', fontsize=12)
plt.xlim((0, ls_coeffs_norm.shape[0]-1))
plt.ylim((0.3, 0.7))

plt.subplot(222)
plt.plot(np.transpose(wn), np.transpose(ls_recovred),
         np.transpose(wn), -np.transpose(mixture), linewidth=0.2)
plt.title('recovered spectra and raw spectra', fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.xlim((np.min(wn), np.max(wn)))

plt.subplot(224)
plt.plot(np.transpose(wn), np.transpose(ls_recovered_norm),
         np.transpose(wn), -np.transpose(mixture_norm), linewidth=0.2)
plt.title('normalized recovered spectra and raw spectra', fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.xlim((np.min(wn), np.max(wn)))

plt.savefig('./RamanNet/output_ls.png', dpi=300)
plt.show()
# In[]
m_choice = np.arange(10)+0
m_part = np.mean(mss[m_choice, :], axis=0)
part = (m_part - X_mean)/X_std
m_y = model.predict(np.reshape(part, (1, part.shape[0], 1)))
m_recovered = np.matmul(m_y, pures)

#tmpCoeff = subpys.asls(poly_pures, m_part, 0.1)
[tmpCoeff, resnorm, residual] = subpys.lsqnonneg(poly_pures.T, m_part)
m_ls = np.matmul(tmpCoeff[:pures.shape[0]], pures)

plt.figure(figsize=(10, 6))
plt.plot(np.transpose(wn), np.transpose(m_recovered)+4, 'r',
         np.transpose(wn), np.transpose(m_part), 'g',
         np.transpose(wn), -m_ls-5, 'r',
         np.transpose(wn), -np.transpose(m_part), 'g', linewidth=0.5)
plt.title('recovered spectra and raw spectra', fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.xlim((np.min(wn), np.max(wn)))
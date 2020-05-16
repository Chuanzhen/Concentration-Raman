# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:35:28 2019

@author: Chuanzhen Hu
"""
from tensorflow import keras
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import spectra_process.subpys as subpys
import scipy.optimize as optimize
import os
import time

# Set default decvice: GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# In[] read pure_spectra and mixture, then smoothing
data = np.load('./spectra_data/my_measurement.npz')
met = data['methanol_nobg']
eth = data['ethanol_nobg']
ace = data['acetonitrile_nobg']
qtz = data['qtz_ss']
wn = data['wn'][:, 0]
names = data['pure_names']
exp = data['exposure']
dark_std = data['dark_std']

real_binary_relative_coeffs = np.array([[1.0,  5, 10, 20, 30, 40, 50, 50, 50, 50, 50, 50, 50],
                                        [ 50, 50, 50, 50, 50, 50, 50, 40, 30, 20, 10,  5,  1]])

real_ternary_relative_coeffs = np.array([[1.0, 10, 20, 30, 40, 50, 30], 
                                         [ 50, 50, 50, 50, 50, 50, 30],
                                         [ 50, 40, 30, 20, 10,  1, 30]])
is_sum_operation = True
if is_sum_operation:
    # repeats: 10 -> exposures: 10 -> groups: 13
    tmp = data['me_mixture']
    me_mixture = np.zeros((tmp.shape[2]*tmp.shape[3], tmp.shape[1]))
    count = 0
    for i in range(tmp.shape[3]):
        for j in range(tmp.shape[2]):
            me_mixture[count, :] = np.mean(tmp[:, :, j, i], axis=0)
            count += 1
    # repeats: 10 -> exposures: 11 -> groups: 7
    tmp = data['mea_mixture']
    mea_mixture = np.zeros((tmp.shape[2]*tmp.shape[3], tmp.shape[1]))
    count = 0
    for i in range(tmp.shape[3]):
        for j in range(tmp.shape[2]):
            mea_mixture[count, :] = np.mean(tmp[:, :, j, i], axis=0)
            count += 1
else:
    # repeats: 10 -> exposures: 11 -> groups: 13
    tmp = data['me_mixture']
    me_mixture = np.zeros((tmp.shape[0]*tmp.shape[2]*tmp.shape[3], tmp.shape[1]))
    count = 0
    for i in range(tmp.shape[3]):
        for j in range(tmp.shape[2]):
            for k in range(tmp.shape[0]):
                me_mixture[count, :] = tmp[k, :, j, i]
                count += 1
    # repeats: 10 -> exposures: 11 -> groups: 7
    tmp = data['mea_mixture']
    mea_mixture = np.zeros((tmp.shape[0]*tmp.shape[2]*tmp.shape[3], tmp.shape[1]))
    count = 0
    for i in range(tmp.shape[3]):
        for j in range(tmp.shape[2]):
            for k in range(tmp.shape[0]):
                mea_mixture[count, :] = tmp[k, :, j, i]
                count += 1
#  make pure array
exp_ch = -1
pures = np.reshape(np.concatenate((met[exp_ch, :], eth[exp_ch, :], 
                                   ace[exp_ch, :])), (3, wn.shape[0]))
pures = pures[[0, 1], :]
# In[] CNN preprocessing
saved_net_path = '20200420_01'
X_mean = np.load('./RamanNet/'+saved_net_path+'/X_scale_mean.npy')
X_std = np.load('./RamanNet/'+saved_net_path+'/X_scale_std.npy')
model = keras.models.load_model('./RamanNet/'+saved_net_path+'/regression_model.h5')
################################################################################
# exp = 0.001s, 0.005s, 0.01s, 0.05s, 0.1s, 0.3s, 0.5s, 0.7s, 1s, 5s
mixture = me_mixture[np.arange(5, me_mixture.shape[0], 10), :]

#mixture = mea_mixture[np.arange(9, mea_mixture.shape[0], 10), :]

#mixture = np.concatenate((me_mixture, mea_mixture))

real_coeffs = np.transpose(real_binary_relative_coeffs)
#real_coeffs = np.transpose(real_ternary_relative_coeffs)

# In[] smoothing and display test spectra
current_time = time.time()
mss = np.transpose(subpys.whittaker_smooth(spectra=np.transpose(mixture), lmbda=0.5, d=2))
asls_smooth = time.time() - current_time

show_choice = -2
plt.figure(figsize=(10, 6))
plt.plot(np.transpose(wn), np.transpose(mixture[show_choice, :]),
         np.transpose(wn), np.transpose(mss[show_choice, :]))
plt.title('smoothed spectra')
plt.xlabel('wavenumber (cm-1)')
plt.ylabel('intensity (a.u.)')
plt.xlim((np.min(wn), np.max(wn)))
plt.show()

# In[]  normalize
current_time = time.time()
#mixture =  mixture - np.transpose(np.tile(np.min(mixture, axis=1), (mixture.shape[1], 1))) #+ 25
X = (mixture - X_mean)/X_std
cnn_normalize = time.time() - current_time
# In[] load trained model and predict
current_time = time.time()
YPredict = model.predict(np.reshape(X, (X.shape[0], X.shape[1], 1)))
cnn_predict = time.time() - current_time
# In[] LS to remove backgrounds
current_time = time.time()
no_bg_mss = np.zeros(mss.shape)
ls_coeffs = np.zeros((YPredict.shape[0], pures.shape[0]))
poly_bg = np.concatenate((qtz, subpys.myploy(3, pures.shape[1])))
#poly_pures = pures
for ij in range(ls_coeffs.shape[0]):
    tmpCoeff = subpys.asls(poly_bg, np.reshape(mss[ij, :], (1, len(wn))), 0.01)
    no_bg_mss[ij, :] = mss[ij, :] - np.matmul(tmpCoeff, poly_bg)
asls_bg_cor = time.time() - current_time
# In[] LS fitting to get concentrations
current_time = time.time()
ls_coeffs = np.zeros((YPredict.shape[0], pures.shape[0]))
poly_pures = np.concatenate((pures, subpys.myploy(3, pures.shape[1])))
#poly_pures = pures
for ij in range(ls_coeffs.shape[0]):
    tmpCoeff = subpys.asls(poly_pures, np.reshape(no_bg_mss[ij, :], (1, len(wn))), 0.1)
#    [tmpCoeff, residual] = optimize.nnls(poly_pures.T, no_bg_mss[ij, :])
#    [tmpCoeff, resnorm, residual] = subpys.lsqnonneg(poly_pures.T, no_bg_mss[ij, :])
    ls_coeffs[ij, :] = tmpCoeff[:pures.shape[0]]
asls_predict = time.time() - current_time
# In[] plot pre-processing
YPredict[YPredict<0] = 0
ls_coeffs[ls_coeffs<0] = 0

recovered = np.matmul(YPredict[:, :2], pures[:2, :])
ls_recovered = np.matmul(ls_coeffs[:, :2], pures[:2, :])

cnn_mse = np.mean((recovered-no_bg_mss)**2, axis=1)
ls_mse = np.mean((ls_recovered-no_bg_mss)**2, axis=1)

sub_YPredict = YPredict[:, :2]
sub_ls_coeffs = ls_coeffs[:, :2]
sub_real_coeffs = real_coeffs[:, :2]

mixture_norm = no_bg_mss/np.max(no_bg_mss)

recovered_norm = recovered/np.max(recovered)
YPredict_norm = sub_YPredict/np.transpose(np.tile(np.sum(sub_YPredict, axis=1), (sub_YPredict.shape[1], 1)))

ls_recovered_norm = ls_recovered/np.max(ls_recovered)
ls_coeffs_norm = sub_ls_coeffs/np.transpose(np.tile(np.sum(sub_ls_coeffs, axis=1), (sub_ls_coeffs.shape[1], 1)))

sub_real_coeffs = sub_real_coeffs/np.transpose(np.tile(np.sum(sub_real_coeffs, axis=1), (sub_real_coeffs.shape[1], 1)))
sub_real_coeffs_norm = np.zeros(YPredict_norm.shape)
for ij in range(sub_real_coeffs_norm.shape[1]):
    sub_real_coeffs_norm[:, ij] = np.squeeze(np.reshape(np.transpose(np.tile(sub_real_coeffs[:, ij], (np.int32(sub_YPredict.shape[0]/sub_real_coeffs.shape[0]), 1))), (sub_real_coeffs_norm.shape[0], 1)))

cnn_coeff_mse = np.reshape(np.mean((YPredict_norm - sub_real_coeffs_norm)**2, axis=1), (YPredict_norm.shape[0], 1))
ls_coeff_mse = np.reshape(np.mean((ls_coeffs_norm - sub_real_coeffs_norm)**2, axis=1), (YPredict_norm.shape[0], 1))

fontsize_val = 25
# In[] plot result of CNN
plt.figure(figsize=(20, 13))
plt.subplot(221)
plt.plot(np.transpose(np.arange(sub_YPredict.shape[0]))+1, sub_YPredict)
plt.title('concentrations predicted by CNN', fontsize=fontsize_val)
plt.xlabel('measurements', fontsize=fontsize_val)
plt.ylabel('coefficients', fontsize=fontsize_val)
plt.legend(names, loc=1, fontsize=fontsize_val)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.xlim((1, YPredict.shape[0]))
#plt.ylim((0, 1))

plt.subplot(223)
plt.plot(np.transpose(np.arange(YPredict_norm.shape[0]))+1, YPredict_norm[:, :])
plt.title('relative concentrations by CNN', fontsize=fontsize_val)
plt.xlabel('measurements', fontsize=fontsize_val)
plt.ylabel('coefficients', fontsize=fontsize_val)
plt.xlim((1, YPredict_norm.shape[0]))
plt.legend(names, loc=1, fontsize=fontsize_val)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.ylim((0, 1))

plt.subplot(222)
plt.plot(wn, np.transpose(recovered),
         wn, -np.transpose(no_bg_mss), linewidth=0.5)
plt.title('recovered spectra and raw spectra without baseline', fontsize=fontsize_val)
plt.xlabel('wavenumber (cm-1)', fontsize=fontsize_val)
plt.ylabel('intensity (a.u.)', fontsize=fontsize_val)
plt.xlim((np.min(wn), np.max(wn)))
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.ylim((-np.max([recovered, no_bg_mss]), np.max([recovered, no_bg_mss])))

plt.subplot(224)
plt.plot(wn, np.transpose(recovered_norm),
         wn, -np.transpose(mixture_norm), linewidth=0.5)
plt.title('normalized recovered spectra and raw spectra without baseline', fontsize=fontsize_val)
plt.xlabel('wavenumber (cm-1)', fontsize=fontsize_val)
plt.ylabel('intensity (a.u.)', fontsize=fontsize_val)
plt.xlim((np.min(wn), np.max(wn)))
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)

plt.savefig('./output/prediction/output_cnn.png', dpi=200)
plt.show()
# In[] plot result of LS
plt.figure(figsize=(20, 13))
plt.subplot(221)
plt.plot(np.transpose(np.arange(sub_ls_coeffs.shape[0]))+1, sub_ls_coeffs)
plt.title('concentrations predicted by AsLS', fontsize=fontsize_val)
plt.xlabel('measurements', fontsize=fontsize_val)
plt.ylabel('coefficients', fontsize=fontsize_val)
plt.xlim((1, ls_coeffs.shape[0]))
plt.legend(names, loc=1, fontsize=fontsize_val)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
#plt.ylim((0, 1))

plt.subplot(223)
plt.plot(np.transpose(np.arange(ls_coeffs_norm.shape[0]))+1, ls_coeffs_norm[:, :])
plt.title('relative concentrations by AsLS', fontsize=fontsize_val)
plt.xlabel('measurements', fontsize=fontsize_val)
plt.ylabel('coefficients', fontsize=fontsize_val)
plt.xlim((1, ls_coeffs_norm.shape[0]))
plt.legend(names, loc=1, fontsize=fontsize_val)
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
plt.ylim((0, 1))

plt.subplot(222)
plt.plot(wn, np.transpose(ls_recovered),
         wn, -np.transpose(no_bg_mss), linewidth=0.5)
plt.title('recovered spectra and raw spectra without baseline', fontsize=fontsize_val)
plt.xlabel('wavenumber (cm-1)', fontsize=fontsize_val)
plt.ylabel('intensity (a.u.)', fontsize=fontsize_val)
plt.xlim((np.min(wn), np.max(wn)))
plt.ylim((-np.max([ls_recovered, no_bg_mss]), np.max([ls_recovered, no_bg_mss])))
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)

plt.subplot(224)
plt.plot(wn, np.transpose(ls_recovered_norm),
         wn, -np.transpose(mixture_norm), linewidth=0.5)
plt.title('normalized recovered spectra and raw spectra without baseline', fontsize=fontsize_val)
plt.xlabel('wavenumber (cm-1)', fontsize=fontsize_val)
plt.ylabel('intensity (a.u.)', fontsize=fontsize_val)
plt.xlim((np.min(wn), np.max(wn)))
plt.grid()
plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)

plt.savefig('./output/prediction/output_ls.png', dpi=200)
plt.show()
# In[] 10*11*13 = 143*10 = 1430; 10*11*7 = 77*10 = 770;
if recovered.shape[0] <= 20:
    div_cols = 3
    div_rows = np.int32(np.ceil(recovered.shape[0]/div_cols))
    plt.figure(figsize=(div_cols*12, div_rows*7))
    count = 1
    for r in range(div_rows):
        for c in range(div_cols):
            if count <= recovered.shape[0]:
                plt.subplot(div_rows, div_cols, count)
                plt.plot(wn, np.transpose(recovered[count-1, :]), 'r',
                         wn, np.transpose(no_bg_mss[count-1, :]), 'g',
                         wn, -np.transpose(ls_recovered[count-1, :]), 'r',
                         wn, -np.transpose(no_bg_mss[count-1, :]), 'g')
#                plt.title('group '+str(count), fontsize=12)
#                plt.text(290, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/9*8, 
#                         'group '+str(count), fontsize=fontsize_val, color='b')
                plt.xlabel('wavenumber (cm-1)', fontsize=fontsize_val)
                plt.ylabel('intensity (a.u.)', fontsize=fontsize_val)
                plt.legend(['recovered spectra', 'reference spectra'], fontsize=fontsize_val, loc=1)
#                plt.text(290, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3, 'CNN, absolute mse=%.2f'%cnn_mse[count-1], fontsize=fontsize_val)
#                plt.text(290, -np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3,'NNLS, absolute mse=%.2f'%ls_mse[count-1], fontsize=fontsize_val)
                plt.text(290, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3, 'CNN', fontsize=fontsize_val)
                plt.text(290, -np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3,'AsLS', fontsize=fontsize_val)
                plt.xlim((np.min(wn), np.max(wn)))
                plt.grid()
                plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
                plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
                count += 1
            
    plt.savefig('./output/prediction/compare.png', dpi=200)
    plt.show()
    
    plt.figure(figsize=(12, 12))
    
    plt.subplot(221)
    plt.plot(sub_real_coeffs_norm[:, 0], YPredict_norm[:, 0], 'ob',
             np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
    plt.ylabel('concentrations by CNN', fontsize=fontsize_val)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.title(names[0], fontsize=fontsize_val)
    plt.subplot(222)
    plt.plot(sub_real_coeffs_norm[:, 1], YPredict_norm[:, 1], 'ob',
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
    plt.plot(sub_real_coeffs_norm[:, 0], ls_coeffs_norm[:, 0], 'ob',
             np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
    plt.xlabel('true concentrations', fontsize=fontsize_val)
    plt.ylabel('concentrations by AsLS', fontsize=fontsize_val)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    plt.subplot(224)
    plt.plot(sub_real_coeffs_norm[:, 1], ls_coeffs_norm[:, 1], 'ob',
             np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g')
    plt.xlabel('true concentrations', fontsize=fontsize_val)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
    
    plt.savefig('./output/prediction/single_group.png', dpi=200)
    plt.show()
        
else:
    div_cols = len(exp)
    div_rows = np.int32(np.ceil(recovered.shape[0]/div_cols))
    plt.figure(figsize=(div_cols*10, div_rows*5))
    count = 1
    for r in range(div_rows): # group
        for c in range(div_cols):
            if count <= recovered.shape[0]:
                plt.subplot(div_rows, div_cols, count)
                plt.plot(wn, np.transpose(recovered[count-1, :]), 'r',
                         wn, np.transpose(no_bg_mss[count-1, :]), 'g',
                         wn, -np.transpose(ls_recovered[count-1, :]), 'r',
                         wn, -np.transpose(no_bg_mss[count-1, :]), 'g')
#                plt.title('group '+str(count), fontsize=12)
#                plt.text(290, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/9*8, 
#                         'group %d'%(r+1)+', exp='+str(exp[c])+'s', fontsize=fontsize_val, color='b')
                plt.xlabel('wavenumber (cm-1)', fontsize=fontsize_val)
                plt.ylabel('intensity (a.u.)', fontsize=fontsize_val)
                plt.legend(['recovered spectra', 'reference spectra'], fontsize=fontsize_val, loc=1)
#                plt.text(290, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3, 'CNN, absolute mse=%.2f'%cnn_mse[count-1], fontsize=fontsize_val)
#                plt.text(290, -np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3,'AsLS, absolute mse=%.2f'%ls_mse[count-1], fontsize=fontsize_val)
                plt.xlim((np.min(wn), np.max(wn)))
                plt.grid()
                plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
                plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
                count += 1
            
    plt.savefig('./output/prediction/all_recovered.png', dpi=200)
    plt.show()
    ############################################################################
    x = np.arange(len(exp))
    y = np.arange(YPredict.shape[0]/len(exp)) + 1
    X, Y = np.meshgrid(x, y)
    
    cnn_z = np.reshape(YPredict, (X.shape[0], X.shape[1], YPredict.shape[1]))
    ls_z = np.reshape(ls_coeffs, (X.shape[0], X.shape[1], ls_coeffs.shape[1]))

    cnn_coeff_mse_z = np.reshape(cnn_coeff_mse, (X.shape[0], X.shape[1]))
    ls_coeff_mse_z = np.reshape(ls_coeff_mse, (X.shape[0], X.shape[1]))
    
    fig = plt.figure(figsize=(20, 10)) 
    ax=fig.add_subplot(221, projection='3d')
    ax.plot_surface(X, Y, cnn_z[:, :, 0],
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=fontsize_val)
    plt.ylabel('mixture group', fontsize=fontsize_val)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=10)
    plt.setp(plt.gca().get_yticklabels(), fontsize=10)
    plt.setp(plt.gca().get_zticklabels(), fontsize=10)
    plt.title('predicted methanol concentrations by CNN', fontsize=fontsize_val)
    
    ax=fig.add_subplot(222, projection='3d')
    ax.plot_surface(X, Y, ls_z[:, :, 0],
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=fontsize_val)
    plt.ylabel('mixture group', fontsize=fontsize_val)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=10)
    plt.setp(plt.gca().get_yticklabels(), fontsize=10)
    plt.setp(plt.gca().get_zticklabels(), fontsize=10)
    plt.title('predicted methanol concentrations by AsLS', fontsize=fontsize_val)
    
    ax=fig.add_subplot(223, projection='3d')
    ax.plot_surface(X, Y, cnn_z[:, :, 1],
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=fontsize_val)
    plt.ylabel('mixture group', fontsize=fontsize_val)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=10)
    plt.setp(plt.gca().get_yticklabels(), fontsize=10)
    plt.setp(plt.gca().get_zticklabels(), fontsize=10)
    plt.title('predicted ethanol concentrations by CNN', fontsize=fontsize_val)
    
    ax=fig.add_subplot(224, projection='3d')
    ax.plot_surface(X, Y, ls_z[:, :, 1],
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=fontsize_val)
    plt.ylabel('mixture group', fontsize=fontsize_val)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=10)
    plt.setp(plt.gca().get_yticklabels(), fontsize=10)
    plt.setp(plt.gca().get_zticklabels(), fontsize=10)
    plt.title('predicted ethanol concentrations by AsLS', fontsize=fontsize_val)
    plt.savefig('./output/prediction/surf_concentration.png', dpi=200)
    plt.show()
    ###########################################################################
    #plot coeff mse
    fig = plt.figure(figsize=(20, 10)) 
    ax=fig.add_subplot(221, projection='3d')
    ax.plot_surface(X, Y, cnn_coeff_mse_z,
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=fontsize_val)
    plt.ylabel('mixture group', fontsize=fontsize_val)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=10)
    plt.setp(plt.gca().get_yticklabels(), fontsize=10)
    plt.setp(plt.gca().get_zticklabels(), fontsize=10)
    plt.title('mse of relative concentrations by CNN', fontsize=fontsize_val)
    
    ax=fig.add_subplot(222, projection='3d')
    ax.plot_surface(X, Y, ls_coeff_mse_z,
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=fontsize_val)
    plt.ylabel('mixture group', fontsize=fontsize_val)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=10)
    plt.setp(plt.gca().get_yticklabels(), fontsize=10)
    plt.setp(plt.gca().get_zticklabels(), fontsize=10)
    plt.title('mse of relative concentrations by AsLS', fontsize=fontsize_val)
    
    ax=fig.add_subplot(223, projection='3d')
    ax.plot_surface(X, Y, ls_coeff_mse_z-cnn_coeff_mse_z,
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=fontsize_val)
    plt.ylabel('mixture group', fontsize=fontsize_val)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.grid()
    plt.setp(plt.gca().get_xticklabels(), fontsize=10)
    plt.setp(plt.gca().get_yticklabels(), fontsize=10)
    plt.setp(plt.gca().get_zticklabels(), fontsize=10)
    plt.title('AsLS - CNN', fontsize=fontsize_val)
    
    plt.savefig('./output/prediction/surf_coeff_mse.png', dpi=200)
    plt.show()
    ############################################################################
# In[] print time cost
print('total time cost of cnn is', (cnn_normalize+cnn_predict), 's')
print('total time cost of asls is', (asls_smooth+asls_bg_cor+asls_predict), 's')
print('mean minima is', np.min(np.mean(mixture, axis=1)))

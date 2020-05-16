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
import os

# Set default decvice: GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# In[] read pure_spectra and mixture, then smoothing
data = np.load('./spectra_data/my_measurement.npz')
met = data['methanol']
eth = data['ethanol']
ace = data['acetonitrile']
qtz = data['quartz']
wn = data['wn'][:, 0]
names = data['pure_names']
exp = data['exposure']


is_sum_operation = True
if is_sum_operation:
    # repeats: 10 -> exposures: 11 -> groups: 13
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
# In[] smoothing
me_mss = np.transpose(subpys.whittaker_smooth(spectra=np.transpose(me_mixture), lmbda=0.5, d=2))
mea_mss = np.transpose(subpys.whittaker_smooth(spectra=np.transpose(mea_mixture), lmbda=0.5, d=2))

show_choice = -2
plt.figure(figsize=(10, 6))
plt.plot(np.transpose(wn), np.transpose(me_mixture[show_choice, :]),
         np.transpose(wn), np.transpose(me_mss[show_choice, :]))
plt.title('smoothed spectra')
plt.xlabel('wavenumber (cm-1)')
plt.ylabel('intensity (a.u.)')
plt.xlim((np.min(wn), np.max(wn)))
plt.show()
###########################################################################
#mixture = me_mixture[np.arange(8, me_mixture.shape[0], 11), :]
#mss = me_mss[np.arange(8, me_mss.shape[0], 11), :]

#mixture = mea_mixture[np.arange(9, mea_mixture.shape[0], 11), :]
#mss = mea_mss[np.arange(9, mea_mss.shape[0], 11), :]

mixture = mea_mixture
mss = mea_mss

#mixture = np.concatenate((me_mixture, mea_mixture))
#mss = np.concatenate((me_mss, mea_mss))
# In[] normalize
#mixture = mss
saved_net_path = 'c2_v1'
X_mean = np.load('./RamanNet/'+saved_net_path+'/X_scale_mean.npy')
X_std = np.load('./RamanNet/'+saved_net_path+'/X_scale_std.npy')
X = (mixture - X_mean)/X_std

#X_scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(mixture)
#X = X_scaler.transform(mixture)
# In[] load trained model and predict
model = keras.models.load_model('./RamanNet/'+saved_net_path+'/regression_model.h5')
YPredict = model.predict(np.reshape(X, (X.shape[0], X.shape[1], 1)))
# In[] LS to remove backgrounds
no_bg_mss = np.zeros(mss.shape)
ls_coeffs = np.zeros((YPredict.shape[0], pures.shape[0]))
poly_bg = np.concatenate((qtz, subpys.myploy(3, pures.shape[1])))
#poly_pures = pures
for ij in range(ls_coeffs.shape[0]):
    tmpCoeff = subpys.asls(poly_bg, np.reshape(mss[ij, :], (1, len(wn))), 0.01)
    no_bg_mss[ij, :] = mss[ij, :] - np.matmul(tmpCoeff, poly_bg)
#    print('spectra %d'%ij)
# In[] LS fitting to get concentrations
ls_coeffs = np.zeros((YPredict.shape[0], pures.shape[0]))
poly_pures = np.concatenate((pures, subpys.myploy(3, pures.shape[1])))
#poly_pures = pures
for ij in range(ls_coeffs.shape[0]):
    tmpCoeff = subpys.asls(poly_pures, np.reshape(no_bg_mss[ij, :], (1, len(wn))), 0.1)
#    [tmpCoeff, resnorm, residual] = subpys.lsqnonneg(poly_pures.T, no_bg_mss[ij, :])
    ls_coeffs[ij, :] = tmpCoeff[:pures.shape[0]]
# In[] plot pre-processing
YPredict[YPredict<0] = 0
ls_coeffs[ls_coeffs<0] = 0

recovered = np.matmul(YPredict[:, :2], pures[:2, :])
ls_recovered = np.matmul(ls_coeffs[:, :2], pures[:2, :])

cnn_mse = np.mean((recovered-no_bg_mss)**2, axis=1)
ls_mse = np.mean((ls_recovered-no_bg_mss)**2, axis=1)

sub_YPredict = YPredict[:, :2]
sub_ls_coeffs = ls_coeffs[:, :2]

mixture_norm = no_bg_mss/np.max(no_bg_mss)

recovered_norm = recovered/np.max(recovered)
YPredict_norm = sub_YPredict/np.transpose(np.tile(np.sum(sub_YPredict, axis=1), (sub_YPredict.shape[1], 1)))

ls_recovered_norm = ls_recovered/np.max(ls_recovered)
ls_coeffs_norm = sub_ls_coeffs/np.transpose(np.tile(np.sum(sub_ls_coeffs, axis=1), (sub_ls_coeffs.shape[1], 1)))
# In[] plot result of CNN
plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.plot(np.transpose(np.arange(sub_YPredict.shape[0])), sub_YPredict)
plt.title('concentrations of CNN', fontsize=15)
plt.xlabel('measurements', fontsize=15)
plt.ylabel('coefficients', fontsize=15)
plt.legend(names, loc=1)
plt.xlim((0, YPredict.shape[0]-1))
#plt.ylim((0, 1))

plt.subplot(223)
plt.plot(np.transpose(np.arange(YPredict_norm.shape[0])), YPredict_norm[:, :])
plt.title('relative concentrations  of CNN', fontsize=15)
plt.xlabel('measurements', fontsize=15)
plt.ylabel('coefficients', fontsize=15)
plt.xlim((0, YPredict_norm.shape[0]-1))
plt.legend(names, loc=1)
plt.ylim((0, 1))

plt.subplot(222)
plt.plot(wn, np.transpose(recovered),
         wn, -np.transpose(no_bg_mss), linewidth=0.5)
plt.title('recovered spectra and raw spectra  of CNN', fontsize=15)
plt.xlabel('wavenumber (cm-1)', fontsize=15)
plt.ylabel('intensity (a.u.)', fontsize=15)
plt.xlim((np.min(wn), np.max(wn)))
plt.ylim((-np.max([recovered, no_bg_mss]), np.max([recovered, no_bg_mss])))

plt.subplot(224)
plt.plot(wn, np.transpose(recovered_norm),
         wn, -np.transpose(mixture_norm), linewidth=0.5)
plt.title('normalized recovered spectra and raw spectra  of CNN', fontsize=15)
plt.xlabel('wavenumber (cm-1)', fontsize=15)
plt.ylabel('intensity (a.u.)', fontsize=15)
plt.xlim((np.min(wn), np.max(wn)))

plt.savefig('./output/prediction/output_cnn.png', dpi=200)
plt.show()
# In[] plot result of LS
plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.plot(np.transpose(np.arange(sub_ls_coeffs.shape[0])), sub_ls_coeffs)
plt.title('concentrations of AsLS', fontsize=15)
plt.xlabel('measurements', fontsize=15)
plt.ylabel('coefficients', fontsize=15)
plt.xlim((0, ls_coeffs.shape[0]-1))
plt.legend(names, loc=1)
#plt.ylim((0, 1))

plt.subplot(223)
plt.plot(np.transpose(np.arange(ls_coeffs_norm.shape[0])), ls_coeffs_norm[:, :])
plt.title('relative concentrations of AsLS', fontsize=15)
plt.xlabel('measurements', fontsize=15)
plt.ylabel('coefficients', fontsize=15)
plt.xlim((0, ls_coeffs_norm.shape[0]-1))
plt.legend(names, loc=1)
plt.ylim((0, 1))

plt.subplot(222)
plt.plot(wn, np.transpose(ls_recovered),
         wn, -np.transpose(no_bg_mss), linewidth=0.5)
plt.title('recovered spectra and raw spectra of AsLS', fontsize=15)
plt.xlabel('wavenumber (cm-1)', fontsize=15)
plt.ylabel('intensity (a.u.)', fontsize=15)
plt.xlim((np.min(wn), np.max(wn)))
plt.ylim((-np.max([ls_recovered, no_bg_mss]), np.max([ls_recovered, no_bg_mss])))

plt.subplot(224)
plt.plot(wn, np.transpose(ls_recovered_norm),
         wn, -np.transpose(mixture_norm), linewidth=0.5)
plt.title('normalized recovered spectra and raw spectra of AsLS', fontsize=15)
plt.xlabel('wavenumber (cm-1)', fontsize=15)
plt.ylabel('intensity (a.u.)', fontsize=15)
plt.xlim((np.min(wn), np.max(wn)))

plt.savefig('./output/prediction/output_ls.png', dpi=200)
plt.show()
# In[] 10*11*13 = 143*10 = 1430; 10*11*7 = 77*10 = 770;
if recovered.shape[0] == len(exp):
    div_cols = 3
    div_rows = np.int32(np.ceil(recovered.shape[0]/div_cols))
    plt.figure(figsize=(30, 20))
    count = 1
    for r in range(div_rows):
        for c in range(div_cols):
            if count <= recovered.shape[0]:
                plt.subplot(div_rows, div_cols, count)
                plt.plot(wn, np.transpose(recovered[count-1, :]), 'r',
                         wn, np.transpose(no_bg_mss[count-1, :]), 'g',
                         wn, -np.transpose(ls_recovered[count-1, :]), 'r',
                         wn, -np.transpose(no_bg_mss[count-1, :]), 'g')
    #            plt.title('group '+str(count), fontsize=12)
                plt.text(290, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/9*8, 
                         'group '+str(count), fontsize=15, color='b')
                plt.xlabel('wavenumber (cm-1)', fontsize=15)
                plt.ylabel('intensity (a.u.)', fontsize=15)
                plt.legend(['recovered spectra', 'reference spectra'], fontsize=15, loc=1)
                plt.text(290, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3, 'CNN, mse=%.2f'%cnn_mse[count-1], fontsize=15)
                plt.text(290, -np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3,'AsLS, mse=%.2f'%ls_mse[count-1], fontsize=15)
                plt.xlim((np.min(wn), np.max(wn)))
                count += 1
            
    plt.savefig('./output/prediction/compare.png', dpi=200)
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
    #            plt.title('group '+str(count), fontsize=12)
                plt.text(290, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/9*8, 
                         'group %d'%(r+1)+', exp='+str(exp[c])+'s', fontsize=15, color='b')
                plt.xlabel('wavenumber (cm-1)', fontsize=15)
                plt.ylabel('intensity (a.u.)', fontsize=15)
                plt.legend(['recovered spectra', 'reference spectra'], fontsize=15, loc=1)
                plt.text(290, np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3, 'CNN, mse=%.2f'%cnn_mse[count-1], fontsize=15)
                plt.text(290, -np.max([recovered[count-1, :], no_bg_mss[count-1, :]])/3,'AsLS, mse=%.2f'%ls_mse[count-1], fontsize=15)
                plt.xlim((np.min(wn), np.max(wn)))
                count += 1
            
    plt.savefig('./output/prediction/all_recovered.png', dpi=200)
    ############################################################################
    x = np.arange(len(exp))
    y = np.arange(YPredict.shape[0]/len(exp)) + 1
    X, Y = np.meshgrid(x, y)
    cnn_z = np.reshape(YPredict, (X.shape[0], X.shape[1], YPredict.shape[1]))
    ls_z = np.reshape(ls_coeffs, (X.shape[0], X.shape[1], ls_coeffs.shape[1]))
    cnn_mse_z = np.reshape(cnn_mse, (X.shape[0], X.shape[1]))
    ls_mse_z = np.reshape(ls_mse, (X.shape[0], X.shape[1]))
    
    fig = plt.figure(figsize=(20, 10)) 
    ax=fig.add_subplot(221, projection='3d')
    ax.plot_surface(X, Y, cnn_z[:, :, 0],
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=15)
    plt.ylabel('mixture group', fontsize=15)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.title('methanol concentrations by CNN', fontsize=15)
    
    ax=fig.add_subplot(222, projection='3d')
    ax.plot_surface(X, Y, ls_z[:, :, 0],
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=15)
    plt.ylabel('mixture group', fontsize=15)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.title('methanol concentrations by AsLS', fontsize=15)
    
    ax=fig.add_subplot(223, projection='3d')
    ax.plot_surface(X, Y, cnn_z[:, :, 1],
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=15)
    plt.ylabel('mixture group', fontsize=15)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.title('ethanol concentrations by CNN', fontsize=15)
    
    ax=fig.add_subplot(224, projection='3d')
    ax.plot_surface(X, Y, ls_z[:, :, 1],
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=15)
    plt.ylabel('mixture group', fontsize=15)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.title('ethanol concentrations by AsLS', fontsize=15)
    plt.savefig('./output/prediction/surf_concentration.png', dpi=200)
    ############################################################################
    fig = plt.figure(figsize=(20, 10)) 
    ax=fig.add_subplot(221, projection='3d')
    ax.plot_surface(X, Y, cnn_mse_z,
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=15)
    plt.ylabel('mixture group', fontsize=15)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.title('mse of CNN', fontsize=15)
    
    ax=fig.add_subplot(222, projection='3d')
    ax.plot_surface(X, Y, ls_mse_z,
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=15)
    plt.ylabel('mixture group', fontsize=15)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.title('mse of AsLS', fontsize=15)
    
    ax=fig.add_subplot(223, projection='3d')
    ax.plot_surface(X, Y, ls_mse_z-cnn_mse_z,
        rstride=1,  # rstride（row）
        cstride=1,  # cstride(column)
        cmap=plt.get_cmap('rainbow'))  
    plt.xlabel('exposure time (s)', fontsize=15)
    plt.ylabel('mixture group', fontsize=15)
    ax.set_xticks(np.arange(len(exp)))
    ax.set_xticklabels(exp)
    plt.title('AsLS - CNN', fontsize=15)
    
    plt.savefig('./output/prediction/surf_mse.png', dpi=200)
    ############################################################################
# In[]

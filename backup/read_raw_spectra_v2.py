# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:07:12 2019

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import spectra_process.subpys as subpys
# In[] read in plastics spectra to calibrate wavenumbers
home_path = './Spectra_data/11062019/'
pl = np.squeeze(subpys.spe_reader(home_path + 'plastics/' + 'plastics_2s.spe')['rawdata'])
pls = subpys.whittaker_smooth(np.reshape(pl, (pl.shape[0], 1)), 0.5, 2)


pixels = np.array([  276,   357,  458.7,    473,    539,    699,    792])-1
ref_wn = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1602.3])
wnn = np.polyval(np.polyfit(pixels, ref_wn, 2), np.arange(pl.shape[0]))
rr = np.argwhere((wnn>= 250) & (wnn<= 1750))
wn = wnn[rr]

plt.figure(figsize=(20, 10))
plt.subplot(211)
plt.plot(pixels, ref_wn, '*',
         np.arange(pl.shape[0]), wnn)
plt.xlabel('pixels', fontsize=12)
plt.ylabel('wavenumber', fontsize=12)
plt.xlim((np.min(np.arange(pl.shape[0])), np.max(np.arange(pl.shape[0]))))

plt.subplot(212)
plt.plot(wn, pl[rr],
         wn, pls[rr, 0])
plt.title('check smoothed polystyrene spectra', fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.xlim(np.min(wn), np.max(wn))
plt.savefig('./output/tmp/tmp_1.png', dpi=200)
plt.show()
# In[] read in dark spectra
dark_spectra = np.squeeze(subpys.spe_reader(home_path + 'plastics/' + 'bg_5s.spe')['rawdata'])
dark_spectra = dark_spectra[:, rr]
dark_std = np.std(dark_spectra) # need to save
dark_offset = np.mean(dark_spectra)
# In[] read in puare methonal, ethonal and acetonitrile
pure_names = np.array(['methanol', 'ethanol', 'acetonitrile', 'quartz'])


methanol = np.reshape(np.mean(np.squeeze(subpys.spe_reader(home_path + 'pures/' + pure_names[0] 
    + '_' + f'{5}' + 's.spe')['rawdata']), axis=0), (pl.shape[0], 1)) - dark_offset
methanol_ss = np.transpose(subpys.whittaker_smooth(methanol, 0.5, 2)[rr, 0])
methanol = np.transpose(methanol[rr, 0])

ethanol = np.reshape(np.mean(np.squeeze(subpys.spe_reader(home_path + 'pures/' + pure_names[1] 
    + '_' + f'{5}' + 's.spe')['rawdata']), axis=0), (pl.shape[0], 1)) - dark_offset
ethanol_ss = np.transpose(subpys.whittaker_smooth(ethanol, 0.5, 2)[rr, 0])
ethanol = np.transpose(ethanol[rr, 0])

acetonitrile = np.reshape(np.mean(np.squeeze(subpys.spe_reader(home_path + 'pures/' + pure_names[2] 
    + '_' + f'{5}' + 's.spe')['rawdata']), axis=0), (pl.shape[0], 1)) - dark_offset
acetonitrile_ss = np.transpose(subpys.whittaker_smooth(acetonitrile, 0.5, 2)[rr, 0])
acetonitrile = np.transpose(acetonitrile[rr, 0])

quartz = np.reshape(np.mean(np.squeeze(subpys.spe_reader(home_path + 'pures/' + pure_names[3] 
    + '_' + f'{5}' + 's.spe')['rawdata']), axis=0), (pl.shape[0], 1)) - dark_offset
quartz_ss = np.transpose(subpys.whittaker_smooth(quartz, 0.5, 2)[rr, 0])
quartz = np.transpose(quartz[rr, 0])

plt.figure(figsize=(20, 10))
plt.plot(wn, np.transpose(methanol),
         wn, np.transpose(ethanol),
         wn, np.transpose(acetonitrile),
         wn, np.transpose(quartz),
         wn, np.transpose(methanol_ss),
         wn, np.transpose(ethanol_ss),
         wn, np.transpose(acetonitrile_ss),
         wn, np.transpose(quartz_ss))

plt.title('check smoothed pure spectra', fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.legend(pure_names, fontsize=12)
plt.xlim(np.min(wn), np.max(wn))
plt.savefig('./output/tmp/tmp_2.png', dpi=200)
plt.show()
# In[] read in mixture of methonal and ethonal
exposure = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 5]
me_mixture = np.zeros((10, wn.shape[0], len(exposure), 13))
for group in range(13):
    for exp in range(len(exposure)):
        me_mixture[:, :, exp, group] = np.squeeze(np.squeeze(subpys.spe_reader(home_path + 'two-component/'
                  + f'{group+1}' + '_' + f'{exposure[exp]}' + 's.spe')['rawdata'])[:, rr]) - dark_offset
# In[] read in mixture of methonal, ethonal and acetonitrile
mea_mixture = np.zeros((10, wn.shape[0], len(exposure), 7))
for group in range(7):
    for exp in range(len(exposure)):
        mea_mixture[:, :, exp, group] = np.squeeze(np.squeeze(subpys.spe_reader(home_path + 'three-component/' 
                   + f'{group+1}' + '_' + f'{exposure[exp]}' + 's.spe')['rawdata'])[:, rr]) - dark_offset
# In[] plot all spectra
plt.figure(figsize=(10, 5))
plt.subplot(211)
plt.plot(wn, me_mixture[0, :, :, 0])
plt.title('raw mixed spectra of two components', fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.xlim(np.min(wn), np.max(wn))
plt.ylim(0, )
           
plt.subplot(212)
plt.plot(wn, mea_mixture[0, :, :, 0])
plt.title('raw mixed spectra of three components', fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.xlim(np.min(wn), np.max(wn))
plt.ylim(0, )
plt.savefig('./output/tmp/tmp_3.png', dpi=200)
plt.show()
print(np.min(me_mixture))
print(np.min(mea_mixture))
# In[]  AsLS fitting
p = np.concatenate((subpys.myploy(3, wn.shape[0]), quartz_ss-100), axis=0)

x = subpys.asls(p, methanol_ss, 0.01)
me_bg = np.matmul(x.T, p)

x = subpys.asls(p, ethanol_ss, 0.01)
et_bg = np.matmul(x.T, p)

x = subpys.asls(p, acetonitrile_ss, 0.01)
ac_bg = np.matmul(x.T, p)

methanol_ss_nobg = methanol_ss - me_bg
ethanol_ss_nobg = ethanol_ss - et_bg
acetonitrile_ss_nobg = acetonitrile_ss - ac_bg

x = subpys.asls(p[:-1, :], quartz_ss, 0.01)
qtz_bg = np.matmul(x.T, p[:-1, :])
quartz_ss_nobg = quartz_ss - qtz_bg


plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.plot(wn, np.transpose(methanol_ss_nobg),
         wn, np.transpose(methanol_ss),
         wn, np.transpose(me_bg))
plt.title('baseline correction of ' + pure_names[0], fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.xlim(np.min(wn), np.max(wn))

plt.subplot(222)
plt.plot( wn, np.transpose(ethanol_ss_nobg),
         wn, np.transpose(ethanol_ss),
         wn, np.transpose(et_bg))

plt.title('baseline correction of ' + pure_names[1], fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.xlim(np.min(wn), np.max(wn))

plt.subplot(223)
plt.plot(wn, np.transpose(acetonitrile_ss_nobg),
         wn, np.transpose(acetonitrile_ss),
         wn, np.transpose(ac_bg))

plt.title('baseline correction of ' + pure_names[2], fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.xlim(np.min(wn), np.max(wn))

plt.subplot(224)
plt.plot(wn, np.transpose(quartz_ss_nobg),
         wn, np.transpose(quartz_ss),
         wn, np.transpose(qtz_bg))
plt.title('baseline correction of ' + pure_names[3], fontsize=12)
plt.xlabel('wavenumber (cm-1)', fontsize=12)
plt.ylabel('intensity (a.u.)', fontsize=12)
plt.xlim(np.min(wn), np.max(wn))
plt.savefig('./output/tmp/tmp_4.png', dpi=300)
plt.show()

# In[] save to npz file

np.savez('./Spectra_data/my_measurement', wn=wn, dark_std=dark_std, dark_offset=dark_offset,
         exposure=exposure, pure_names=pure_names, methanol=methanol_ss_nobg, ethanol=ethanol_ss_nobg, 
         acetonitrile=acetonitrile_ss_nobg, quartz=quartz_ss_nobg, me_mixture=me_mixture, 
         mea_mixture=mea_mixture) 
# In[]


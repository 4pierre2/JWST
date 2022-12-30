#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:58:29 2022

@author: pierre
"""
import numpy as np
from ifuobj import sinfobj
from astropy.io import fits
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import median_filter, gaussian_filter, minimum_filter, maximum_filter, generic_filter
from scipy.signal import argrelextrema
import astropy.units as u
import astropy.constants as c
from astropy.modeling import models
# from specutils import Spectrum1D, SpectralRegion
from copy import deepcopy
# from scipy.ndimage import median_filter, gaussian_filter, minimum_filter, maximum_filter
#%%

filenames = glob.glob('./NGC7469/*/*3d.fits')
filenames = glob.glob('./NGC7469/*3d.fits')

objs = []
for filename in filenames:
    hdu = fits.open(filename)
    obj = sinfobj(hdu, z=0.016565)
    objs.append(obj)
    print(hdu[1].header['CDELT1'])
# lines = []
# for obj in tqdm(objs):
#     lines_t = obj.detect_lines_new(step=1e-2, threshold=15)
#     lines.append(lines_t[0])
    
# lines = np.concatenate(lines)

# #%%

# for l in lines:
#     for obj in objs:
#         lam_t = obj.lam
#         if (lam_t[0] < l < lam_t[-1]) and not (6.3 < l < 6.33):
#             name = str(int(l*1000))
#             a, b, c, d, mask = obj.fit_all_gauss(l, l-1e-2, l+1e-2)
#             amap = median_filter(a, (3,3))
#             plt.figure()
#             plt.imshow(amap, norm=LogNorm())
#             plt.title(str(int(100*l)/100)+'$\mu m$ - Amplitude')
#             plt.savefig('./ANALYSIS/'+name+'_flux_AU.pdf')
#             b0 = np.median(np.unique(b))
#             v = 3e5*(b-b0)/b0
#             vmap = median_filter(v, (3,3))
#             plt.figure()
#             plt.imshow(vmap, vmin=-100, vmax=100)
#             plt.title(str(int(100*l)/100)+'$\mu m$ - LOSV')
#             plt.colorbar()
#             plt.savefig('./ANALYSIS/'+name+'_LOSV_kms.pdf')
#             plt.figure()
#             std = 1/(2*c)**0.5
#             v_std = std*3e5/b0
#             smap = median_filter(v_std, (3,3))
#             plt.imshow(smap)
#             plt.title(str(int(100*l)/100)+'$\mu m$ - LOSVD')
#             plt.colorbar()
#             plt.savefig('./ANALYSIS/'+name+'_LOSVD_kms.pdf')
    
# #%%
# l = 6.31
# name = str(int(l*1000))
# a, b, c, d, mask = obj.fit_all_gauss(l, l-1e-1, l+1e-1)
# amap = median_filter(a, (3,3))
# plt.figure()
# plt.imshow(amap, norm=LogNorm())
# plt.title(str(int(100*l)/100)+'$\mu m$ - Amplitude')
# plt.savefig('./ANALYSIS/'+name+'_flux_AU.pdf')
# b0 = np.median(np.unique(b))
# v = 3e5*(b-b0)/b0
# vmap = median_filter(v, (3,3))
# plt.figure()
# plt.imshow(vmap, vmin=-100, vmax=100)
# plt.title(str(int(100*l)/100)+'$\mu m$ - LOSV')
# plt.colorbar()
# plt.savefig('./ANALYSIS/'+name+'_LOSV_kms.pdf')
# plt.figure()
# std = 1/(2*c)**0.5
# v_std = std*3e5/b0
# smap = median_filter(v_std, (3,3))
# plt.imshow(smap)
# plt.title(str(int(100*l)/100)+'$\mu m$ - LOSVD')
# plt.colorbar()
# plt.savefig('./ANALYSIS/'+name+'_LOSVD_kms.pdf')

# #%%

# plt.figure()
# for obj in objs:
#     obj.recenter_on_max()
#     obj.plot_spec(-0.5,0.5,-0.5,0.5, newfig=False)
    

# plt.figure()
# for obj in objs:
#     obj.recenter_on_max()
#     obj.plot_spec(-1.5,-0.5,-2,-1, newfig=False)
    
    
# plt.figure()
# for obj in objs:
#     obj.recenter_on_max()
#     obj.plot_spec(-2.5,-1.5,-3,-2, newfig=False)
    
# plt.figure()
# for obj in objs:
#     obj.recenter_on_max()
#     obj.plot_spec(newfig=False)
    

#%%
k = 0
lams = []
specs_center = []
raw_conts = []
em_specs = []
objs_el = []
for n in [3,2,0,1]:
    obj = objs[n]
    obj.recenter_on_max()
    raw_cont = median_filter(obj.data, (100, 1, 1))
    obj_el = deepcopy(obj)
    obj_el.data = obj_el.data-raw_cont
    objs_el.append(obj_el)
    l, s = obj.get_spec(-2, 2, -2, 2)
    specs_center.append(s)
    lams.append(l)
    raw_cont = median_filter(s, 100)
    raw_conts.append(raw_cont)
    em_specs.append(s-raw_cont)
    plt.plot(l, em_specs[-1], label=str(k))
    # plt.plot(l, median_filter(np.max(obj.data, (1,2))[:-1], 3), label=str(k))
    k += 1
plt.legend()
lam_unique = np.concatenate(lams)
em_unique = np.concatenate(em_specs)
spec_unique = np.concatenate(specs_center)

lam_sorted = np.sort(np.unique(lam_unique))
spec_em_interp = np.interp(lam_sorted, lam_unique, em_unique)
spec_interp = np.interp(lam_sorted, lam_unique, spec_unique)

np.savetxt("./lam.txt", lam_sorted)
np.savetxt('./spec.txt', spec_interp)
np.savetxt('./spec_em.txt', spec_em_interp)

#%%
plt.figure()
plt.plot(lam_sorted, spec_interp)

#%%

# Detect lines
stds = []
for obj in objs_el:
    data = obj.data/np.mean(abs(obj.data))
    l = obj.lam
    std = generic_filter(data, np.std, (20, 1, 1))
    stds.append(std)
#%%
  
from scipy.interpolate import interp1d
lams_ems = [[12.366]]
for n in range(len(objs_el)):
    obj = objs_el[n]
    l = obj.lam
    lams_rebinned = np.arange(l[0]-1, l[-1]+1, np.mean(l[1:]-l[:-1]))
    lams_rebinner = interp1d(lams_rebinned, lams_rebinned, kind='nearest')
    data = obj.data/np.mean(abs(obj.data))
    std = stds[n]
    med_el = gaussian_filter(np.mean(data, (1, 2)), 3)
    plt.plot(l, med_el)
    med_std = median_filter(np.mean(std, (1, 2)), 100)
    plt.plot(l, med_std)
    args = argrelextrema(med_el, np.greater)
    lam_ems = l[args][med_el[args]>med_std[args]]
    lun = np.unique(lams_rebinner(lam_ems))
    lams_ems.append(lun)
    
lams_ems = np.concatenate(lams_ems)

for l in lams_ems:
    plt.plot([l, l], [0, 60])

#%%


import pickle
lines_cloudy = pickle.load(open('./lines_cloudy', 'rb')) 
infos_cloudy = pickle.load(open('./infos_cloudy', 'rb')) 

cl_lines = {}
cl_infos = {}
for lines in lines_cloudy:
    for line in lines:
        cl_lines[line[1][:-1]]=0
        cl_infos[line[1][:-1]]=line[:3]
for lines in lines_cloudy:
    for line in lines:
        cl_lines[line[1][:-1]] += 1#line[-1]
        
keys = list(cl_lines.keys())
keys_float = np.array(keys, dtype=float)
zipped_lists = zip(keys_float, keys)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
keys_float, keys = [ list(tuple) for tuple in  tuples]

lam_cl = keys_float
flu_cl = []
for key in keys:
    flu_cl.append(cl_lines[key])

def id_line(lam):
    tau = lam*100/3e5
    score = flu_cl*np.exp(-(abs(lam_cl-lam))/tau)
    args = np.argsort(score)[::-1]
    lines = []
    for arg in args[:5]:
        key = keys[arg]
        l = [*cl_infos[key], flu_cl[arg], lam_cl[arg]-lam, 3e5*(lam_cl[arg]-lam)/lam, np.log10(score[arg])]
        lines.append([*l])
    return lines

def analyse_id(lam, path):
    header = ["Element", "wavelength", "wavelength", "N", "Delta_lambda", "Delta_v", "score"]
    lines = [header, *id_line(lam)]
    with open(path+'/identification_line.txt', 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

for l in lams_ems:
    i = id_line(l)
    print(i[0][-1]) 
    
#%%

import os
def analyse(objs, lam, path='./ANALYSIS/'):
    
    for obj in objs:
        if obj.lam[0] < lam < obj.lam[-1]:
            break    
    lam = obj.get_lam(lam)
    im = obj.plot_im(lam-10*obj.pix_bw, lam+10*obj.pix_bw, kernel=(3,3))
    plt.close('all')
    lam_str = str(int(lam*10000))
    path = path+'/'+lam_str+'/'
    if not os.path.isdir(path):
        os.mkdir(path)
        
    obj.analyse_line_raw(lam, path)
    obj.analyse_line_gauss(lam, path)
    analyse_id(lam, path)
    return id_line(lam)[0], np.log10(np.sum(im))
    
ids = ["Lambda_obs : Element / Lambda_cloudy / Delta_v / Score / log_flux"]
for l in lams_ems:
    idl, fl = analyse(objs_el, l, path='./ANALYSIS/')
    ids.append(str(l)+' : '+idl[0]+' / '+idl[1]+' / '+'{:.1f}'.format(idl[-2])+' / '+'{:.1f}'.format(idl[-1])+' / '+'{:.1f}'.format(fl))

with open('./ANALYSIS/identification_lines.txt', 'w') as f:
    for line in ids:
        f.write(f"{line}\n")

#%%

im_HI_745 = abs(median_filter(np.loadtxt('./ANALYSIS/74576/im_raw.txt'), (1,1)))
im_HI_1704 = abs(median_filter(np.loadtxt('./ANALYSIS/170446/im_raw.txt'), (1,1)))
im_HI_1236 = abs(median_filter(np.loadtxt('./ANALYSIS/123681/im_gauss.txt'), (1,1)))
im_He2_522 = abs(median_filter(np.loadtxt('./ANALYSIS/52293/im_raw.txt'), (1,1)))/im_HI_745
im_Ar2_698 = abs(median_filter(np.loadtxt('./ANALYSIS/69839/im_raw.txt'), (1,1)))/im_HI_745
im_mg5_560 = abs(median_filter(np.loadtxt('./ANALYSIS/56064/im_raw.txt'), (1,1)))/im_HI_745
im_mg7_551 = abs(median_filter(np.loadtxt('./ANALYSIS/55100/im_raw.txt'), (1,1)))/im_HI_745

#%%
lines_fls = []
n = 0
for lines in tqdm(lines_cloudy):
    infos = infos_cloudy[n]
    
    h1 = 0
    he2 = 0
    for line in lines:
        lam = line[2]
        if int(lam*1e9) == 7457:
            h1 = line[-2]
    ar2, mg5, mg7 = 0, 0, 0
    if h1>0:
        for line in lines:
            lam = line[2]
            if int(lam*1e9) == 5226:
                he2 = line[-2]/h1
            if int(lam*1e9) == 6983:
                ar2 = line[-2]/h1
            if int(lam*1e9) == 5607:
                mg5 = line[-2]/h1
            if int(lam*1e9) == 5501:
                mg7 = line[-2]/h1
        # print(he2, ar2, mg5, mg7)
    line_fls = [he2, ar2, mg5, mg7]
    lines_fls.append(line_fls)
    n += 1
    
lines_fls = np.array(lines_fls)
ims = np.array([im_He2_522, im_Ar2_698, im_mg5_560, im_mg7_551])

indices = np.argmin(np.sum((lines_fls[:, :, None, None]-ims[None, :, :, :])**2, 1), 0)

shape = np.shape(im_HI_745)
t0s = np.zeros(shape)
i1s = t0s.copy()
t1s = t0s.copy()
ns = t0s.copy()

for i in range(shape[0]):
    for j in range(shape[1]):
        index = indices[i, j]
        infos = infos_cloudy[index]
        t0 = float(infos[-4])
        t0s[i, j] = t0
        i1 = float(infos[-3])
        i1s[i, j] = i1
        t1 = float(infos[-2])
        t1s[i, j] = t1
        n = float(infos[-1])
        ns[i, j] = n

ra0 = np.min(objs_el[0].RA)
ra1 = np.max(objs_el[0].RA)
dec0 = np.min(objs_el[0].dec)
dec1 = np.max(objs_el[0].dec)
lam0 = np.min(objs_el[0].lam)
lam1 = np.max(objs_el[0].lam)

path = './ANALYSIS/CLOUDY/MODEL_LOW_L/'

plt.figure()
plt.imshow(median_filter(t0s, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower')
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('$T_{UV}$ (log(K))')
plt.savefig(path+'t0s.png')
plt.savefig(path+'t0s.pdf')
np.savetxt(path+'t0s.txt', t0s)

plt.figure()
plt.imshow(median_filter(i1s, (3,3))-4, extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower')
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('Relative X-ray flux (log(UV/X))')
plt.savefig(path+'i1s.png')
plt.savefig(path+'i1s.pdf')
np.savetxt(path+'i1s.txt', i1s)

plt.figure()
plt.imshow(median_filter(t1s, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower')
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('$T_{X}$ (log(K))')
plt.savefig(path+'t1s.png')
plt.savefig(path+'t1s.pdf')
np.savetxt(path+'t1s.txt', t1s)

plt.figure()
plt.imshow(median_filter(ns, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower', vmin=0, vmax=2.5)
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('Density (log($cm^{-3}$))')
plt.savefig(path+'ns.png')
plt.savefig(path+'ns.pdf')
np.savetxt(path+'ns.txt', ns)


#%%

im_Ne2_1281 = abs(median_filter(np.loadtxt('./ANALYSIS/128093/im_raw.txt'), (1,1)))
im_Ne5_1427 = abs(median_filter(np.loadtxt('./ANALYSIS/142676/im_raw.txt'), (1,1)))
im_Ne5_1432 = abs(median_filter(np.loadtxt('./ANALYSIS/143192/im_raw.txt'), (1,1)))

rat_1 = np.nan_to_num(median_filter(im_Ne5_1432+im_Ne5_1427, (3,3))/median_filter(im_Ne2_1281, (3,3))).flatten()
rat_good_1 = rat_1[(rat_1<0.1)*(rat_1>0.001)]
rat_2 = np.nan_to_num(median_filter(im_Ne5_1432, (3,3))/median_filter(im_Ne2_1281, (3,3))).flatten()
rat_good_2 = rat_2[(rat_2<0.1)*(rat_2>0.001)]
rat_3 = np.nan_to_num(median_filter(im_Ne5_1427, (3,3))/median_filter(im_Ne2_1281, (3,3))).flatten()
rat_good_3 = rat_3[(rat_3<0.1)*(rat_3>0.001)]

lines_fls = []
infos_cl2 = []
n = 0
for lines in tqdm(lines_cloudy):
    infos = infos_cloudy[n]
    
    ne2 = 0
    ne5 = 0
    cl2 = 0
    for line in lines:
        lam = line[2]
        if int(lam*1e9) == 14322:
            ne5 = line[-1]
        if int(lam*1e9) == 12810:
            ne2 = line[-1]
        if int(lam*1e9) == 14363:
            cl2 = line[-1]
    if ne2: 
        if 0.006 < (cl2/ne2) < 0.013:
            infos_cl2.append(infos)
    line_fls = [ne2, ne5, cl2]
    lines_fls.append(line_fls)
    n += 1
    
lines_fls = np.array(lines_fls)
rat_mod = lines_fls[:,1]/lines_fls[:,0]
rat_mod_good = rat_mod[(rat_mod<0.1)*(rat_mod>0.001)]


rat_mod_cl = lines_fls[:,2]/lines_fls[:,0]
rat_mod_good_cl = rat_mod_cl[(rat_mod_cl<0.1)*(rat_mod_cl>0.001)]

#%%

plt.figure()
plt.hist(rat_good_1, 100, density=True, alpha=0.5, label='Obs 1428+1432')
plt.hist(rat_good_2, 100, density=True, alpha=0.5, label='Obs 1432')
plt.hist(rat_good_3, 100, density=True, alpha=0.5, label='Obs 1428')
plt.hist(rat_mod_good, 100, density=True, alpha=0.5, label='CLOUDY')
plt.legend()

plt.xlabel('Emission line ratio: Ne 5/Ne 2')
plt.ylabel('Number density')

#%%

plt.figure()
plt.hist(rat_good_1, 100, density=True, alpha=0.5, label='Obs 1428+1432')
plt.hist(rat_good_2, 100, density=True, alpha=0.5, label='Obs 1432')
plt.hist(rat_good_3, 100, density=True, alpha=0.5, label='Obs 1428')
plt.hist(rat_mod_good_cl, 100, density=True, alpha=0.5, label='CLOUDY')
plt.legend()

plt.xlabel('Emission line ratio: Cl 2/Ne 2')
plt.ylabel('Number density')

#%%

t0s = []
i1s = []
t1s = []
ns = []

for infos in infos_cl2:
    t0 = float(infos[-4])
    t0s.append(t0)
    i1 = float(infos[-3])
    i1s.append(i1)
    t1 = float(infos[-2])
    t1s.append(t1)
    n = float(infos[-1])
    ns.append(n)

plt.figure()
plt.hist(t0s)
plt.xlabel('$T_{UV}$ (log(K))')

plt.figure()
plt.hist(t1s)
plt.xlabel('$T_{X}$ (log(K))')
    
plt.figure()
plt.hist(i1s)
plt.xlabel('Relative X-ray flux (log(UV/X))')

plt.figure()
plt.hist(ns)
plt.xlabel('Density (log($cm^{-3}$))')

#%%

import pickle
lines_cloudy = pickle.load(open('./lines_cloudy_gen', 'rb')) 
infos_cloudy = pickle.load(open('./infos_cloudy_gen', 'rb')) 

cl_lines = {}
cl_infos = {}
for lines in lines_cloudy:
    for line in lines:
        cl_lines[line[1][:-1]]=0
        cl_infos[line[1][:-1]]=line[:3]
for lines in lines_cloudy:
    for line in lines:
        cl_lines[line[1][:-1]] += 1#line[-1]
        
keys = list(cl_lines.keys())
keys_float = np.array(keys, dtype=float)
zipped_lists = zip(keys_float, keys)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
keys_float, keys = [ list(tuple) for tuple in  tuples]

lam_cl = keys_float
flu_cl = []
for key in keys:
    flu_cl.append(cl_lines[key])


#%%
lines_fls = []
n = 0
for lines in tqdm(lines_cloudy):
    infos = infos_cloudy[n]
    
    h1 = 0
    he2 = 0
    for line in lines:
        lam = line[2]
        if int(lam*1e9) == 7457:
            h1 = line[-2]
    ar2, mg5, mg7 = 0, 0, 0
    if h1>0:
        for line in lines:
            lam = line[2]
            if int(lam*1e9) == 5226:
                he2 = line[-2]/h1
            if int(lam*1e9) == 6983:
                ar2 = line[-2]/h1
            if int(lam*1e9) == 5607:
                mg5 = line[-2]/h1
            if int(lam*1e9) == 5501:
                mg7 = line[-2]/h1
        # print(he2, ar2, mg5, mg7)
    line_fls = [he2, ar2, mg5, mg7]
    lines_fls.append(line_fls)
    n += 1
    
lines_fls = np.array(lines_fls)
ims = np.array([im_He2_522, im_Ar2_698, im_mg5_560, im_mg7_551])

indices = np.argmin(np.sum((lines_fls[:, :, None, None]-ims[None, :, :, :])**2, 1), 0)

shape = np.shape(im_HI_745)
t0s = np.zeros(shape)
i1s = t0s.copy()
t1s = t0s.copy()
ns = t0s.copy()

for i in range(shape[0]):
    for j in range(shape[1]):
        index = indices[i, j]
        infos = infos_cloudy[index]
        t1 = float(infos[-2])
        t1s[i, j] = t1
        n = float(infos[-1])
        ns[i, j] = n

ra0 = np.min(objs_el[0].RA)
ra1 = np.max(objs_el[0].RA)
dec0 = np.min(objs_el[0].dec)
dec1 = np.max(objs_el[0].dec)
lam0 = np.min(objs_el[0].lam)
lam1 = np.max(objs_el[0].lam)

path = './ANALYSIS/CLOUDY/MODEL_GEN/'

plt.figure()
plt.imshow(median_filter(t1s, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower')
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('$T$ (log(K))')
plt.savefig(path+'t1s.png')
plt.savefig(path+'t1s.pdf')
np.savetxt(path+'t1s.txt', t1s)

plt.figure()
plt.imshow(median_filter(ns, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower', vmin=0, vmax=2.5)
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('Density (log($cm^{-3}$))')
plt.savefig(path+'ns.png')
plt.savefig(path+'ns.pdf')
np.savetxt(path+'ns.txt', ns)


#%%

import pickle
lines_cloudy = pickle.load(open('./lines_cloudy_coro', 'rb')) 
infos_cloudy = pickle.load(open('./infos_cloudy_coro', 'rb')) 

cl_lines = {}
cl_infos = {}
for lines in lines_cloudy:
    for line in lines:
        cl_lines[line[1][:-1]]=0
        cl_infos[line[1][:-1]]=line[:3]
for lines in lines_cloudy:
    for line in lines:
        cl_lines[line[1][:-1]] += 1#line[-1]
        
keys = list(cl_lines.keys())
keys_float = np.array(keys, dtype=float)
zipped_lists = zip(keys_float, keys)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
keys_float, keys = [ list(tuple) for tuple in  tuples]

lam_cl = keys_float
flu_cl = []
for key in keys:
    flu_cl.append(cl_lines[key])


#%%
lines_fls = []
n = 0
for lines in tqdm(lines_cloudy):
    infos = infos_cloudy[n]
    
    h1 = 0
    he2 = 0
    for line in lines:
        lam = line[2]
        if int(lam*1e9) == 7457:
            h1 = line[-2]
    ar2, mg5, mg7 = 0, 0, 0
    if h1>0:
        for line in lines:
            lam = line[2]
            if int(lam*1e9) == 5226:
                he2 = line[-2]/h1
            if int(lam*1e9) == 6983:
                ar2 = line[-2]/h1
            if int(lam*1e9) == 5607:
                mg5 = line[-2]/h1
            if int(lam*1e9) == 5501:
                mg7 = line[-2]/h1
        # print(he2, ar2, mg5, mg7)
    line_fls = [he2, ar2, mg5, mg7]
    lines_fls.append(line_fls)
    n += 1
    
lines_fls = np.array(lines_fls)
ims = np.array([im_He2_522, im_Ar2_698, im_mg5_560, im_mg7_551])

indices = np.argmin(np.sum((lines_fls[:, :, None, None]-ims[None, :, :, :])**2, 1), 0)

shape = np.shape(im_HI_745)
t0s = np.zeros(shape)
i1s = t0s.copy()
t1s = t0s.copy()
ns = t0s.copy()

for i in range(shape[0]):
    for j in range(shape[1]):
        index = indices[i, j]
        infos = infos_cloudy[index]
        t1 = float(infos[-2])
        t1s[i, j] = t1
        n = float(infos[-1])
        ns[i, j] = n

ra0 = np.min(objs_el[0].RA)
ra1 = np.max(objs_el[0].RA)
dec0 = np.min(objs_el[0].dec)
dec1 = np.max(objs_el[0].dec)
lam0 = np.min(objs_el[0].lam)
lam1 = np.max(objs_el[0].lam)

path = './ANALYSIS/CLOUDY/MODEL_CORONAL/'

plt.figure()
plt.imshow(median_filter(t1s, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower')
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('$T$ (log(K))')
plt.savefig(path+'t1s.png')
plt.savefig(path+'t1s.pdf')
np.savetxt(path+'t1s.txt', t1s)

plt.figure()
plt.imshow(median_filter(ns, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower', vmin=0, vmax=2.5)
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('Density (log($cm^{-3}$))')
plt.savefig(path+'ns.png')
plt.savefig(path+'ns.pdf')
np.savetxt(path+'ns.txt', ns)



#%%

import pickle
lines_cloudy = pickle.load(open('./lines_cloudy_consT', 'rb')) 
infos_cloudy = pickle.load(open('./infos_cloudy_consT', 'rb')) 

cl_lines = {}
cl_infos = {}
for lines in lines_cloudy:
    for line in lines:
        cl_lines[line[1][:-1]]=0
        cl_infos[line[1][:-1]]=line[:3]
for lines in lines_cloudy:
    for line in lines:
        cl_lines[line[1][:-1]] += 1#line[-1]
        
keys = list(cl_lines.keys())
keys_float = np.array(keys, dtype=float)
zipped_lists = zip(keys_float, keys)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
keys_float, keys = [ list(tuple) for tuple in  tuples]

lam_cl = keys_float
flu_cl = []
for key in keys:
    flu_cl.append(cl_lines[key])


#%%
lines_fls = []
n = 0
for lines in tqdm(lines_cloudy):
    infos = infos_cloudy[n]
    
    h1 = 0
    he2 = 0
    for line in lines:
        lam = line[2]
        if int(lam*1e9) == 7457:
            h1 = line[-2]
    ar2, mg5, mg7 = 0, 0, 0
    if h1>0:
        for line in lines:
            lam = line[2]
            if int(lam*1e9) == 5226:
                he2 = line[-2]/h1
            if int(lam*1e9) == 6983:
                ar2 = line[-2]/h1
            if int(lam*1e9) == 5607:
                mg5 = line[-2]/h1
            if int(lam*1e9) == 5501:
                mg7 = line[-2]/h1
        # print(he2, ar2, mg5, mg7)
    line_fls = [he2, ar2, mg5, mg7]
    lines_fls.append(line_fls)
    n += 1
    
lines_fls = np.array(lines_fls)
ims = np.array([im_He2_522, im_Ar2_698, im_mg5_560, im_mg7_551])

indices = np.argmin(np.sum((lines_fls[:, :, None, None]-ims[None, :, :, :])**2, 1), 0)

shape = np.shape(im_HI_745)
t0s = np.zeros(shape)
i1s = t0s.copy()
t1s = t0s.copy()
ns = t0s.copy()

for i in range(shape[0]):
    for j in range(shape[1]):
        index = indices[i, j]
        infos = infos_cloudy[index]
        t1 = float(infos[-2])
        t1s[i, j] = t1
        n = float(infos[-1])
        ns[i, j] = n

ra0 = np.min(objs_el[0].RA)
ra1 = np.max(objs_el[0].RA)
dec0 = np.min(objs_el[0].dec)
dec1 = np.max(objs_el[0].dec)
lam0 = np.min(objs_el[0].lam)
lam1 = np.max(objs_el[0].lam)

path = './ANALYSIS/CLOUDY/MODEL_CONSTANT_TEMPERATURE/'

plt.figure()
plt.imshow(median_filter(t1s, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower')
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('$T$ (log(K))')
plt.savefig(path+'t1s.png')
plt.savefig(path+'t1s.pdf')
np.savetxt(path+'t1s.txt', t1s)

plt.figure()
plt.imshow(median_filter(ns, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower', vmin=0, vmax=2.5)
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('Density (log($cm^{-3}$))')
plt.savefig(path+'ns.png')
plt.savefig(path+'ns.pdf')
np.savetxt(path+'ns.txt', ns)



#%%

import pickle
lines_cloudy = pickle.load(open('./lines_cloudy_consT2', 'rb')) 
infos_cloudy = pickle.load(open('./infos_cloudy_consT2', 'rb')) 

cl_lines = {}
cl_infos = {}
for lines in lines_cloudy:
    for line in lines:
        cl_lines[line[1][:-1]]=0
        cl_infos[line[1][:-1]]=line[:3]
for lines in lines_cloudy:
    for line in lines:
        cl_lines[line[1][:-1]] += 1#line[-1]
        
keys = list(cl_lines.keys())
keys_float = np.array(keys, dtype=float)
zipped_lists = zip(keys_float, keys)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
keys_float, keys = [ list(tuple) for tuple in  tuples]

lam_cl = keys_float
flu_cl = []
for key in keys:
    flu_cl.append(cl_lines[key])


#%%
lines_fls = []
n = 0
for lines in tqdm(lines_cloudy):
    infos = infos_cloudy[n]
    
    h1 = 0
    he2 = 0
    for line in lines:
        lam = line[2]
        if int(lam*1e9) == 7457:
            h1 = line[-2]
    ar2, mg5, mg7 = 0, 0, 0
    if h1>0:
        for line in lines:
            lam = line[2]
            if int(lam*1e9) == 5226:
                he2 = line[-2]/h1
            if int(lam*1e9) == 6983:
                ar2 = line[-2]/h1
            if int(lam*1e9) == 5607:
                mg5 = line[-2]/h1
            if int(lam*1e9) == 5501:
                mg7 = line[-2]/h1
        # print(he2, ar2, mg5, mg7)
    line_fls = [he2, ar2, mg5, mg7]
    lines_fls.append(line_fls)
    n += 1
    
lines_fls = np.array(lines_fls)
ims = np.array([im_He2_522, im_Ar2_698, im_mg5_560, im_mg7_551])

indices = np.argmin(np.sum((lines_fls[:, :, None, None]-ims[None, :, :, :])**2, 1), 0)

shape = np.shape(im_HI_745)
t0s = np.zeros(shape)
i1s = t0s.copy()
t1s = t0s.copy()
ns = t0s.copy()

for i in range(shape[0]):
    for j in range(shape[1]):
        index = indices[i, j]
        infos = infos_cloudy[index]
        t1 = float(infos[-2])
        t1s[i, j] = t1
        n = float(infos[-1])
        ns[i, j] = n

ra0 = np.min(objs_el[0].RA)
ra1 = np.max(objs_el[0].RA)
dec0 = np.min(objs_el[0].dec)
dec1 = np.max(objs_el[0].dec)
lam0 = np.min(objs_el[0].lam)
lam1 = np.max(objs_el[0].lam)

path = './ANALYSIS/CLOUDY/MODEL_CONSTANT_TEMPERATURE_HD/'

plt.figure()
plt.imshow(median_filter(t1s, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower')
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('$T$ (log(K))')
plt.savefig(path+'t1s.png')
plt.savefig(path+'t1s.pdf')
np.savetxt(path+'t1s.txt', t1s)

plt.figure()
plt.imshow(median_filter(ns, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower', vmin=0, vmax=2.5)
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('Density (log($cm^{-3}$))')
plt.savefig(path+'ns.png')
plt.savefig(path+'ns.pdf')
np.savetxt(path+'ns.txt', ns)


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

path = './ANALYSIS/55100/'

im = np.loadtxt(path+'im_raw.txt')
RA = np.loadtxt(path+'zinfos_RA.txt')
dec = np.loadtxt(path+'zinfos_dec.txt')

im_smooth = median_filter(im, (3, 3))
plt.imshow(im_smooth, extent=[np.max(RA), np.min(RA), np.min(dec), np.max(dec)], origin='lower')
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('Flux ($W.m^{-2}.arcsec^{-2}$)')

#%%

from scipy import interpolate

ra_745 = np.loadtxt('./ANALYSIS/74576/zinfos_RA.txt') 
dec_745 = np.loadtxt('./ANALYSIS/74576/zinfos_dec.txt') 
mesh_745 = np.meshgrid(ra_745, dec_745)
f_745 = interpolate.interp2d(ra_745, dec_745, median_filter(im_HI_745, (3,3)), kind='linear', fill_value=0)

ra_1704 = np.loadtxt('./ANALYSIS/170446/zinfos_RA.txt') 
dec_1704 = np.loadtxt('./ANALYSIS/170446/zinfos_dec.txt') 
mesh_1704 = np.meshgrid(ra_1704, dec_1704)
f_1704 = interpolate.interp2d(ra_1704, dec_1704, median_filter(im_HI_1704, (3,3)), kind='linear', fill_value=0)

ra_1236 = np.loadtxt('./ANALYSIS/123681/zinfos_RA.txt') 
dec_1236 = np.loadtxt('./ANALYSIS/123681/zinfos_dec.txt') 
mesh_1236 = np.meshgrid(ra_1236, dec_1236)
f_1236 = interpolate.interp2d(ra_1236, dec_1236, median_filter(im_HI_1236, (3,3)), kind='linear', fill_value=0)

im_745_1704 = f_745(ra_1704, dec_1704)
im_745_1236 = f_745(ra_1236, dec_1236)

th_r = 0.0245/0.00927

plt.figure(); plt.imshow(median_filter(im_745_1236, (3,3))/median_filter(im_HI_1236, (3,3))/th_r)

# ext = 

#%%

import pickle
lines_cloudy_z0 = pickle.load(open('./lines_cloudy_z0', 'rb')) 
infos_cloudy_z0 = pickle.load(open('./infos_cloudy_z0', 'rb')) 
lines_cloudy_z1 = pickle.load(open('./lines_cloudy_z1', 'rb')) 
infos_cloudy_z1 = pickle.load(open('./infos_cloudy_z1', 'rb')) 
lines_cloudy_z2 = pickle.load(open('./lines_cloudy_z2', 'rb')) 
infos_cloudy_z2 = pickle.load(open('./infos_cloudy_z2', 'rb')) 

cl_lines = {}
cl_infos = {}
for lines in lines_cloudy:
    for line in lines:
        cl_lines[line[1][:-1]]=0
        cl_infos[line[1][:-1]]=line[:3]
for lines in lines_cloudy:
    for line in lines:
        cl_lines[line[1][:-1]] += 1#line[-1]
        
keys = list(cl_lines.keys())
keys_float = np.array(keys, dtype=float)
zipped_lists = zip(keys_float, keys)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
keys_float, keys = [ list(tuple) for tuple in  tuples]

lam_cl = keys_float
flu_cl = []
for key in keys:
    flu_cl.append(cl_lines[key])

for info in infos_cloudy_z0:
    info.append('0')

for info in infos_cloudy_z1:
    info.append('1')

for info in infos_cloudy_z2:
    info.append('2')

lines_cloudy = np.concatenate([lines_cloudy_z0, lines_cloudy_z1, lines_cloudy_z2])
infos_cloudy = np.concatenate([infos_cloudy_z0, infos_cloudy_z1, infos_cloudy_z2])

#%%
lines_fls = []
n = 0
for lines in tqdm(lines_cloudy):
    infos = infos_cloudy[n]
    
    h1 = 0
    he2 = 0
    for line in lines:
        lam = line[2]
        if int(lam*1e9) == 7457:
            h1 = line[-2]
    ar2, mg5, mg7 = 0, 0, 0
    if h1>0:
        for line in lines:
            lam = line[2]
            if int(lam*1e9) == 5226:
                he2 = line[-2]/h1
            if int(lam*1e9) == 6983:
                ar2 = line[-2]/h1
            if int(lam*1e9) == 5607:
                mg5 = line[-2]/h1
            if int(lam*1e9) == 5501:
                mg7 = line[-2]/h1
        # print(he2, ar2, mg5, mg7)
    line_fls = [he2, ar2, mg5, mg7]
    lines_fls.append(line_fls)
    n += 1
    
lines_fls = np.array(lines_fls)
ims = np.array([im_He2_522, im_Ar2_698, im_mg5_560, im_mg7_551])

indices = np.argmin(np.sum((lines_fls[:, :, None, None]-ims[None, :, :, :])**2, 1), 0)

shape = np.shape(im_HI_745)
t0s = np.zeros(shape)
i1s = t0s.copy()
t1s = t0s.copy()
ns = t0s.copy()
zs = t0s.copy()

for i in range(shape[0]):
    for j in range(shape[1]):
        index = indices[i, j]
        infos = infos_cloudy[index]
        t1 = float(infos[-3])
        t1s[i, j] = t1
        n = float(infos[-2])
        ns[i, j] = n
        z = float(infos[-1])
        zs[i, j] = z

ra0 = np.min(objs_el[0].RA)
ra1 = np.max(objs_el[0].RA)
dec0 = np.min(objs_el[0].dec)
dec1 = np.max(objs_el[0].dec)
lam0 = np.min(objs_el[0].lam)
lam1 = np.max(objs_el[0].lam)

path = './ANALYSIS/CLOUDY/MODEL_Z/'

plt.figure()
plt.imshow(median_filter(t1s, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower')
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('$T$ (log(K))')
plt.savefig(path+'t1s.png')
plt.savefig(path+'t1s.pdf')
np.savetxt(path+'t1s.txt', t1s)

plt.figure()
plt.imshow(median_filter(ns, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower', vmin=0, vmax=2.5)
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('Density (log($cm^{-3}$))')
plt.savefig(path+'ns.png')
plt.savefig(path+'ns.pdf')
np.savetxt(path+'ns.txt', ns)

plt.figure()
plt.imshow(median_filter(zs, (3,3)), extent=[np.max([ra0,ra1]), np.min([ra0,ra1]), np.min([dec0, dec1]), np.max([dec0,dec1])], origin='lower', vmin=2, vmax=0)
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('Abundance (log(solar))')
plt.savefig(path+'zs.png')
plt.savefig(path+'zs.pdf')
np.savetxt(path+'zs.txt', ns)


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

path = './ANALYSIS/55100/'

im = np.loadtxt(path+'im_raw.txt')
RA = np.loadtxt(path+'zinfos_RA.txt')
dec = np.loadtxt(path+'zinfos_dec.txt')

im_smooth = median_filter(im, (3, 3))
plt.imshow(im_smooth, extent=[np.max(RA), np.min(RA), np.min(dec), np.max(dec)], origin='lower')
plt.xlabel('Relative Right Ascension (")')
plt.ylabel('Relative Declination (")')
cbar = plt.colorbar()
cbar.set_label('Flux ($W.m^{-2}.arcsec^{-2}$)')

#%%

from scipy import interpolate

ra_745 = np.loadtxt('./ANALYSIS/74576/zinfos_RA.txt') 
dec_745 = np.loadtxt('./ANALYSIS/74576/zinfos_dec.txt') 
mesh_745 = np.meshgrid(ra_745, dec_745)
f_745 = interpolate.interp2d(ra_745, dec_745, median_filter(im_HI_745, (3,3)), kind='linear', fill_value=0)

ra_1704 = np.loadtxt('./ANALYSIS/170446/zinfos_RA.txt') 
dec_1704 = np.loadtxt('./ANALYSIS/170446/zinfos_dec.txt') 
mesh_1704 = np.meshgrid(ra_1704, dec_1704)
f_1704 = interpolate.interp2d(ra_1704, dec_1704, median_filter(im_HI_1704, (3,3)), kind='linear', fill_value=0)

ra_1236 = np.loadtxt('./ANALYSIS/123681/zinfos_RA.txt') 
dec_1236 = np.loadtxt('./ANALYSIS/123681/zinfos_dec.txt') 
mesh_1236 = np.meshgrid(ra_1236, dec_1236)
f_1236 = interpolate.interp2d(ra_1236, dec_1236, median_filter(im_HI_1236, (3,3)), kind='linear', fill_value=0)

im_745_1704 = f_745(ra_1704, dec_1704)
im_745_1236 = f_745(ra_1236, dec_1236)

th_r = 0.0245/0.00927

plt.figure(); plt.imshow(median_filter(im_745_1236, (3,3))/median_filter(im_HI_1236, (3,3))/th_r)
plt.close("all")
# ext = 
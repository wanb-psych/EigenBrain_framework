import numpy as np
import matplotlib.pyplot as plt
import func_plot as fp
import scipy.stats as ss
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from neuromaps.datasets import fetch_annotation
import neuromaps.transforms as nt
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
import colors as clrs
import pickle

plt.rcParams["font.family"] = "Arial"

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

## HCP unrelated data group level
mmp_fsLR = np.loadtxt('../../src/fs_LR.64k.mmp_360.txt')
max_mode = 100
fc_eigen = np.loadtxt('../../results/hcp/fc_eigen_hcp_group.txt')
sc_eigen = np.loadtxt('../../results/hcp/sc_eigen_hcp_group.txt')
gd_eigen = np.loadtxt('../../results/hcp/gd_eigen_hcp_group.txt')
fc_align = np.load('../../results/hcp/fc_eigen_hcp_align.npy')
sc_align = np.load('../../results/hcp/sc_eigen_hcp_align.npy')
gd_align = np.load('../../results/hcp/gd_eigen_hcp_align.npy')
n_sub = fc_align.shape[0]


fig, ax = plt.subplots(figsize=(4,4))
tmp=np.concatenate((gd_eigen.T, sc_eigen.T, fc_eigen.T))
ax.imshow(np.corrcoef(tmp), vmin=-0.3, vmax=0.3, cmap='RdBu_r')
ax.set_xticks([0,100,200,300])
ax.set_yticks([0,100,200,300])
ax.set_xticklabels(['','','',''])
ax.set_yticklabels(['','','',''])
fig.tight_layout()
fig.savefig('../../figures/neuromap_hcp.png', transparent=True, dpi=300)

transmitter_name = ['5HT1a', '5HT1b', '5HT2a', '5HT4','5HT6','5HTT',
                    'D1', 'D2', 'DAT',
                    'NET', 'H3','a4b2', 'M1',
                    'VAChT', 'CB1', 'MOR',
                    'NMDA', 'mGluR5',
                    'GABA', 'KOR']

nmda_path = "/Users/wanb/neuromaps-data/annotations/galovic2021/MNI152/NMDA_ge179_29hc_galovic2021.nii"

map_vol = [
    fetch_annotation(source='savli2012', desc='way100635', space='MNI152', res='3mm'),
    fetch_annotation(source='gallezot2010', desc='p943', space='MNI152', res='1mm'),
    fetch_annotation(source='savli2012', desc='altanserin', space='MNI152', res='3mm'),
    fetch_annotation(source='beliveau2017', desc='sb207145', space='fsaverage', den='164k'),
    fetch_annotation(source='radnakrishnan2018', desc='gsk215083', space='MNI152', res='1mm'),
    fetch_annotation(source='savli2012', desc='dasb', space='MNI152', res='3mm'),
    fetch_annotation(source='kaller2017', desc='sch23390', space='MNI152', res='3mm'),
    fetch_annotation(source='jaworska2020', desc='fallypride', space='MNI152', res='1mm'),
    fetch_annotation(source='dukart2018', desc='fpcit', space='MNI152', res='3mm'),
    fetch_annotation(source='hesse2017', desc='methylreboxetine', space='MNI152', res='3mm'),
    fetch_annotation(source='gallezot2017', desc='gsk189254', space='MNI152', res='1mm'),
    fetch_annotation(source='hillmer2016', desc='flubatine', space='MNI152', res='1mm'),
    fetch_annotation(source='naganawa2020', desc='lsn3172176', space='MNI152', res='1mm'),
    fetch_annotation(source='aghourian2017', desc='feobv', space='MNI152', res='1mm'),
    fetch_annotation(source='normandin2015', desc='omar', space='MNI152', res='1mm'),
    fetch_annotation(source='kantonen2020', desc='carfentanil', space='MNI152', res='3mm'),
    nmda_path,
    fetch_annotation(source='smart2019', desc='abp688', space='MNI152', res='1mm'),
    fetch_annotation(source='norgaard2021', desc='flumazenil', space='fsaverage', res='164k'),
    fetch_annotation(source='vijay2018', desc='ly2795050', space='MNI152', res='2mm')
]

map_mmp = []
for i in range(len(transmitter_name)):   
    try:
        tmp = nt.mni152_to_fslr(map_vol[i], '32k')
    except:
        tmp = nt.fsaverage_to_fslr(map_vol[i], '32k')
  
    map = np.concatenate((tmp[0].agg_data(),tmp[1].agg_data()))
    map = ss.zscore([np.nanmean(map[mmp_fsLR==i+1]) for i in range(360)])
    np.savetxt('../../results/neuromap/mmp_'+transmitter_name[i]+'.txt', map)
    map_mmp.append(map.copy())
    
elbow_final=[]
elbow_final.append('neurotransmitter,modality,elbow,corr')
    
for trans in range(len(transmitter_name)):
    elbow_save=[]
    map = map_mmp[trans]   
    r2 = np.zeros((6,max_mode))
    w_tmp = ['GD','SC','FC', 'GD_FC','SC_FC','GD_SC_FC']
    stop = {}
    for j in range(max_mode):
        tmp=[sm.add_constant(ss.zscore(gd_eigen[:,:j+1], axis=0)), 
             sm.add_constant(ss.zscore(sc_eigen[:,:j+1], axis=0)), 
             sm.add_constant(ss.zscore(fc_eigen[:,:j+1], axis=0))]
        for i in range(3): 
            glm = OLS(map, tmp[i]).fit()
            r2[i,j]=glm.rsquared
            
    for i in range(3):
        kl_pred = KneeLocator(range(1,101), r2[i], S=6, curve="concave", direction="increasing")
        stop[w_tmp[i]] = kl_pred.knee
        if stop[w_tmp[i]] is None:
            stop[w_tmp[i]] = 100
        if stop[w_tmp[i]] <=20: # not robust estimate
            stop[w_tmp[i]] = 20
            
        glm = OLS(map, tmp[i][:,:stop[w_tmp[i]]+1]).fit()
        
        with open(f'../../results/hcp/glm_{transmitter_name[trans]}_{w_tmp[i]}.pkl', 'wb') as f:
            pickle.dump(glm, f)
        
        #print(transmitter_name[trans] + ' ' + w_tmp[i] + ' : ' +str(stop[w_tmp[i]]))
        elbow_save.append(transmitter_name[trans] + ',' + w_tmp[i] + ',' +str(stop[w_tmp[i]]))
    
    for j in range(max_mode):
        tmp=[sm.add_constant(np.concatenate((ss.zscore(gd_eigen[:,:stop['GD']], axis=0),
                                             ss.zscore(fc_eigen[:,:j+1], axis=0)),axis=1)),
             sm.add_constant(np.concatenate((ss.zscore(sc_eigen[:,:stop['SC']], axis=0),
                                             ss.zscore(fc_eigen[:,:j+1], axis=0)),axis=1)),
             sm.add_constant(np.concatenate((ss.zscore(gd_eigen[:,:stop['GD']], axis=0),
                                             ss.zscore(sc_eigen[:,:stop['SC']], axis=0),
                                             ss.zscore(fc_eigen[:,:j+1], axis=0)),axis=1))]
        for i in range(3): 
            glm = OLS(map, tmp[i]).fit()
            r2[i+3,j]=glm.rsquared
    
    for i in range(3):
        kl_pred = KneeLocator(range(1,101), r2[i+3], S=6, curve="concave", direction="increasing")
        stop[w_tmp[i+3]] = kl_pred.knee
        if stop[w_tmp[i+3]] is None:
            stop[w_tmp[i+3]] = 100
        if stop[w_tmp[i+3]] <=20: # not robust estimate
            stop[w_tmp[i+3]] = stop[w_tmp[2]] # use the FC elbow instead
        if i < 2:    
            glm = OLS(map, tmp[i][:,:stop[w_tmp[i]]+stop[w_tmp[i+3]]+1]).fit()
        else:
            glm = OLS(map, tmp[i][:,:stop[w_tmp[0]]+stop[w_tmp[1]]+stop[w_tmp[i+3]]+1]).fit()
        
        with open(f'../../results/hcp/glm_{transmitter_name[trans]}_{w_tmp[i+3]}.pkl', 'wb') as f:
            pickle.dump(glm, f)
        
        #print(transmitter_name[trans] + ' ' + w_tmp[i+3] + ' : ' +str(stop[w_tmp[i+3]]))
        elbow_save.append(transmitter_name[trans] + ',' + w_tmp[i+3] + ',' +str(stop[w_tmp[i+3]]))
                    
    np.savetxt(f'../../results/neuromap/r2_{transmitter_name[trans]}.txt', r2.T)
    
    corr=[]
    for i in range(6):
        with open(f'../../results/hcp/glm_{transmitter_name[trans]}_{w_tmp[i]}.pkl', 'rb') as f:
            glm = pickle.load(f)
        sub_pred = []
        sub_r = []
        for sub in range(n_sub):
            tmp=[sm.add_constant(ss.zscore(gd_align[sub][:,:stop['GD']], axis=0)), 
                 sm.add_constant(ss.zscore(sc_align[sub][:,:stop['SC']], axis=0)), 
                 sm.add_constant(ss.zscore(fc_align[sub][:,:stop['FC']], axis=0)),
                 sm.add_constant(np.concatenate((ss.zscore(gd_align[sub][:,:stop['GD']], axis=0),
                                                 ss.zscore(fc_align[sub][:,:stop['GD_FC']], axis=0)),axis=1)),
                 sm.add_constant(np.concatenate((ss.zscore(sc_align[sub][:,:stop['SC']], axis=0),
                                                 ss.zscore(fc_align[sub][:,:stop['SC_FC']], axis=0)),axis=1)),
                 sm.add_constant(np.concatenate((ss.zscore(gd_align[sub][:,:stop['GD']], axis=0),
                                                 ss.zscore(sc_align[sub][:,:stop['SC']], axis=0),
                                                 ss.zscore(fc_align[sub][:,:stop['GD_SC_FC']], axis=0)),axis=1))]
            y_pred = glm.predict(tmp[i])
            sub_pred.append(y_pred)
            sub_r.append(ss.pearsonr(map, y_pred))
        
        np.savetxt(f'../../results/hcp/hcp_pred_{transmitter_name[trans]}_{w_tmp[i]}.txt', np.array(sub_pred))
        np.savetxt(f'../../results/hcp/hcp_r_{transmitter_name[trans]}_{w_tmp[i]}.txt', np.array(sub_r))
    
        tmp = [map, np.array(sub_pred).mean(0)]
        corr.append(','+str(ss.pearsonr(map, np.array(sub_pred).mean(0))[0].round(5)))
        plot = [None]*len(tmp)
        for num in range(len(tmp)):
            plot[num] = mmp_fsLR.copy()
            for node in range(360):
                plot[num][mmp_fsLR==node+1] = tmp[num][node]
            plot[num][mmp_fsLR==0] = np.nan

        fp.plot_surface(data = plot, nan_color=(0.5,0.5,0.5,1), 
                size = (400, 300*len(plot)), layout_style = 'grid', 
                cmap = 'CMRmap', filename = f'../../figures/neuromap_{transmitter_name[trans]}_{w_tmp[i]}_mean.png',
                display=False)

        tmp = [np.array(sub_pred).std(0)/np.array(sub_pred).mean(0)]
        plot = [None]*len(tmp)
        for num in range(len(tmp)):
            plot[num] = mmp_fsLR.copy()
            for node in range(360):
                plot[num][mmp_fsLR==node+1] = tmp[num][node]
            plot[num][mmp_fsLR==0] = np.nan

        fp.plot_surface(data = plot, color_range=(-10,10),
                    size = (400, 300*len(plot)), layout_style = 'grid', 
                    cmap = 'Spectral_r', filename = f'../../figures/neuromap_{transmitter_name[trans]}_{w_tmp[i]}_cv.png',
                    display=False)
    
        print('finished', transmitter_name[trans])
        
    for x, y in zip(elbow_save, corr):
        print(x,y)
        elbow_final.append(x + y)
    
np.savetxt('../../results/neuromap/Elbows4ThreeModality.txt', elbow_final, fmt='%s')
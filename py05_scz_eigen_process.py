import numpy as np
import pandas as pd
import pickle
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import scipy.stats as ss
import nibabel as nib
import os
from brainspace.gradient import GradientMaps

## Load HCP data
hcp_fc = np.loadtxt('../../results/hcp/fc_eigen_hcp_group.txt')
hcp_gd = np.loadtxt('../../results/hcp/gd_eigen_hcp_group.txt')
hcp_sc = np.loadtxt('../../results/hcp/sc_eigen_hcp_group.txt')
mmp_fsLR = np.loadtxt('../../src/fs_LR.64k.mmp_360.txt')

sub_list = np.array(pd.read_csv('../../results/Geneva/demo_sort.csv')['ID'])
n_sub = len(sub_list)
"""
for ses in ['ses-T1','ses-T2']:
    for sub in sub_list:
        ts = nib.load(f'../../results/Geneva/connectome/{sub}_{ses}_surf-fsLR-32k_desc-timeseries_clean.shape.gii').agg_data()
        tmp = np.zeros((ts.shape[0], 360))
        for i in range(360):
            tmp[:,i] = np.nanmean(ts[:,mmp_fsLR==i+1], axis=1)
        fc = np.corrcoef(tmp.T)
        fc[fc>0.99999] = 1
        fc = np.arctanh(fc)
        fc[fc==np.inf] = 0
        np.savetxt(f'../../results/Geneva/connectome/fc_{sub}_{ses}_mmp.txt', fc)
        print('fc ...', sub)
""" 
embedding = 'dm'
para = pd.read_csv('../../results/neuromap/Elbows4ThreeModality.txt')
for ses in ['ses-T1','ses-T2']:
    """
    for i in range(n_sub):
        print('starting gd eigen ...', sub_list[i], ses)
        gd = np.delete(np.delete(nib.load(f'../../results/Geneva/connectome/{sub_list[i]}_{ses}_atlas-glasser-360_GD.shape.gii').agg_data(), 
                            [0,181], axis=0), [0,181], axis=1)
        gm_l = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
        gm_l.fit(gd[:180,:180],reference=hcp_gd[:180], sparsity=0.9)
        gm_r = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
        gm_r.fit(gd[180:,180:],reference=hcp_gd[180:], sparsity=0.9)

        gd_eigen = np.concatenate((gm_l.gradients_, gm_r.gradients_))
        for j in range(100):
            if  ss.pearsonr(gd_eigen[:180,j],gd_eigen[180:,j])[0] < 0:
                gd_eigen[:180,j]= -gd_eigen[:180,j]
        gd_eigen_align = np.concatenate((gm_l.aligned_, gm_r.aligned_))

        np.savetxt(f'../../results/Geneva/eigen/gd_eigen_{sub_list[i]}_{ses}_mmp_raw.txt', gd_eigen)
        np.savetxt(f'../../results/Geneva/eigen/gd_eigen_{sub_list[i]}_{ses}_mmp_hcp.txt', gd_eigen_align)
            
    for i in range(n_sub):
        print('starting fc eigen ...', sub_list[i], ses)
        fc = np.loadtxt(f'../../results/Geneva/connectome/fc_{sub_list[i]}_{ses}_mmp.txt')
        np.fill_diagonal(fc, 0)
        gm = GradientMaps(kernel='normalized_angle', approach=embedding, n_components=100, random_state=0, alignment='procrustes')
        gm.fit(fc, reference=hcp_fc, sparsity=0.9)
        np.savetxt(f'../../results/Geneva/eigen/fc_eigen_{sub_list[i]}_{ses}_mmp_raw.txt',gm.gradients_)
        np.savetxt(f'../../results/Geneva/eigen/fc_eigen_{sub_list[i]}_{ses}_mmp_hcp.txt',gm.aligned_)
    """
    ## model for predicting neurotransmitters
    qc = []
    qc.append('ID,neurotransmitter,modality,QChcp_gd_r,QCself_gd_r')    
    for i in range(n_sub):
        fc_eigen = np.loadtxt(f'../../results/Geneva/eigen/fc_eigen_{sub_list[i]}_{ses}_mmp_hcp.txt')
        gd_eigen = np.loadtxt(f'../../results/Geneva/eigen/gd_eigen_{sub_list[i]}_{ses}_mmp_hcp.txt')
        for trans in para['neurotransmitter'].unique():
            os.makedirs(f'../../results/Geneva/pred/{trans}', exist_ok=True)
            map = np.loadtxt(f'../../results/neuromap/mmp_{trans}.txt')    
            mod='GD_SC_FC'
            glm = pickle.load(open(f'../../results/hcp/glm_{trans}_{mod}.pkl', 'rb'))
            stop1 = int(para[(para['neurotransmitter']==trans) & (para['modality']=='GD')]['elbow'].iloc[0])
            stop2 = int(para[(para['neurotransmitter']==trans) & (para['modality']=='SC')]['elbow'].iloc[0])
            stop3 = int(para[(para['neurotransmitter']==trans) & (para['modality']==mod)]['elbow'].iloc[0])
            x = np.concatenate((ss.zscore(hcp_gd[:,:stop1], axis=0),
                                ss.zscore(hcp_sc[:,:stop2], axis=0),
                                ss.zscore(fc_eigen[:, :stop3], axis=0)),axis=1)
            sub_pred = glm.predict(sm.add_constant(x))
            np.savetxt(f'../../results/Geneva/pred/{trans}/{sub_list[i]}_{ses}_{trans}_{mod}_hcpGD_pred.txt', sub_pred)
            sub_r_hcp = ss.pearsonr(sub_pred, map)[0].round(5)
            
            ## Using self GD because of widespread cortical atrophy in SCZ
            x = np.concatenate((ss.zscore(gd_eigen[:,:stop1], axis=0),
                                ss.zscore(hcp_sc[:,:stop2], axis=0),
                                ss.zscore(fc_eigen[:, :stop3], axis=0)),axis=1)
            sub_pred = glm.predict(sm.add_constant(x))
            np.savetxt(f'../../results/Geneva/pred/{trans}/{sub_list[i]}_{ses}_{trans}_{mod}_selfGD_pred.txt', sub_pred)
            sub_r_self = ss.pearsonr(sub_pred, map)[0].round(5)
            
            qc.append(sub_list[i]+','+trans+','+mod+','+str(sub_r_hcp)+','+str(sub_r_self))
            
            print('Finished subject:', sub_list[i], 'neurotransmitter prediction:', trans)
    np.savetxt(f'../../results/Geneva/pred_qc_{ses}_neurotransmitter.txt', qc, fmt='%s')